# --- START OF FILE vector_store.py ---
"""Handles interaction with ChromaDB: embedding, adding data, and querying."""

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import time
import os

# Import necessary functions from other modules
from utils import log_message, chunk_text, generate_unique_id
from file_parsers import parse_pdf, parse_docx, parse_pptx, parse_data_file, parse_txt, parse_image
from web_crawler import crawl_url
from config import (
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP,
    MAX_CONTEXT_RESULTS,
    VECTOR_DB_PERSIST_PATH,
    ALL_SUPPORTED_TYPES,
    SUPPORTED_DATA_TYPES,
    SUPPORTED_TEXT_TYPES,
    SUPPORTED_IMAGE_TYPES # Added
)

# --- Embedding Function ---
# ... (keep as is) ...
@st.cache_resource
def initialize_embedding_function(api_key):
     """Creates the embedding function object, cached for the session."""
     if "embedding_func_initialized" in st.session_state and st.session_state.embedding_func_initialized:
         existing_func = st.session_state.get("embedding_func_instance")
         if existing_func: return existing_func

     log_message("Initializing embedding function...", "debug")
     if not api_key: log_message("API Key missing, cannot initialize embedding function.", "error"); return None
     try:
          func = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key, model_name=EMBEDDING_MODEL_NAME)
          st.session_state.embedding_func_initialized = True
          st.session_state.embedding_func_instance = func
          log_message(f"Embedding function initialized with model: {EMBEDDING_MODEL_NAME}", "info")
          return func
     except Exception as e:
          log_message(f"Failed to create embedding function: {e}", "error"); st.exception(e)
          st.session_state.embedding_func_initialized = False
          if "embedding_func_instance" in st.session_state: del st.session_state.embedding_func_instance
          return None


# --- Core Processing and Embedding Pipeline (MODIFIED STATUS LOGIC) ---
def process_and_embed(uploaded_files, collection, embedding_function):
    """
    Processes uploaded files:
    - Parses text/image files, extracts content, chunks, embeds, and stores in ChromaDB.
    - Calls specific parser for data files (CSV/XLSX/JSON) which imports them
      to a separate SQLite DB and returns only a marker segment for ChromaDB.
    - Crawls URLs found in text files.
    Uses UUIDs for chunk IDs. Updates chat_state["processed_files"] status based on
    DB import success (for data files) or ChromaDB add success (for others).

    Returns:
        bool: True if ChromaDB addition was ATTEMPTED successfully for at least one
              non-data segment/chunk, False otherwise. This indicates Chroma status.
              Data file status is tracked separately in chat_state.
    """
    if not uploaded_files: log_message("No files provided for processing.", "warning"); return False
    if collection is None: log_message("Cannot process: ChromaDB collection unavailable.", "error"); return False
    if embedding_function is None: log_message("Cannot process: Embedding function unavailable.", "error"); return False

    all_segments_for_chroma = []
    files_processed_this_run = set()
    processed_count = 0
    chroma_add_success_flag = False # Track if any non-data segment was added successfully

    chat_id = st.session_state.get("current_chat_id")
    chat_state = st.session_state.chats.get(chat_id)
    if not chat_state: log_message("Critical Error in process_and_embed: Cannot get chat state.", "error"); return False
    if "processed_files" not in chat_state: chat_state["processed_files"] = {}

    overall_start_time = time.time()
    log_message(f"--- Starting processing job for {len(uploaded_files)} files (Chat: {chat_id}) ---")

    parser_map = {
        'pdf': parse_pdf, 'docx': parse_docx, 'pptx': parse_pptx,
        'csv': parse_data_file, 'xlsx': parse_data_file, 'json': parse_data_file,
        'txt': parse_txt,
        'png': parse_image, 'jpg': parse_image, 'jpeg': parse_image,
    }
    data_file_extensions = tuple(f".{ext}" for ext in SUPPORTED_DATA_TYPES)

    for i, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name
        file_start_time = time.time()
        log_message(f"Starting file {i+1}/{len(uploaded_files)}: {filename}")

        # Initial status - will be updated by parser or later by chroma result (if not data file)
        chat_state["processed_files"][filename] = 'processing'
        files_processed_this_run.add(filename)
        is_data_file = filename.lower().endswith(data_file_extensions)

        file_content = uploaded_file.getvalue()
        file_type = filename.split('.')[-1].lower()
        parser_func = parser_map.get(file_type)

        if not parser_func:
            log_message(f"Unsupported file type: {filename}. Skipping.", "warning")
            chat_state["processed_files"][filename] = 'skipped'
            continue

        # --- File Parsing ---
        try:
            # parse_data_file updates chat_state["processed_files"] directly for success/failure
            parsed_segments, urls = parser_func(file_content, filename)

            if not parsed_segments:
                 log_message(f"No content segments extracted from {filename}.", "warning")
                 # If it wasn't a data file and no segments, mark skipped. Data file status set by parser.
                 if not is_data_file:
                      chat_state["processed_files"][filename] = 'skipped'
                 continue

            # --- Process Segments for Chroma ---
            file_specific_segments = []
            # URL Crawling (only for text-based types)
            if file_type in SUPPORTED_TEXT_TYPES and urls:
                 unique_urls = set(u for u in urls if u)
                 log_message(f"Found {len(unique_urls)} unique URLs in {filename}. Crawling...", "info")
                 for url in unique_urls:
                     crawled_text = crawl_url(url)
                     if crawled_text:
                          crawl_meta = {"source": filename, "crawled_url": url, "content_type": "crawled_web"}
                          parsed_segments.append((f"\n--- Content from {url} ---\n{crawled_text}\n--- End Content ---", crawl_meta))

            # Chunking Segments (Text/Image/Marker/OCR)
            for text_segment, segment_meta in parsed_segments:
                 if not text_segment or not isinstance(text_segment, str): continue
                 segment_content_type = segment_meta.get("content_type", "unknown")

                 # Data file markers - add as single segment
                 if segment_content_type.startswith("data_import"):
                     chunk_id = generate_unique_id()
                     file_specific_segments.append((chunk_id, text_segment, segment_meta))
                     continue

                 # Chunk other content
                 chunks = chunk_text(text_segment, CHUNK_SIZE, CHUNK_OVERLAP)
                 for chunk_index, chunk in enumerate(chunks):
                      chunk_id = generate_unique_id()
                      chunk_meta = segment_meta.copy()
                      chunk_meta["chunk_index"] = chunk_index; chunk_meta["segment_length"] = len(text_segment)
                      chunk_meta["chunk_in_segment"] = f"{chunk_index + 1}/{len(chunks)}"
                      chunk_meta.setdefault("source", filename); chunk_meta.setdefault("content_type", "unknown")
                      file_specific_segments.append((chunk_id, chunk, chunk_meta))

            all_segments_for_chroma.extend(file_specific_segments)
            log_message(f"Prepared {len(file_specific_segments)} segments/chunks for file: {filename}", "debug")
            file_end_time = time.time()
            log_message(f"Finished parsing/preparing file: {filename} (took {file_end_time - file_start_time:.2f}s)", "info")

        except Exception as e:
            st.exception(e)
            log_message(f"Critical error during processing pipeline for {filename}: {e}", "error")
            chat_state["processed_files"][filename] = 'failed' # Mark specific file as failed if parser fails

        processed_count += 1

    # --- Batch Add to ChromaDB ---
    if all_segments_for_chroma:
        ids_batch = [item[0] for item in all_segments_for_chroma]
        docs_batch = [item[1] for item in all_segments_for_chroma]
        metadatas_batch = [item[2] for item in all_segments_for_chroma]

        if len(ids_batch) != len(set(ids_batch)):
             log_message("Error: Duplicate IDs generated within batch!", "error")
             for f_name in files_processed_this_run: chat_state["processed_files"][f_name] = 'failed'; return False

        batch_size = 100
        num_batches = (len(ids_batch) + batch_size - 1) // batch_size
        log_message(f"Preparing to add {len(ids_batch)} items to ChromaDB in {num_batches} batches...")
        add_errors = 0
        add_start_time = time.time()
        successful_chroma_ids = set() # Track IDs successfully added to Chroma

        try:
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(ids_batch))
                batch_ids = ids_batch[start_idx:end_idx]
                batch_docs = docs_batch[start_idx:end_idx]
                batch_metas = metadatas_batch[start_idx:end_idx]
                if not batch_ids: continue
                log_message(f"Adding batch {i+1}/{num_batches} ({len(batch_ids)} items)...", "debug")
                collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                successful_chroma_ids.update(batch_ids) # Mark batch IDs as successful if add doesn't throw

            add_end_time = time.time()
            log_message(f"Finished adding {len(successful_chroma_ids)} items to ChromaDB (took {add_end_time - add_start_time:.2f}s).", "info")
            if len(successful_chroma_ids) > 0:
                 # Check if any *non-data* segments were added
                 for item_id, meta in zip(ids_batch, metadatas_batch):
                     if item_id in successful_chroma_ids and not meta.get("content_type", "").startswith("data_import"):
                          chroma_add_success_flag = True
                          break

        except Exception as e:
            add_errors += 1
            failed_batch_ids = set(ids_batch[start_idx:end_idx])
            st.exception(e)
            log_message(f"Error adding batch {i+1} to ChromaDB: {e}", "error")
            log_message("Stopping ChromaDB add due to batch error.", "error")
            # We don't know exactly which item failed, treat whole batch as failed for Chroma status

        # --- *** Final Status Update Logic - MODIFIED *** ---
        # Update status for files processed in this run
        for f_name in files_processed_this_run:
            is_data_file = f_name.lower().endswith(data_file_extensions)
            current_status = chat_state["processed_files"].get(f_name)

            # If it's a data file, its status was ALREADY set by parse_data_file. DO NOT OVERWRITE.
            if is_data_file:
                if current_status not in ['success', 'failed', 'skipped']:
                    # Should not happen if parse_data_file works correctly, but as failsafe:
                    log_message(f"Warning: Status for data file {f_name} was '{current_status}', expected success/failed/skipped. Marking failed.", "warning")
                    chat_state["processed_files"][f_name] = 'failed'
                continue # Skip to next file

            # If it's NOT a data file, update status based on Chroma add results
            if current_status == 'processing': # Only update if still pending
                # Check if *any* segment from this file was successfully added to Chroma
                file_segments_in_chroma = False
                for item_id, meta in zip(ids_batch, metadatas_batch):
                     if meta.get("source") == f_name and item_id in successful_chroma_ids:
                          file_segments_in_chroma = True
                          break

                if file_segments_in_chroma:
                     chat_state["processed_files"][f_name] = 'success'
                     chroma_add_success_flag = True # Mark overall Chroma success
                else:
                     # If no segments added (maybe parsing yielded nothing, or add failed)
                     log_message(f"No segments added to Chroma for non-data file {f_name}. Marking failed.", "warning")
                     chat_state["processed_files"][f_name] = 'failed'

            # Don't change status if it was already 'failed' or 'skipped' during parsing

        if add_errors > 0:
             log_message(f"Completed processing with {add_errors} ChromaDB batch error(s).", "error")
             # chroma_add_success_flag might still be True if *some* batches succeeded before the error

    else:
        log_message("No new segments/chunks were generated to add to ChromaDB.", "info")
        # Update status for files that produced no segments
        for f_name in files_processed_this_run:
            if chat_state["processed_files"].get(f_name) == 'processing':
                 log_message(f"File '{f_name}' produced no content for ChromaDB.", "debug")
                 chat_state["processed_files"][f_name] = 'skipped'

    # Return flag indicating if *any* non-data segment addition succeeded
    return chroma_add_success_flag


# --- get_relevant_context ---
# ... (keep as is) ...
def get_relevant_context(query, collection, n_results=MAX_CONTEXT_RESULTS):
    """Retrieves relevant text chunks from ChromaDB."""
    if collection is None: log_message("Cannot retrieve context, collection is None.", "error"); return [], []
    try:
        log_message(f"Querying vector store for '{query[:50]}...' (n_results={n_results})", "debug")
        results = collection.query(query_texts=[query], n_results=n_results, include=['documents', 'metadatas', 'distances'])

        if not results or not results.get('ids') or not results['ids'][0]: log_message("No relevant context found in vector store.", "info"); return [], []

        docs = results.get('documents', [[]])[0]; metas = results.get('metadatas', [[]])[0]; dists = results.get('distances', [[]])[0]
        max_len = len(results['ids'][0])
        docs = (docs + [None] * max_len)[:max_len]; metas = (metas + [None] * max_len)[:max_len]; dists = (dists + [None] * max_len)[:max_len]

        combined = list(zip(docs, metas, dists))
        valid_results = [(doc, meta, dist) for doc, meta, dist in combined if dist is not None and meta is not None]
        if not valid_results: log_message("No valid context results after filtering.", "info"); return [],[]

        sorted_results = sorted(valid_results, key=lambda x: x[2])
        context_docs = [item[0] for item in sorted_results]; context_metadatas = [item[1] for item in sorted_results]
        log_message(f"Retrieved {len(context_docs)} relevant context chunks.", "debug")
        return context_docs, context_metadatas
    except Exception as e:
        st.exception(e); log_message(f"Error querying ChromaDB: {e}", "error"); return [], []

# --- END OF FILE vector_store.py ---
