# vector_store.py
"""Handles interaction with ChromaDB: embedding, adding data, and querying."""

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import time
import os # <-- Add os

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
    SUPPORTED_IMAGE_TYPES 
)

# --- Embedding Function ---
@st.cache_resource
def initialize_embedding_function(api_key):
     """Creates the embedding function object, cached for the session."""
     # Check if already initialized in session state to avoid re-initializing
     if "embedding_func_initialized" in st.session_state and st.session_state.embedding_func_initialized:
         # Retrieve the existing function if available (though cache_resource should handle this)
         # This check prevents redundant logging if called multiple times.
         existing_func = st.session_state.get("embedding_func_instance")
         if existing_func:
             # log_message("Using cached embedding function instance.", "debug") # Optional: reduce log noise
             return existing_func

     log_message("Initializing embedding function...", "debug")
     if not api_key:
          log_message("API Key is missing, cannot initialize embedding function.", "error")
          return None
     try:
          func = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
              api_key=api_key,
              model_name=EMBEDDING_MODEL_NAME
          )
          st.session_state.embedding_func_initialized = True
          st.session_state.embedding_func_instance = func # Store the instance
          log_message(f"Embedding function initialized with model: {EMBEDDING_MODEL_NAME}", "info")
          return func
     except Exception as e:
          log_message(f"Failed to create embedding function: {e}", "error")
          st.exception(e)
          st.session_state.embedding_func_initialized = False # Mark as failed
          if "embedding_func_instance" in st.session_state:
               del st.session_state.embedding_func_instance # Clear potentially bad instance
          return None


# --- Core Processing and Embedding Pipeline ---
def process_and_embed(uploaded_files, collection, embedding_function):
    """
    Processes uploaded files:
    - Parses text/image files, extracts content, chunks, embeds, and stores in ChromaDB.
    - Calls specific parser for data files (CSV/XLSX/JSON) which imports them
      to a separate SQLite DB and returns only a marker segment for ChromaDB.
    - Crawls URLs found in text files.
    Uses UUIDs for chunk IDs. Updates chat_state["processed_files"] status.

    Returns:
        bool: True if ChromaDB addition was attempted (even if partially failed),
              False if critical errors occurred before adding to Chroma.
    """
    if not uploaded_files:
        log_message("No files provided for processing.", "warning")
        return False # Nothing to do
    if collection is None:
         log_message("Cannot process: ChromaDB collection is not available.", "error")
         return False
    if embedding_function is None:
         log_message("Cannot process: Embedding function is not available.", "error")
         return False

    all_segments_for_chroma = [] # List of tuples: (id, text_segment_or_marker, metadata)
    files_processed_this_run = set() # Track files processed in this specific call
    total_files = len(uploaded_files)
    processed_count = 0
    any_chroma_add_attempted = False # Track if we got to the point of adding to Chroma

    # Get current chat state - essential for parsers and status updates
    chat_id = st.session_state.get("current_chat_id")
    chat_state = st.session_state.chats.get(chat_id)
    if not chat_state:
        log_message("Critical Error in process_and_embed: Could not get current chat state.", "error")
        return False # Cannot proceed without chat state

    # Ensure processed_files dict exists in chat state
    if "processed_files" not in chat_state:
        chat_state["processed_files"] = {}

    overall_start_time = time.time()
    log_message(f"--- Starting processing job for {total_files} files (Chat: {chat_id}) ---")

    # Define parser map including the updated data parser
    parser_map = {
        'pdf': parse_pdf, 'docx': parse_docx, 'pptx': parse_pptx,
        'csv': parse_data_file, 'xlsx': parse_data_file, 'json': parse_data_file, # Uses the DB import parser
        'txt': parse_txt,
        'png': parse_image, 'jpg': parse_image, 'jpeg': parse_image,
    }

    for i, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name
        file_start_time = time.time()
        log_message(f"Starting file {i+1}/{total_files}: {filename}")

        # Check if already successfully processed *in this chat* before re-processing
        # Note: Re-uploading will trigger reprocessing due to handle_file_upload resetting status
        # This check is more for internal robustness if called multiple times accidentally
        # if chat_state["processed_files"].get(filename) == 'success':
        #      log_message(f"Skipping already successfully processed file in this chat: {filename}", "info")
        #      continue

        # Mark file as started processing (will be updated to success/failed later)
        chat_state["processed_files"][filename] = 'processing'
        files_processed_this_run.add(filename) # Track that we touched this file

        file_content = uploaded_file.getvalue()
        file_type = filename.split('.')[-1].lower()
        parser_func = parser_map.get(file_type)

        if not parser_func:
            log_message(f"Unsupported file type: {filename}. Skipping.", "warning")
            chat_state["processed_files"][filename] = 'skipped'
            continue

        # --- File Parsing ---
        try:
            # parse_data_file now directly uses chat_state implicitly via session_state
            # Other parsers don't need chat_state directly
            parsed_segments, urls = parser_func(file_content, filename)

            # Handle case where parser returns nothing (e.g., empty file)
            if not parsed_segments:
                 log_message(f"No content segments extracted from {filename}. Marking as skipped/empty.", "warning")
                 # Keep 'processing' status for now, will update based on Chroma add success later
                 # Or mark as 'skipped' if definitely nothing to add
                 if file_type not in SUPPORTED_DATA_TYPES: # Don't mark data files skipped if marker wasn't even generated
                      chat_state["processed_files"][filename] = 'skipped'
                 continue # Move to next file

            # --- Process Segments for Chroma ---
            file_specific_segments = []

            # URL Crawling (only for text-based types)
            if file_type in SUPPORTED_TEXT_TYPES and urls:
                 unique_urls = set(u for u in urls if u)
                 log_message(f"Found {len(unique_urls)} unique URLs in {filename}. Attempting crawl...", "info")
                 for url in unique_urls:
                     crawled_text = crawl_url(url) # crawl_url handles caching
                     if crawled_text:
                          crawl_meta = {"source": filename, "crawled_url": url, "content_type": "crawled_web"}
                          # Add crawled content as a separate segment for this file
                          parsed_segments.append((f"\n--- Content from {url} ---\n{crawled_text}\n--- End Content ---", crawl_meta))

            # --- Chunking Each Segment (Text/Image/Marker) ---
            for text_segment, segment_meta in parsed_segments:
                 if not text_segment or not isinstance(text_segment, str): continue

                 # Data file markers should NOT be chunked - add them as single segments
                 if segment_meta.get("content_type", "").startswith("data_import"):
                     chunk_id = generate_unique_id()
                     file_specific_segments.append((chunk_id, text_segment, segment_meta))
                     continue # Move to next segment

                 # Chunk other content (text, image analysis, OCR)
                 chunks = chunk_text(text_segment, CHUNK_SIZE, CHUNK_OVERLAP)
                 for chunk_index, chunk in enumerate(chunks):
                      chunk_id = generate_unique_id()
                      chunk_meta = segment_meta.copy()
                      chunk_meta["chunk_index"] = chunk_index
                      chunk_meta["segment_length"] = len(text_segment)
                      chunk_meta["chunk_in_segment"] = f"{chunk_index + 1}/{len(chunks)}"
                      # Ensure basic metadata always exists
                      chunk_meta.setdefault("source", filename)
                      chunk_meta.setdefault("content_type", "unknown") # Should be set by parser

                      file_specific_segments.append((chunk_id, chunk, chunk_meta))

            # Add all processed segments for this file to the main batch
            all_segments_for_chroma.extend(file_specific_segments)

            # Mark file as processed *provisionally* - success depends on Chroma add
            # The final status 'success'/'failed' will be set after Chroma batch add
            log_message(f"Prepared {len(file_specific_segments)} segments/chunks for file: {filename}", "debug")
            file_end_time = time.time()
            log_message(f"Finished parsing/preparing file: {filename} (took {file_end_time - file_start_time:.2f}s)", "info")

        except Exception as e:
            st.exception(e)
            log_message(f"Critical error during processing pipeline for {filename}: {e}", "error")
            chat_state["processed_files"][filename] = 'failed' # Mark specific file as failed

        processed_count += 1
        # Add status update here if needed, e.g., progress bar update

    # --- Batch Add to ChromaDB ---
    if all_segments_for_chroma:
        any_chroma_add_attempted = True # Mark that we are attempting the add
        ids_batch = [item[0] for item in all_segments_for_chroma]
        docs_batch = [item[1] for item in all_segments_for_chroma]
        metadatas_batch = [item[2] for item in all_segments_for_chroma]

        # Check for duplicate IDs within the batch (shouldn't happen with UUID)
        if len(ids_batch) != len(set(ids_batch)):
             log_message("Error: Duplicate IDs generated within the same batch!", "error")
             # Mark all files in this batch as failed? Or just log? Mark as failed for safety.
             for f_name in files_processed_this_run:
                 chat_state["processed_files"][f_name] = 'failed'
             return False # Indicate failure

        batch_size = 100 # Adjust as needed, consider ChromaDB limits/performance
        num_batches = (len(ids_batch) + batch_size - 1) // batch_size
        log_message(f"Preparing to add {len(ids_batch)} segments/chunks to ChromaDB in {num_batches} batches...")
        add_errors = 0
        add_start_time = time.time()

        successful_ids = set()
        failed_ids = set()

        try:
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(ids_batch))
                batch_ids = ids_batch[start_idx:end_idx]
                batch_docs = docs_batch[start_idx:end_idx]
                batch_metas = metadatas_batch[start_idx:end_idx]

                if not batch_ids: continue # Skip empty batch

                log_message(f"Adding batch {i+1}/{num_batches} ({len(batch_ids)} items)...", "debug")
                # Use 'add'. If an ID exists (e.g., reprocessing attempt failed midway),
                # Chroma might throw an error depending on version/setup. UUIDs should prevent this.
                collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                successful_ids.update(batch_ids) # Assume success if no exception

            add_end_time = time.time()
            log_message(f"Finished adding {len(successful_ids)} items to ChromaDB (took {add_end_time - add_start_time:.2f}s).", "info")

        except Exception as e:
            # This usually indicates the *entire batch* failed with current Chroma client behavior
            add_errors += 1
            failed_batch_ids = set(ids_batch[start_idx:end_idx]) # IDs in the failed batch
            failed_ids.update(failed_batch_ids)
            st.exception(e)
            log_message(f"Error adding batch {i+1} to ChromaDB: {e}", "error")
            # Continue to next batch? Or stop? Let's stop on first batch error for now.
            log_message("Stopping ChromaDB add due to batch error.", "error")
            # Mark all remaining items as failed? Complicated. Simpler to mark files associated with failed IDs.

        # --- Update file statuses based on Chroma results ---
        if add_errors > 0:
             log_message(f"Completed adding items to ChromaDB with {add_errors} batch error(s).", "error")
             # Determine which files had segments that failed
             failed_files = set()
             for item_id, meta in zip(ids_batch, metadatas_batch):
                 if item_id in failed_ids:
                      failed_files.add(meta.get("source", "Unknown_Source"))
             for f_name in files_processed_this_run:
                 if f_name in failed_files:
                      chat_state["processed_files"][f_name] = 'failed'
                 elif chat_state["processed_files"].get(f_name) == 'processing': # If not failed explicitly and still 'processing'
                      chat_state["processed_files"][f_name] = 'success' # Mark as success if some chunks might have landed
                 # Don't override 'skipped' or existing 'failed' status from parsing phase
        else:
             log_message("All batches added successfully to ChromaDB.", "info")
             # Mark all files processed in this run as success if they weren't already failed/skipped
             for f_name in files_processed_this_run:
                  if chat_state["processed_files"].get(f_name) == 'processing':
                       chat_state["processed_files"][f_name] = 'success'

    else:
        log_message("No new segments/chunks were generated to add to ChromaDB.", "info")
        # Update status for files that were processed but generated no chunks (e.g., empty text files)
        for f_name in files_processed_this_run:
            if chat_state["processed_files"].get(f_name) == 'processing':
                 # If parse_data_file resulted in success marker, it's already added.
                 # If other parsers resulted in no segments, mark as skipped/empty?
                 log_message(f"File '{f_name}' produced no content for ChromaDB.", "debug")
                 chat_state["processed_files"][f_name] = 'skipped' # Mark as skipped if no content was added

    # Return True if the add operation was attempted, False otherwise
    return any_chroma_add_attempted


# --- get_relevant_context (Keep As Is) ---
def get_relevant_context(query, collection, n_results=MAX_CONTEXT_RESULTS):
    """Retrieves relevant text chunks from ChromaDB."""
    if collection is None:
        log_message("Cannot retrieve context, collection is None.", "error")
        return [], []
    try:
        log_message(f"Querying vector store for '{query[:50]}...' (n_results={n_results})", "debug")
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

        # Check results format carefully
        if not results or not results.get('ids') or not results['ids'][0]:
             log_message("No relevant context found in vector store for the query.", "info")
             return [], []

        # Combine, filter Nones, sort by distance (lower is better)
        docs = results.get('documents', [[]])[0]
        metas = results.get('metadatas', [[]])[0]
        dists = results.get('distances', [[]])[0]

        # Ensure all lists have the same length, pad with None if necessary (shouldn't happen ideally)
        max_len = len(results['ids'][0])
        docs = (docs + [None] * max_len)[:max_len]
        metas = (metas + [None] * max_len)[:max_len]
        dists = (dists + [None] * max_len)[:max_len]


        combined = list(zip(docs, metas, dists))
        # Filter out entries where distance is None (problematic) or metadata is None
        valid_results = [(doc, meta, dist) for doc, meta, dist in combined if dist is not None and meta is not None]

        if not valid_results:
             log_message("No valid context results after filtering.", "info")
             return [],[]

        # Sort by distance (ascending)
        sorted_results = sorted(valid_results, key=lambda x: x[2])

        context_docs = [item[0] for item in sorted_results]
        context_metadatas = [item[1] for item in sorted_results]
        log_message(f"Retrieved {len(context_docs)} relevant context chunks.", "debug")
        return context_docs, context_metadatas

    except Exception as e:
        st.exception(e)
        log_message(f"Error querying ChromaDB: {e}", "error")
        return [], []