# --- START OF FILE main_app.py ---
"""
Main Streamlit application file for the Multi-Modal Q&A Chatbot.
Handles UI, state management, and orchestrates calls to other modules.
V5.8 Changes:
- Refined status reporting logic in handle_file_upload based on separate
  tracking of ChromaDB add success and data file import success.
"""

# --- !! PATCH SQLITE FIRST !! ---
try:
    import pysqlite3
    import sys
    if 'sqlite3' not in sys.modules or sys.modules['sqlite3'].sqlite_version_info < (3, 35, 0):
        print("Attempting to patch sqlite3 with pysqlite3...")
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        import sqlite3
        print(f"Successfully patched sqlite3. Using version: {sqlite3.sqlite_version}")
    else:
        print(f"sqlite3 version {sys.modules['sqlite3'].sqlite_version} >= 3.35.0 or already patched. Skipping patch.")
except ImportError: print("WARNING: pysqlite3 not found. Using default sqlite3. ChromaDB may fail.")
except KeyError: print("WARNING: Error during pysqlite3 patching (KeyError). Using default sqlite3.")
except Exception as e: print(f"ERROR: An unexpected error occurred during pysqlite3 patching: {e}")
# --- !! END PATCH !! ---

import streamlit as st
import google.generativeai as genai
import chromadb
import os
import time
import random
import string
import pandas as pd
from datetime import datetime
import re

# Import functions from our modules
from config import (
    DEFAULT_COLLECTION_NAME_PREFIX, ALL_SUPPORTED_TYPES,
    VECTOR_DB_PERSIST_PATH, CHAT_DB_DIRECTORY,
    TEXT_MODEL_NAME as DEFAULT_TEXT_MODEL_NAME,
    SUPPORTED_DATA_TYPES, SUPPORTED_TEXT_TYPES
)
from utils import log_message, generate_unique_id
from vector_store import initialize_embedding_function, process_and_embed, get_relevant_context
from qa_engine import is_data_query, answer_data_query, generate_answer_from_context, generate_followup_questions

# --- Page Configuration ---
st.set_page_config(
    page_title="Persistent Multi-Modal Q&A + Data Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("📄 Persistent Multi-Modal Q&A & Data Analysis")
st.caption("Switch chats, upload docs/data, ask questions about text, images, or analyze your CSV/Excel/JSON!")

# --- Constants for Model Selection ---
AVAILABLE_TEXT_MODELS = ["gemini-1.5-flash"]

# --- Global Variables / Initializations ---
@st.cache_resource
def initialize_chroma_client():
    """Initializes ChromaDB client (FORCED IN-MEMORY for debugging)."""
    log_message("Initializing ChromaDB client (Forced In-Memory)...", "info")
    try:
        client = chromadb.Client()
        client.heartbeat() # Check connection
        log_message("In-memory client initialized successfully.", "info")
        return client
    except Exception as e:
        log_message(f"FATAL: Failed to initialize ChromaDB client: {e}", "error")
        st.error(f"Failed to initialize Database Backend: {e}. Please check logs or environment configuration.")
        st.stop()

# --- Session State Management ---
# ... (create_new_chat_state, initialize_multi_chat_state, get_current_chat_state remain the same) ...
def create_new_chat_state(chat_id):
    return {"chat_id": chat_id, "collection_name": DEFAULT_COLLECTION_NAME_PREFIX + chat_id,
            "messages": [{"role": "assistant", "content": "Hi there! Upload documents or data files to get started."}],
            "processed_files": {}, "chat_db_path": None, "imported_tables": [], "crawled_data": {},
            "processing_status": "idle", "data_import_status": "idle", "created_at": datetime.now().isoformat()}
def initialize_multi_chat_state():
    if "chats" not in st.session_state:
        st.session_state.chats = {}
        if "processing_log" not in st.session_state: st.session_state.processing_log = []
        log_message("Initializing multi-chat state.", "info")
        first_chat_id = generate_unique_id(); st.session_state.chats[first_chat_id] = create_new_chat_state(first_chat_id)
        st.session_state.current_chat_id = first_chat_id; log_message(f"Created initial chat: {first_chat_id}", "info")
    elif "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
        if st.session_state.chats:
            valid_chats = {k: v for k, v in st.session_state.chats.items() if isinstance(v, dict)}
            if valid_chats: most_recent_chat_id = max(valid_chats.keys(), key=lambda k: valid_chats[k].get('created_at', '')); st.session_state.current_chat_id = most_recent_chat_id; log_message(f"Set current chat to most recent: {most_recent_chat_id}", "debug")
            else: new_chat_id = generate_unique_id(); st.session_state.chats[new_chat_id] = create_new_chat_state(new_chat_id); st.session_state.current_chat_id = new_chat_id; log_message("No valid chats found, created new one: {new_chat_id}", "info")
        else: new_chat_id = generate_unique_id(); st.session_state.chats[new_chat_id] = create_new_chat_state(new_chat_id); st.session_state.current_chat_id = new_chat_id; log_message("No chats found, created new one: {new_chat_id}", "info")
    if "api_key_configured" not in st.session_state: st.session_state.api_key_configured = False
    if "embedding_func" not in st.session_state: st.session_state.embedding_func = None
    if "rerun_query" not in st.session_state: st.session_state.rerun_query = None
    if "selected_text_model" not in st.session_state:
        default_model = DEFAULT_TEXT_MODEL_NAME
        if default_model not in AVAILABLE_TEXT_MODELS and AVAILABLE_TEXT_MODELS: default_model = AVAILABLE_TEXT_MODELS[0]
        elif not AVAILABLE_TEXT_MODELS: default_model = "gemini-1.5-flash"; log_message("Warning: AVAILABLE_TEXT_MODELS list empty/invalid.", "warning")
        st.session_state.selected_text_model = default_model
initialize_multi_chat_state()
def get_current_chat_state():
    current_id = st.session_state.get("current_chat_id")
    if not current_id or current_id not in st.session_state.get("chats", {}):
        initialize_multi_chat_state(); current_id = st.session_state.get("current_chat_id")
        if not current_id or current_id not in st.session_state.get("chats", {}): log_message("Critical Error: Cannot determine current chat state after re-init.", "error"); st.error("Critical error loading chat state."); return None
    chat_state = st.session_state.chats.get(current_id)
    if not isinstance(chat_state, dict): log_message(f"Critical Error: Chat state for {current_id} is not dict.", "error"); st.error(f"Chat data for '{current_id[-4:]}' corrupted."); return None
    return chat_state

# --- API Key Handling & Sidebar ---
# ... (keep as is) ...
with st.sidebar:
    st.header("⚙️ Configuration")
    current_selection = st.session_state.get("selected_text_model", DEFAULT_TEXT_MODEL_NAME)
    if current_selection not in AVAILABLE_TEXT_MODELS: current_selection = AVAILABLE_TEXT_MODELS[0] if AVAILABLE_TEXT_MODELS else "gemini-1.5-flash"
    selected_model = st.selectbox("Choose Text Model:", options=AVAILABLE_TEXT_MODELS, key="selected_text_model", index=AVAILABLE_TEXT_MODELS.index(current_selection) if current_selection in AVAILABLE_TEXT_MODELS else 0, help="Select LLM for Q&A and Data Analysis.")
    st.header("🔑 API Configuration")
    try:
        input_key = st.session_state.get("api_key_input_sidebar", ""); current_key = st.session_state.get("google_api_key", ""); env_api_key = os.environ.get("GOOGLE_API_KEY"); secrets_api_key = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, 'secrets') else None
        display_value = input_key or current_key or env_api_key or secrets_api_key or ""
        google_api_key_input = st.text_input("Enter Google AI API Key:", type="password", key="api_key_input_sidebar", value=display_value, help="Overrides env/secrets.")
        if google_api_key_input and google_api_key_input != display_value: final_google_api_key = google_api_key_input; st.session_state.google_api_key = google_api_key_input
        else: final_google_api_key = current_key or env_api_key or secrets_api_key
        if final_google_api_key and final_google_api_key != st.session_state.get("google_api_key"): st.session_state.google_api_key = final_google_api_key
        elif not final_google_api_key and "google_api_key" in st.session_state: del st.session_state.google_api_key
    except Exception as e: st.error(f"Error accessing secrets/env vars: {e}"); final_google_api_key = None; st.session_state.google_api_key = None
if final_google_api_key and not st.session_state.api_key_configured:
    try:
        log_message("Attempting to configure Google AI API...", "debug")
        genai.configure(api_key=final_google_api_key)
        st.session_state.embedding_func = initialize_embedding_function(final_google_api_key)
        if st.session_state.embedding_func: st.session_state.api_key_configured = True; st.session_state.google_api_key = final_google_api_key; log_message("Google AI API configured.", "info")
        else: st.session_state.api_key_configured = False; log_message("Embedding function init failed.", "error");
        if not st.session_state.api_key_configured and "google_api_key" in st.session_state: del st.session_state.google_api_key
    except Exception as e: log_message(f"Error configuring Google AI API: {e}", "error"); st.session_state.api_key_configured = False; st.session_state.embedding_func = None;
    if not st.session_state.api_key_configured and "google_api_key" in st.session_state: del st.session_state.google_api_key
with st.sidebar:
    if st.session_state.get("api_key_configured"): st.success("API Key Configured.")
    elif final_google_api_key: st.error("API Key Invalid or Config Failed.")
    else: st.warning("API Key Needed.")
chroma_client = initialize_chroma_client()

# --- File Processing Callback (MODIFIED STATUS HANDLING) ---
def handle_file_upload():
    """Callback for file uploader. Processes files for the CURRENT chat."""
    log_message("File uploader callback started.", "debug")
    current_chat_id = st.session_state.get("current_chat_id")
    if not current_chat_id: log_message("Upload callback: No current chat ID.", "error"); st.toast("⚠️ No active chat."); return

    chat_state = get_current_chat_state()
    if not chat_state: log_message("Upload callback: Could not get chat state.", "error"); st.toast("⚠️ Error loading chat."); return
    if not st.session_state.api_key_configured: st.toast("⚠️ Configure API Key first."); return

    uploaded_files = st.session_state.get("doc_uploader_key", [])
    if not uploaded_files: log_message("Upload callback: No files found.", "warning"); return

    # Indicate processing start
    chat_state["processing_status"] = "processing" # General status
    chat_state["data_import_status"] = "idle"     # Reset data status
    st.info("⏳ Processing uploaded files...") # Give immediate feedback

    log_message(f"Callback: Processing {len(uploaded_files)} files for chat {current_chat_id}...", "info")
    chat_state["processed_files"] = {} # Reset file-specific status for this batch

    client = initialize_chroma_client()
    if not client:
        log_message(f"[{current_chat_id}] Chroma client unavailable.", "error"); st.error("DB backend init failed.")
        chat_state["processing_status"] = "error"; chat_state["data_import_status"] = "error"; return

    emb_func = st.session_state.embedding_func
    collection = None
    chroma_overall_success = False # Track if *any* non-data processing worked
    data_import_occurred = False
    any_data_import_failed = False # Track if *any* data file failed

    if not emb_func:
         log_message(f"[{current_chat_id}] Embedding func unavailable.", "error")
         chat_state["processing_status"] = "error"; chat_state["data_import_status"] = "error"; return

    collection_name = chat_state["collection_name"]
    try:
        log_message(f"[{current_chat_id}] Getting/creating collection: '{collection_name}'.", "info")
        collection = client.get_or_create_collection(name=collection_name, embedding_function=emb_func, metadata={"hnsw:space": "cosine"})
        log_message(f"[{current_chat_id}] Collection ready.", "info")
    except Exception as e:
         log_message(f"[{current_chat_id}] Fatal Error getting/creating Chroma collection: {e}", "error"); st.exception(e)
         if "no such table: acquire_write" in str(e): st.error("Failed to access vector DB backend (SQLite/ChromaDB setup issue).")
         else: st.error(f"Failed to init vector collection: {e}")
         chat_state["processing_status"] = "error"; chat_state["data_import_status"] = "error"; return

    # --- Call process_and_embed ---
    if collection and emb_func:
         log_message(f"[{current_chat_id}] Calling process_and_embed...", "debug")
         # process_and_embed now returns True if Chroma add for non-data files was successful
         chroma_overall_success = process_and_embed(uploaded_files, collection, emb_func)
         log_message(f"[{current_chat_id}] process_and_embed chroma success flag: {chroma_overall_success}", "debug")

         # Check final status of data files set by the parser/importer
         data_file_extensions = tuple(f".{ext}" for ext in SUPPORTED_DATA_TYPES)
         for f in uploaded_files:
             if f.name.lower().endswith(data_file_extensions):
                 data_import_occurred = True
                 status = chat_state["processed_files"].get(f.name)
                 if status != 'success':
                     any_data_import_failed = True
                     log_message(f"Data import for {f.name} final status: {status}.", "warning")
                     # No break, check all data files

    else: # Should not happen if checks above passed
         log_message(f"[{current_chat_id}] Processing aborted pre-call.", "error")
         chroma_overall_success = False
         any_data_import_failed = True # Assume failure if we couldn't call

    # --- *** Determine Final Status Messages *** ---
    final_chroma_status = "success" if chroma_overall_success else "error"
    if not any(not f.name.lower().endswith(tuple(f".{ext}" for ext in SUPPORTED_DATA_TYPES)) for f in uploaded_files):
        final_chroma_status = "idle" # No non-data files processed

    if not data_import_occurred:
        final_data_status = "idle"
    elif any_data_import_failed:
        final_data_status = "error"
    else:
        final_data_status = "success"
        st.toast(f"📊 Data file(s) imported successfully!", icon="📊") # Toast on success

    # Update chat state with final statuses
    chat_state["processing_status"] = final_chroma_status
    chat_state["data_import_status"] = final_data_status

    log_message(f"Callback Final: Chroma status='{final_chroma_status}', Data status='{final_data_status}' for chat {current_chat_id}", "info")
    # Rerun will happen automatically, showing status from chat state

# --- Helper to get Chroma Collection ---
# ... (keep as is) ...
def get_current_collection():
    chat_state = get_current_chat_state()
    if not chat_state or not st.session_state.api_key_configured or not st.session_state.embedding_func: log_message("Cannot get collection: State/API/Emb missing.", "warning"); return None
    collection_name = chat_state.get("collection_name");
    if not collection_name: log_message("Cannot get collection: Name missing.", "warning"); return None
    try:
        client = initialize_chroma_client()
        if not client: log_message("Cannot get collection: Chroma client unavailable.", "error"); return None
        collection = client.get_collection(name=collection_name, embedding_function=st.session_state.embedding_func)
        return collection
    except Exception as e:
        log_message(f"Collection '{collection_name}' not found or error: {e}", "debug")
        if "no such table: acquire_write" in str(e): st.error("Vector DB backend error (SQLite/ChromaDB).")
        return None

# --- Sidebar UI ---
# ... (keep as is) ...
with st.sidebar:
    st.header("📄 Chat Management");
    if st.button("➕ New Chat", key="new_chat_button", use_container_width=True): new_chat_id = generate_unique_id(); st.session_state.chats[new_chat_id] = create_new_chat_state(new_chat_id); st.session_state.current_chat_id = new_chat_id; log_message(f"Created/switched to new chat: {new_chat_id}", "info"); st.rerun()
    st.divider(); st.header("🗂️ Chat Sessions");
    sorted_chat_ids = sorted(st.session_state.chats.keys(), key=lambda k: st.session_state.chats.get(k, {}).get('created_at', ''), reverse=True)
    chat_list_container = st.container(height=300)
    with chat_list_container:
        if not sorted_chat_ids: st.caption("No chat sessions yet.")
        else:
            current_chat_id_local = st.session_state.get("current_chat_id"); chat_to_delete = None
            for chat_id in sorted_chat_ids:
                chat_state = st.session_state.chats.get(chat_id);
                if not isinstance(chat_state, dict): continue
                first_user_message = next((msg['content'] for msg in chat_state.get('messages', []) if msg.get('role') == 'user'), None)
                label_base = first_user_message[:30] + "..." if first_user_message else f"Chat {chat_id[-4:]}"; has_db = bool(chat_state.get("chat_db_path")); label = f"{'📊 ' if has_db else ''}{label_base}"
                created_at_str = chat_state.get('created_at', None); timestamp = datetime.fromisoformat(created_at_str).strftime('%b %d, %H:%M') if created_at_str else "???"
                button_label = f"{label} ({timestamp})"; button_key = f"switch_chat_{chat_id}"; delete_button_key = f"delete_chat_{chat_id}"; button_type = "primary" if chat_id == current_chat_id_local else "secondary"
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    if st.button(button_label, key=button_key, use_container_width=True, type=button_type):
                        if st.session_state.current_chat_id != chat_id: log_message(f"Switching to chat: {chat_id}", "info"); st.session_state.current_chat_id = chat_id; st.rerun()
                with col2:
                    if st.button("🗑️", key=delete_button_key, help=f"Delete chat '{label_base}'"): chat_to_delete = chat_id; log_message(f"Delete clicked for chat: {chat_id}", "info")
            if chat_to_delete:
                log_message(f"Performing deletion of chat: {chat_to_delete}", "info"); chat_state_to_delete = st.session_state.chats.get(chat_to_delete)
                if chat_state_to_delete:
                    collection_name_to_delete = chat_state_to_delete.get('collection_name')
                    if collection_name_to_delete:
                        try: client = initialize_chroma_client();
                             if client: log_message(f"Deleting Chroma collection: {collection_name_to_delete}", "info"); client.delete_collection(name=collection_name_to_delete)
                             else: log_message(f"Cannot delete Chroma coll {collection_name_to_delete}: client unavailable", "error")
                        except Exception as e: log_message(f"Chroma Collection '{collection_name_to_delete}' delete failed: {e}", "warning")
                    db_path_to_delete = chat_state_to_delete.get('chat_db_path')
                    if db_path_to_delete and os.path.exists(db_path_to_delete):
                         try: log_message(f"Deleting chat DB file: {db_path_to_delete}", "info"); os.remove(db_path_to_delete); log_message(f"Deleted DB file: {db_path_to_delete}", "info")
                         except OSError as e: log_message(f"Error deleting DB file '{db_path_to_delete}': {e}", "error"); st.warning(f"Could not delete DB file for chat {chat_to_delete[-4:]}.")
                    elif db_path_to_delete: log_message(f"DB file path recorded but not found: {db_path_to_delete}", "debug")
                if chat_to_delete in st.session_state.chats: del st.session_state.chats[chat_to_delete]; log_message(f"Deleted chat state for: {chat_to_delete}", "info")
                if st.session_state.current_chat_id == chat_to_delete:
                    log_message(f"Current chat ({chat_to_delete}) deleted. Switching...", "info"); remaining_chats = {k:v for k,v in st.session_state.chats.items() if isinstance(v, dict)}
                    if remaining_chats: new_current_chat_id = max(remaining_chats.keys(), key=lambda k: remaining_chats[k].get('created_at', '')); st.session_state.current_chat_id = new_current_chat_id; log_message(f"Switched current chat to: {new_current_chat_id}", "info")
                    else: log_message("No chats remaining, creating new initial.", "info"); st.session_state.chats = {}; initialize_multi_chat_state()
                st.rerun()

# --- Helper Function to Detect Summary Queries ---
# ... (keep as is) ...
def is_summary_query(query: str) -> bool:
    query_lower = query.lower().strip(); summary_keywords = ["what is this document about", "summarize this document", "summarize", "summary of", "tell me about this doc", "overview of", "what does this file say", "give me a summary", "general topic", "main points"]
    if query_lower in summary_keywords: return True
    if query_lower.startswith(("summarize", "what is", "tell me about", "overview of")):
        if "document" in query_lower or "file" in query_lower or "pdf" in query_lower or "doc" in query_lower: return True
    return False

# --- Main Chat Area UI ---
# ... (keep as is) ...
st.header("💬 Chat Interface"); current_chat_state = get_current_chat_state()
message_container = st.container()
with message_container:
    if not current_chat_state: st.info("Select or start a new chat.")
    else:
        for i, message in enumerate(current_chat_state.get("messages", [])):
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=False)
                if message["role"] == "assistant" and "dataframe_result" in message:
                    df_result = message["dataframe_result"]
                    if isinstance(df_result, pd.DataFrame) and not df_result.empty:
                        button_key = f"show_df_{current_chat_state.get('chat_id','_')}_{i}";
                        if f"show_df_state_{button_key}" not in st.session_state: st.session_state[f"show_df_state_{button_key}"] = False
                        btn_cols = st.columns(4);
                        with btn_cols[0]:
                             if st.button("Show/Hide Full Table", key=button_key): st.session_state[f"show_df_state_{button_key}"] = not st.session_state[f"show_df_state_{button_key}"]; st.rerun()
                        if st.session_state.get(f"show_df_state_{button_key}", False): st.dataframe(df_result)
                if message["role"] == "assistant" and message.get("metadata"):
                     exp_col1, exp_col2 = st.columns([0.9, 0.1])
                     with exp_col1:
                         with st.expander("Sources Used", expanded=False):
                             sources_display = []; processed_sources = set()
                             for meta in message["metadata"]:
                                if not meta: continue
                                source = meta.get('source', 'Unknown'); content_type = meta.get('content_type','unknown'); db_source = meta.get('database')
                                if db_source and content_type == 'database_query_result':
                                    source_key = f"db_{db_source}"
                                    if source_key not in processed_sources: tables_list_str = meta.get('tables', 'unknown'); display_str = f"- **Chat Database** (Tables: {tables_list_str})"; sources_display.append(display_str); processed_sources.add(source_key)
                                elif content_type != 'database_query_result':
                                    page_num = meta.get('page_number'); slide_num = meta.get('slide_number'); crawled = meta.get('crawled_url'); source_key = f"{source}"
                                    if page_num: source_key += f"_p{page_num}"
                                    elif slide_num: source_key += f"_s{slide_num}"
                                    elif crawled: source_key += f"_u{hash(crawled)}"
                                    if source_key not in processed_sources:
                                        display_str = f"- **{source}**"
                                        if page_num: display_str += f" (Page {page_num})"
                                        elif slide_num: display_str += f" (Slide {slide_num})"
                                        elif crawled: display_str += f" (From URL)"
                                        elif content_type == 'data_import_success': display_str += " (Data File - Imported)"
                                        elif content_type == 'data_import_failed': display_str += " (Data File - Import Failed)"
                                        elif content_type == 'embedded_image': display_str += " (Image Analysis)"
                                        elif content_type == 'image_analysis': display_str += " (Image File Analysis)"
                                        elif content_type == 'ocr_page': display_str += f" (OCR Page {page_num if page_num else '?'})"
                                        elif content_type == 'text': display_str += " (Text Content)"
                                        elif content_type == 'slide_text': display_str += f" (Slide {slide_num} Text)"
                                        sources_display.append(display_str); processed_sources.add(source_key)
                             st.markdown("\n".join(sorted(sources_display)))
                if message["role"] == "assistant" and message.get("follow_ups"):
                     st.markdown("**Suggested Questions:**"); cols = st.columns(len(message["follow_ups"]))
                     for j, q in enumerate(message["follow_ups"]):
                          if cols[j].button(q, key=f"followup_{current_chat_state.get('chat_id','_')}_{i}_{j}"): st.session_state.rerun_query = q; st.rerun()

# --- Input Area & Status ---
input_container = st.container()
with input_container:
    query_input_value = "";
    if st.session_state.rerun_query: query_input_value = st.session_state.rerun_query; st.session_state.rerun_query = None
    chat_input_disabled = not st.session_state.api_key_configured
    chat_input_placeholder = "Ask about documents OR analyze CSV/Excel/JSON data..." if st.session_state.api_key_configured else "Please configure API Key in sidebar"
    prompt = st.chat_input(chat_input_placeholder, key="main_query_input", disabled=chat_input_disabled)
    st.file_uploader("Attach Files (Docs, Images, Data) for **this chat**", accept_multiple_files=True, type=ALL_SUPPORTED_TYPES, key="doc_uploader_key", on_change=handle_file_upload, label_visibility="collapsed", disabled=chat_input_disabled)

    # --- Processing Status Display Area (Uses updated statuses) ---
    status_placeholder = st.empty()
    final_message = ""
    if current_chat_state:
        current_chroma_status = current_chat_state.get("processing_status", "idle")
        current_data_status = current_chat_state.get("data_import_status", "idle")

        # Consolidate message based on final statuses
        if current_chroma_status == "processing" or current_data_status == "processing": # Check if still processing
             final_message = "⏳ Processing uploaded files..."
             status_level = "info"
        elif current_chroma_status == "error" or current_data_status == "error":
             final_message = "⚠️ File processing failed."
             if current_data_status == "error" and current_chroma_status != "error": final_message = "⚠️ Data import failed."
             elif current_chroma_status == "error" and current_data_status != "error": final_message = "⚠️ Text/Image processing failed."
             elif current_chroma_status == "error" and current_data_status == "error": final_message = "⚠️ Both Text/Image processing and Data import failed."
             status_level = "error"
        elif current_chroma_status == "success" or current_data_status == "success":
             final_message = "✅ Files processed successfully."
             if current_data_status == "success": final_message += " 📊 Data imported."
             status_level = "success"
        else: # Both idle?
             final_message = "" # No message needed if idle and no errors occurred
             status_level = "idle"

        if final_message and status_level != "idle":
            if status_level == "error": status_placeholder.error(final_message)
            elif status_level == "info": status_placeholder.info(final_message)
            else: status_placeholder.success(final_message)
            # Optionally clear status after display?
            # current_chat_state["processing_status"] = "idle"
            # current_chat_state["data_import_status"] = "idle"

    # --- Query Handling Logic ---
    # ... (keep as is) ...
    actual_prompt = prompt or query_input_value
    if actual_prompt and current_chat_state:
        current_chat_id_for_query = current_chat_state["chat_id"]; selected_model = st.session_state.selected_text_model
        current_chat_state.setdefault("messages", []).append({"role": "user", "content": actual_prompt})
        assistant_response_content = None; follow_ups = []; used_metadata = []; response_df = None
        prompt_lower = actual_prompt.lower().strip(); greetings = ["hi", "hello", "hey", "yo", "greetings", "good morning", "good afternoon", "good evening"]; help_keywords = ["help", "what can you do", "capabilities", "features", "functions", "what are your functions"]
        is_greeting = any(greet == prompt_lower for greet in greetings); is_help_request = any(keyword in prompt_lower for keyword in help_keywords)

        if is_greeting: assistant_response_content = random.choice(["Hello!", "Hi there!", "Hey!"]) + " How can I help?"
        elif is_help_request: assistant_response_content = f"""I can help with:\n1. **File Processing:** Upload Docs, Images, or Data (CSV/XLSX/JSON).\n2. **Document Q&A:** Ask about text/images in uploaded files.\n3. **Data Analysis:** Ask questions about uploaded data files (uses SQL generation).\n\nModel used: `{selected_model}`"""
        elif current_chroma_status == "processing" or current_data_status == "importing": assistant_response_content = "Hold on! Files are still processing."
        elif not st.session_state.api_key_configured: assistant_response_content = "API Key needed (sidebar)."
        else:
            chat_has_database = current_chat_state.get("chat_db_path") is not None
            looks_like_data_query = is_data_query(actual_prompt); looks_like_summary_query = is_summary_query(actual_prompt)
            if chat_has_database and looks_like_data_query and not looks_like_summary_query:
                log_message(f"[{current_chat_id_for_query}] Query -> Data Analysis Engine.", "info")
                with st.spinner("Analyzing data..."): assistant_response_content, used_metadata, response_df = answer_data_query(actual_prompt, selected_model)
            else:
                log_message(f"[{current_chat_id_for_query}] Query -> Document Q&A (RAG). Summary: {looks_like_summary_query}", "info"); current_collection = get_current_collection()
                if current_collection is None:
                      if not initialize_chroma_client(): assistant_response_content = "Error: Vector DB backend failed."
                      else:
                           files_were_processed = bool(current_chat_state.get("processed_files"))
                           if not files_were_processed: assistant_response_content = f"No documents/data processed yet in chat '{current_chat_id_for_query[-4:]}'. Upload files first."
                           else: assistant_response_content = f"No text content found for chat '{current_chat_id_for_query[-4:]}'. Ask about imported data or re-upload docs."; log_message(f"[{current_chat_id_for_query}] RAG query, but collection None/empty.", "warning")
                else:
                    with st.spinner("Searching documents..."):
                        context_docs, context_metadatas = get_relevant_context(actual_prompt, current_collection)
                        if looks_like_summary_query and context_metadatas:
                             log_message(f"[{current_chat_id_for_query}] Filtering context for summary.", "debug"); filtered_docs = []; filtered_metadatas = []; summary_content_types = {'text', 'slide_text', 'ocr_page', 'crawled_web'}
                             for doc, meta in zip(context_docs, context_metadatas):
                                 if meta and meta.get('content_type') in summary_content_types: filtered_docs.append(doc); filtered_metadatas.append(meta)
                             if not filtered_docs: log_message(f"[{current_chat_id_for_query}] No text content after filtering for summary. Using original.", "warning")
                             else: context_docs = filtered_docs; context_metadatas = filtered_metadatas
                        if not context_docs:
                             if looks_like_summary_query: assistant_response_content = "Couldn't find text content for summary."
                             else: assistant_response_content = "Couldn't find relevant info in documents."
                             used_metadata = []
                        else:
                             assistant_response_content, used_metadata = generate_answer_from_context(actual_prompt, context_docs, context_metadatas, selected_model)
                             if not looks_like_summary_query and assistant_response_content and isinstance(assistant_response_content, str) and "cannot answer" not in assistant_response_content.lower() and "error" not in assistant_response_content.lower() and "blocked" not in assistant_response_content.lower():
                                  follow_ups = generate_followup_questions(actual_prompt, assistant_response_content, selected_model)

        if assistant_response_content is not None:
            assistant_message = {"role": "assistant", "content": assistant_response_content, "follow_ups": follow_ups, "metadata": used_metadata}
            if response_df is not None and isinstance(response_df, pd.DataFrame): assistant_message["dataframe_result"] = response_df
            current_chat_state["messages"].append(assistant_message)
        st.rerun()

# --- Footer ---
st.divider(); st.caption("Verify critical info. Use 'New Chat' for separate contexts.")
# --- END OF FILE main_app.py ---
