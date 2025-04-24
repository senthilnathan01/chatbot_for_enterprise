# --- START OF FILE main_app.py ---
"""
Main Streamlit application file for the Multi-Modal Q&A Chatbot.
Handles UI, state management, and orchestrates calls to other modules.
V5.6 Changes:
- Added logic to detect summary queries.
- Filter retrieved context for summary queries to prioritize text over image analysis.
- Updated qa_engine call for summary context.
"""
try:
    __import__('pysqlite3')
    import sys
    # Print statements to help debug in Streamlit Cloud logs
    print("Attempting to patch sqlite3 with pysqlite3...")
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully patched sqlite3 with pysqlite3.")
except ImportError:
    print("pysqlite3 not found. Using default sqlite3. This will likely cause ChromaDB errors.")
except KeyError:
    print("pysqlite3 was imported but perhaps already handled? Or an issue popping it.")

import streamlit as st
import google.generativeai as genai
import chromadb
import os
import time
import random
import string
import pandas as pd
from datetime import datetime
import re # Import re for summary query detection

# Import functions from our modules
from config import (
    DEFAULT_COLLECTION_NAME_PREFIX, ALL_SUPPORTED_TYPES,
    VECTOR_DB_PERSIST_PATH, CHAT_DB_DIRECTORY,
    TEXT_MODEL_NAME as DEFAULT_TEXT_MODEL_NAME,
    SUPPORTED_DATA_TYPES, SUPPORTED_TEXT_TYPES # Added text types
)
from utils import log_message, generate_unique_id
from vector_store import initialize_embedding_function, process_and_embed, get_relevant_context
from qa_engine import is_data_query, answer_data_query, generate_answer_from_context, generate_followup_questions

# --- Page Configuration ---
# ... (keep as is) ...
st.set_page_config(
    page_title="Persistent Multi-Modal Q&A + Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Persistent Multi-Modal Q&A & Data Analysis")
st.caption("Switch chats, upload docs/data, ask questions about text, images, or analyze your CSV/Excel/JSON!")


# --- Constants for Model Selection ---
AVAILABLE_TEXT_MODELS = ["gemini-2.0-flash"]

# --- Global Variables / Initializations ---
# ... (keep initialize_chroma_client as is) ...
@st.cache_resource
def initialize_chroma_client():
    """Initializes ChromaDB client, cached for the session."""
    log_message("Initializing ChromaDB client...", "debug")
    persist_path = VECTOR_DB_PERSIST_PATH
    if persist_path and os.path.exists(persist_path):
        log_message(f"Using persistent ChromaDB storage at: {persist_path}", "info")
        try:
            client = chromadb.PersistentClient(path=persist_path)
            return client
        except Exception as e:
             log_message(f"Error loading persistent ChromaDB from {persist_path}: {e}. Falling back to in-memory.", "error")
             log_message("Using in-memory ChromaDB storage.", "info")
             client = chromadb.Client()
             return client
    else:
        if persist_path:
            log_message(f"Persistent path {persist_path} not found or invalid. Using in-memory ChromaDB storage.", "warning")
        else:
            log_message("Using in-memory ChromaDB storage.", "info")
        client = chromadb.Client()
        return client

# --- Session State Management ---
# ... (keep create_new_chat_state, initialize_multi_chat_state, get_current_chat_state as is) ...
def create_new_chat_state(chat_id):
    """Creates the default state dictionary for a new chat."""
    return {
        "chat_id": chat_id,
        "collection_name": DEFAULT_COLLECTION_NAME_PREFIX + chat_id,
        "messages": [{"role": "assistant", "content": "Hi there! Upload documents (PDF, DOCX, TXT, images) or data files (CSV, XLSX, JSON) to get started. Ask questions about your uploads!"}],
        "processed_files": {},
        "chat_db_path": None,
        "imported_tables": [],
        "crawled_data": {},
        "processing_status": "idle",
        "data_import_status": "idle",
        "created_at": datetime.now().isoformat()
    }

def initialize_multi_chat_state():
    """Initializes the main chat structure if it doesn't exist."""
    if "chats" not in st.session_state:
        st.session_state.chats = {}
        if "processing_log" not in st.session_state: st.session_state.processing_log = []
        log_message("Initializing multi-chat state.", "info")
        first_chat_id = generate_unique_id()
        st.session_state.chats[first_chat_id] = create_new_chat_state(first_chat_id)
        st.session_state.current_chat_id = first_chat_id
        log_message(f"Created initial chat: {first_chat_id}", "info")
    elif "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
        if st.session_state.chats:
            valid_chats = {k: v for k, v in st.session_state.chats.items() if isinstance(v, dict)}
            if valid_chats:
                most_recent_chat_id = max(valid_chats.keys(), key=lambda k: valid_chats[k].get('created_at', ''))
                st.session_state.current_chat_id = most_recent_chat_id
                log_message(f"Set current chat to most recent: {most_recent_chat_id}", "debug")
            else:
                new_chat_id = generate_unique_id()
                st.session_state.chats[new_chat_id] = create_new_chat_state(new_chat_id)
                st.session_state.current_chat_id = new_chat_id
                log_message("No valid chats found, created new one: {new_chat_id}", "info")
        else:
            new_chat_id = generate_unique_id()
            st.session_state.chats[new_chat_id] = create_new_chat_state(new_chat_id)
            st.session_state.current_chat_id = new_chat_id
            log_message("No chats found, created new one: {new_chat_id}", "info")

    # Ensure other global states exist
    if "api_key_configured" not in st.session_state: st.session_state.api_key_configured = False
    if "embedding_func" not in st.session_state: st.session_state.embedding_func = None
    if "rerun_query" not in st.session_state: st.session_state.rerun_query = None
    if "selected_text_model" not in st.session_state:
        default_model = DEFAULT_TEXT_MODEL_NAME
        if default_model not in AVAILABLE_TEXT_MODELS and AVAILABLE_TEXT_MODELS:
             default_model = AVAILABLE_TEXT_MODELS[0]
        elif not AVAILABLE_TEXT_MODELS:
             default_model = "gemini-1.5-flash" # Absolute fallback
             log_message("Warning: AVAILABLE_TEXT_MODELS list might be empty or default invalid.", "warning")
        st.session_state.selected_text_model = default_model

initialize_multi_chat_state()

def get_current_chat_state():
    """Safely retrieves the state dictionary for the currently active chat."""
    current_id = st.session_state.get("current_chat_id")
    if not current_id or current_id not in st.session_state.get("chats", {}):
        initialize_multi_chat_state()
        current_id = st.session_state.get("current_chat_id")
        if not current_id or current_id not in st.session_state.get("chats", {}):
             log_message("Critical Error: Cannot determine current chat state after re-initialization.", "error")
             st.error("Critical error loading chat state. Please refresh.")
             return None
    chat_state = st.session_state.chats.get(current_id)
    if not isinstance(chat_state, dict):
         log_message(f"Critical Error: Chat state for ID {current_id} is not a dictionary.", "error")
         st.error(f"Chat session data for '{current_id[-4:]}' seems corrupted. Please start a new chat.")
         return None
    return chat_state

# --- API Key Handling & Model Selection (Sidebar) ---
# ... (keep as is) ...
with st.sidebar:
    st.header("⚙️ Configuration")
    # Model Selection
    current_selection = st.session_state.get("selected_text_model", DEFAULT_TEXT_MODEL_NAME)
    if current_selection not in AVAILABLE_TEXT_MODELS:
        current_selection = AVAILABLE_TEXT_MODELS[0] if AVAILABLE_TEXT_MODELS else "gemini-1.5-flash"
    selected_model = st.selectbox(
        "Choose Text Model:", options=AVAILABLE_TEXT_MODELS, key="selected_text_model",
        index=AVAILABLE_TEXT_MODELS.index(current_selection) if current_selection in AVAILABLE_TEXT_MODELS else 0,
        help="Select the language model for answering questions (used for both RAG and Data Analysis)."
    )
    # API Key Input
    st.header("🔑 API Configuration")
    try:
        input_key = st.session_state.get("api_key_input_sidebar", "")
        current_key = st.session_state.get("google_api_key", "")
        env_api_key = os.environ.get("GOOGLE_API_KEY")
        secrets_api_key = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, 'secrets') else None
        display_value = input_key or current_key or env_api_key or secrets_api_key or ""
        google_api_key_input = st.text_input(
            "Enter Google AI API Key:", type="password", key="api_key_input_sidebar",
            value=display_value, help="Get key from Google AI Studio. Overrides env/secrets for this session."
        )
        # Determine final key based on input priority
        if google_api_key_input and google_api_key_input != display_value:
             final_google_api_key = google_api_key_input
             st.session_state.google_api_key = google_api_key_input # Store user input
        else:
             final_google_api_key = current_key or env_api_key or secrets_api_key # Use existing or fallback

        # Store the determined key if it's different from current session key or if none is stored
        if final_google_api_key and final_google_api_key != st.session_state.get("google_api_key"):
             st.session_state.google_api_key = final_google_api_key
        elif not final_google_api_key and "google_api_key" in st.session_state:
              del st.session_state.google_api_key # Clear if no key is available

    except Exception as e:
        st.error(f"Error accessing secrets/env vars: {e}")
        final_google_api_key = None
        st.session_state.google_api_key = None

# --- Configure Google AI API and Embedding Function ---
# ... (keep as is) ...
if final_google_api_key and not st.session_state.api_key_configured:
    try:
        log_message("Attempting to configure Google AI API...", "debug")
        genai.configure(api_key=final_google_api_key)
        st.session_state.embedding_func = initialize_embedding_function(final_google_api_key)
        if st.session_state.embedding_func:
             st.session_state.api_key_configured = True
             st.session_state.google_api_key = final_google_api_key
             log_message("Google AI API configured and embedding function ready.", "info")
        else:
             st.session_state.api_key_configured = False
             log_message("Embedding function initialization failed.", "error")
             if "google_api_key" in st.session_state: del st.session_state.google_api_key
    except Exception as e:
        log_message(f"Error configuring Google AI API: {e}", "error")
        st.session_state.api_key_configured = False
        st.session_state.embedding_func = None
        if "google_api_key" in st.session_state: del st.session_state.google_api_key

# --- Display API status ---
# ... (keep as is) ...
with st.sidebar:
    if st.session_state.get("api_key_configured"):
        st.success("API Key Configured.")
    elif final_google_api_key:
         st.error("API Key Invalid or Configuration Failed.")
    else:
         st.warning("API Key Needed.")

# --- Initialize ChromaDB Client ---
chroma_client = initialize_chroma_client()

# --- File Processing Callback ---
# ... (keep handle_file_upload as is) ...
def handle_file_upload():
    """Callback for file uploader. Processes files for the CURRENT chat."""
    log_message("File uploader callback started.", "debug")
    current_chat_id = st.session_state.get("current_chat_id")
    if not current_chat_id:
         log_message("Upload callback: No current chat ID.", "error")
         st.toast("⚠️ Error: No active chat selected.", icon="⚠️")
         return

    chat_state = get_current_chat_state()
    if not chat_state:
        log_message("Upload callback: Could not get chat state.", "error")
        st.toast("⚠️ Error: Could not load chat state.", icon="⚠️")
        return

    if not st.session_state.api_key_configured: # Check key *before* proceeding
         st.toast("⚠️ Please configure a valid Google AI API Key first.", icon="🔑")
         return

    uploaded_files = st.session_state.get("doc_uploader_key", [])
    if not uploaded_files:
        log_message("Upload callback: No files found in uploader state.", "warning")
        return

    # Set status to processing
    chat_state["processing_status"] = "processing"
    chat_state["data_import_status"] = "idle" # Reset data status specifically

    log_message(f"Callback: Set Chroma status to 'processing' for chat {current_chat_id}", "debug")

    # --- Start processing ---
    log_message(f"Callback: Starting processing for {len(uploaded_files)} files in chat {current_chat_id}...", "info")
    chat_state["processed_files"] = {} # Reset file status for this batch
    log_message(f"[{current_chat_id}] Resetting file status for new upload...")
    log_message(f"[{current_chat_id}] Processing {len(uploaded_files)} files...")

    client = initialize_chroma_client()
    emb_func = st.session_state.embedding_func # Should be ready if api_key_configured is True
    collection = None
    chroma_success = False
    data_import_occurred = False
    data_import_success = True # Assume true unless a data file fails

    if not client or not emb_func:
         log_message(f"[{current_chat_id}] Cannot process files: Client/Embedding func unavailable.", "error")
         chat_state["processing_status"] = "error"
         chat_state["data_import_status"] = "error"
         return

    collection_name = chat_state["collection_name"]
    try:
        log_message(f"[{current_chat_id}] Getting or creating collection: '{collection_name}'.", "info")
        collection = client.get_or_create_collection(
             name=collection_name, embedding_function=emb_func, metadata={"hnsw:space": "cosine"}
         )
        log_message(f"[{current_chat_id}] Collection ready: '{collection_name}'.", "info")
    except Exception as e:
         log_message(f"[{current_chat_id}] Fatal Error getting/creating ChromaDB collection: {e}", "error")
         st.exception(e)
         chat_state["processing_status"] = "error"
         chat_state["data_import_status"] = "error"
         return

    # --- Call process_and_embed ---
    if collection and emb_func:
         log_message(f"[{current_chat_id}] Calling process_and_embed...", "debug")
         # process_and_embed updates chat_state["processed_files"] internally via file_parsers
         chroma_add_attempted = process_and_embed(uploaded_files, collection, emb_func)
         # Check overall chroma success by seeing if *any* file associated with chroma add succeeded
         chroma_success = any(status == 'success' for filename, status in chat_state.get("processed_files", {}).items()
                              if not filename.lower().endswith(tuple(f".{ext}" for ext in SUPPORTED_DATA_TYPES)))


         # Check individual file statuses for data import outcome
         data_file_extensions = tuple(f".{ext}" for ext in SUPPORTED_DATA_TYPES)
         for f in uploaded_files:
             if f.name.lower().endswith(data_file_extensions):
                 data_import_occurred = True # Mark that we processed at least one data file
                 status = chat_state["processed_files"].get(f.name)
                 if status != 'success': # Consider anything not success as failure for data import
                     data_import_success = False
                     log_message(f"Data import for {f.name} marked as {status}.", "warning")
                     # break # Let all files process, report aggregate status

    else:
         log_message(f"[{current_chat_id}] Processing aborted: Collection/Embedding func unavailable.", "error")
         chroma_success = False
         data_import_success = False

    # Update final statuses
    chat_state["processing_status"] = "success" if chroma_success else "error"
    # Determine final data status
    if not data_import_occurred:
         chat_state["data_import_status"] = "idle" # No data files attempted
    elif data_import_success:
         chat_state["data_import_status"] = "success"
         st.toast(f"📊 Data file(s) imported successfully into chat database!", icon="📊")
    else:
         chat_state["data_import_status"] = "error"

    log_message(f"Callback: Set Chroma status to '{chat_state['processing_status']}' for chat {current_chat_id}", "debug")
    log_message(f"Callback: Set Data Import status to '{chat_state['data_import_status']}' for chat {current_chat_id}", "debug")

# --- Helper to get Chroma Collection ---
# ... (keep as is) ...
def get_current_collection():
    """Gets the ChromaDB collection object for the current chat."""
    chat_state = get_current_chat_state()
    if not chat_state or not st.session_state.api_key_configured or not st.session_state.embedding_func:
        log_message("Cannot get collection: Chat state or API/Embedding config missing.", "warning")
        return None
    collection_name = chat_state.get("collection_name")
    if not collection_name:
         log_message("Cannot get collection: Collection name missing in chat state.", "warning")
         return None
    try:
        client = initialize_chroma_client()
        collection = client.get_collection(name=collection_name, embedding_function=st.session_state.embedding_func)
        return collection
    except Exception as e:
        log_message(f"Collection '{collection_name}' not found or error getting it (may need upload). Error: {e}", "debug")
        return None

# --- Sidebar UI ---
# ... (keep as is, including deletion logic) ...
with st.sidebar:
    # Chat Management
    st.header("📄 Chat Management")
    if st.button("➕ New Chat", key="new_chat_button", use_container_width=True):
        new_chat_id = generate_unique_id()
        st.session_state.chats[new_chat_id] = create_new_chat_state(new_chat_id)
        st.session_state.current_chat_id = new_chat_id
        log_message(f"Created and switched to new chat: {new_chat_id}", "info")
        st.rerun()
    st.divider()
    # Chat History List
    st.header("🗂️ Chat Sessions")
    sorted_chat_ids = sorted(
        st.session_state.chats.keys(),
        key=lambda k: st.session_state.chats.get(k, {}).get('created_at', ''),
        reverse=True
    )
    chat_list_container = st.container(height=300)
    with chat_list_container:
        if not sorted_chat_ids:
            st.caption("No chat sessions yet.")
        else:
            current_chat_id_local = st.session_state.get("current_chat_id")
            chat_to_delete = None
            for chat_id in sorted_chat_ids:
                chat_state = st.session_state.chats.get(chat_id)
                if not isinstance(chat_state, dict): continue # Skip invalid
                # Generate label
                first_user_message = next((msg['content'] for msg in chat_state.get('messages', []) if msg.get('role') == 'user'), None)
                label_base = first_user_message[:30] + "..." if first_user_message else f"Chat {chat_id[-4:]}"
                has_db = bool(chat_state.get("chat_db_path"))
                label = f"{'📊 ' if has_db else ''}{label_base}" # Add icon if DB exists
                created_at_str = chat_state.get('created_at', None)
                timestamp = datetime.fromisoformat(created_at_str).strftime('%b %d, %H:%M') if created_at_str else "???"
                button_label = f"{label} ({timestamp})"
                button_key = f"switch_chat_{chat_id}"
                delete_button_key = f"delete_chat_{chat_id}"
                button_type = "primary" if chat_id == current_chat_id_local else "secondary"
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    if st.button(button_label, key=button_key, use_container_width=True, type=button_type):
                        if st.session_state.current_chat_id != chat_id:
                            log_message(f"Switching to chat: {chat_id}", "info")
                            st.session_state.current_chat_id = chat_id
                            st.rerun()
                with col2:
                    if st.button("🗑️", key=delete_button_key, help=f"Delete chat '{label_base}'"):
                        chat_to_delete = chat_id
                        log_message(f"Delete button clicked for chat: {chat_id}", "info")
            # Perform Deletion After Iteration
            if chat_to_delete:
                log_message(f"Performing deletion of chat: {chat_to_delete}", "info")
                chat_state_to_delete = st.session_state.chats.get(chat_to_delete)
                # Delete Chroma Collection
                if chat_state_to_delete:
                    collection_name_to_delete = chat_state_to_delete.get('collection_name')
                    if collection_name_to_delete:
                        try:
                            log_message(f"Deleting Chroma collection: {collection_name_to_delete}", "info")
                            chroma_client.delete_collection(name=collection_name_to_delete)
                        except Exception as e:
                            log_message(f"Chroma Collection '{collection_name_to_delete}' not found or delete failed: {e}", "warning")
                    # Delete Associated SQLite Database File
                    db_path_to_delete = chat_state_to_delete.get('chat_db_path')
                    if db_path_to_delete and os.path.exists(db_path_to_delete):
                         try:
                              log_message(f"Deleting chat database file: {db_path_to_delete}", "info")
                              os.remove(db_path_to_delete)
                              log_message(f"Successfully deleted database file: {db_path_to_delete}", "info")
                         except OSError as e:
                              log_message(f"Error deleting database file '{db_path_to_delete}': {e}", "error")
                              st.warning(f"Could not delete database file for chat {chat_to_delete[-4:]}. Manual cleanup might be needed.")
                    elif db_path_to_delete:
                         log_message(f"Database file path recorded but not found: {db_path_to_delete}", "debug")
                # Delete Chat State from Session
                if chat_to_delete in st.session_state.chats:
                    del st.session_state.chats[chat_to_delete]
                    log_message(f"Deleted chat state for: {chat_to_delete}", "info")
                # Handle if the deleted chat was the current one
                if st.session_state.current_chat_id == chat_to_delete:
                    log_message(f"Current chat ({chat_to_delete}) was deleted. Switching...", "info")
                    remaining_chats = {k:v for k,v in st.session_state.chats.items() if isinstance(v, dict)}
                    if remaining_chats:
                        new_current_chat_id = max(remaining_chats.keys(), key=lambda k: remaining_chats[k].get('created_at', ''))
                        st.session_state.current_chat_id = new_current_chat_id
                        log_message(f"Switched current chat to: {new_current_chat_id}", "info")
                    else:
                        log_message("No chats remaining, creating a new initial chat.", "info")
                        st.session_state.chats = {}
                        initialize_multi_chat_state()
                st.rerun() # Rerun to update the UI

# --- Helper Function to Detect Summary Queries ---
def is_summary_query(query: str) -> bool:
    """Checks if a query is likely asking for a general summary."""
    query_lower = query.lower().strip()
    summary_keywords = [
        "what is this document about", "summarize this document", "summarize",
        "summary of", "tell me about this doc", "overview of", "what does this file say",
        "give me a summary", "general topic", "main points"
    ]
    # Check for exact matches or variations
    if query_lower in summary_keywords:
        return True
    # Check for start patterns
    if query_lower.startswith(("summarize", "what is", "tell me about", "overview of")):
        # Add checks for document references if needed, e.g., "summarize the pdf"
        if "document" in query_lower or "file" in query_lower or "pdf" in query_lower or "doc" in query_lower:
            return True
    return False

# --- Main Chat Area UI ---
st.header("💬 Chat Interface")
current_chat_state = get_current_chat_state()

# Display chat messages for the current chat
message_container = st.container()
with message_container:
    if not current_chat_state:
         st.info("Select a chat session or start a new one from the sidebar.")
    else:
        # Display messages
        for i, message in enumerate(current_chat_state.get("messages", [])):
            with st.chat_message(message["role"]):
                # Display content
                st.markdown(message["content"], unsafe_allow_html=False)

                # --- Display Full DataFrame Button & Content ---
                if message["role"] == "assistant" and "dataframe_result" in message:
                    df_result = message["dataframe_result"]
                    if isinstance(df_result, pd.DataFrame) and not df_result.empty:
                        button_key = f"show_df_{current_chat_state.get('chat_id','_')}_{i}"
                        if f"show_df_state_{button_key}" not in st.session_state:
                             st.session_state[f"show_df_state_{button_key}"] = False
                        # Use columns for button layout
                        btn_cols = st.columns(4) # Adjust column count as needed
                        with btn_cols[0]:
                             if st.button("Show/Hide Full Table", key=button_key):
                                 st.session_state[f"show_df_state_{button_key}"] = not st.session_state[f"show_df_state_{button_key}"]
                                 st.rerun()
                        # Display dataframe if state is True
                        if st.session_state.get(f"show_df_state_{button_key}", False):
                            st.dataframe(df_result)

                # Display sources
                if message["role"] == "assistant" and message.get("metadata"):
                     exp_col1, exp_col2 = st.columns([0.9, 0.1])
                     with exp_col1:
                         with st.expander("Sources Used", expanded=False):
                             sources_display = []
                             processed_sources = set()
                             for meta in message["metadata"]:
                                if not meta: continue
                                source = meta.get('source', 'Unknown')
                                content_type = meta.get('content_type','unknown')
                                db_source = meta.get('database')

                                if db_source and content_type == 'database_query_result':
                                    source_key = f"db_{db_source}"
                                    if source_key not in processed_sources:
                                         tables_list_str = meta.get('tables', 'unknown')
                                         display_str = f"- **Chat Database** (Tables: {tables_list_str})"
                                         sources_display.append(display_str)
                                         processed_sources.add(source_key)
                                elif content_type != 'database_query_result':
                                    page_num = meta.get('page_number')
                                    slide_num = meta.get('slide_number')
                                    crawled = meta.get('crawled_url')
                                    source_key = f"{source}"
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
                                        elif content_type == 'text': display_str += " (Text Content)" # Added
                                        elif content_type == 'slide_text': display_str += f" (Slide {slide_num} Text)" # Added
                                        sources_display.append(display_str)
                                        processed_sources.add(source_key)
                             st.markdown("\n".join(sorted(sources_display)))

                # Display follow-up buttons
                if message["role"] == "assistant" and message.get("follow_ups"):
                     st.markdown("**Suggested Questions:**")
                     cols = st.columns(len(message["follow_ups"]))
                     for j, q in enumerate(message["follow_ups"]):
                          if cols[j].button(q, key=f"followup_{current_chat_state.get('chat_id','_')}_{i}_{j}"):
                               st.session_state.rerun_query = q
                               st.rerun()

# --- Input Area ---
# ... (keep as is) ...
input_container = st.container()
with input_container:
    # Handle follow-up query state
    query_input_value = ""
    if st.session_state.rerun_query:
        query_input_value = st.session_state.rerun_query
        st.session_state.rerun_query = None

    # Chat input field
    chat_input_disabled = not st.session_state.api_key_configured
    chat_input_placeholder = "Ask about documents OR analyze CSV/Excel/JSON data..." if st.session_state.api_key_configured else "Please configure API Key in sidebar"
    prompt = st.chat_input(
        chat_input_placeholder, key="main_query_input", disabled=chat_input_disabled
    )

    # File uploader
    st.file_uploader(
            "Attach Files (Docs, Images, Data) for **this chat**",
            accept_multiple_files=True, type=ALL_SUPPORTED_TYPES, key="doc_uploader_key",
            on_change=handle_file_upload, label_visibility="collapsed", disabled=chat_input_disabled
        )

    # --- Processing Status Display Area ---
    status_placeholder = st.empty()
    current_chroma_status = "idle"
    current_data_status = "idle"
    if current_chat_state:
        current_chroma_status = current_chat_state.get("processing_status", "idle")
        current_data_status = current_chat_state.get("data_import_status", "idle")

    status_messages = []
    # Combine status logic
    if current_chroma_status == "processing" or current_data_status == "importing": # Use distinct status if available
         status_messages.append("⏳ Processing uploaded files...")
    elif current_chroma_status == "success" and current_data_status in ["idle", "success"]:
         status_messages.append("✅ Files processed successfully.")
         if current_data_status == "success": status_messages.append("📊 Data imported.")
    elif current_chroma_status == "error" or current_data_status == "error":
         status_messages.append("⚠️ File processing failed.")
         if current_data_status == "error": status_messages.append("Data import failed.")
    elif current_chroma_status == "success" and current_data_status == "idle": # Text processed, no data files or data idle
         status_messages.append("✅ Text/Image documents processed.")

    if status_messages:
         full_message = " ".join(status_messages)
         if "⚠️" in full_message: status_placeholder.error(full_message)
         elif "⏳" in full_message: status_placeholder.info(full_message)
         else: status_placeholder.success(full_message)


    # --- Query Handling Logic (MODIFIED FOR SUMMARY) ---
    actual_prompt = prompt or query_input_value

    if actual_prompt and current_chat_state:
        current_chat_id_for_query = current_chat_state["chat_id"]
        selected_model = st.session_state.selected_text_model

        current_chat_state.setdefault("messages", []).append({"role": "user", "content": actual_prompt})

        # --- Determine how to respond ---
        assistant_response_content = None
        follow_ups = []
        used_metadata = []
        response_df = None
        prompt_lower = actual_prompt.lower().strip()
        greetings = ["hi", "hello", "hey", "yo", "greetings", "good morning", "good afternoon", "good evening"]
        help_keywords = ["help", "what can you do", "capabilities", "features", "functions", "what are your functions"]

        # 1. Handle conversational prompts
        is_greeting = any(greet == prompt_lower for greet in greetings)
        is_help_request = any(keyword in prompt_lower for keyword in help_keywords)

        if is_greeting:
            assistant_response_content = random.choice(["Hello!", "Hi there!", "Hey!"]) + " How can I help? Upload files or ask a question."
        elif is_help_request:
            # ... (keep help text as is) ...
            assistant_response_content = f"""I can help with the following in this chat session:
1.  **Process Files:** Upload documents (PDF, DOCX, TXT, PNG, JPG) or data files (CSV, XLSX, JSON). Text/image content is indexed for retrieval. Data files are loaded into a temporary database for this chat only.
2.  **Answer Document Questions:** Ask questions based on the text/image documents uploaded *in this chat*. Uses Google's embedding model and the selected text model (`{selected_model}`).
3.  **Analyze Data:** Ask questions in natural language about the CSV, Excel, or JSON files uploaded *in this chat* (e.g., "show me total sales", "what are the average prices per category?", "list customers from London"). This uses the selected text model (`{selected_model}`) to generate and run SQL queries on the chat's database.

Use the 'New Chat' button in the sidebar to start fresh with different documents or data."""
        # 2. Check processing status
        elif current_chroma_status == "processing" or current_data_status == "importing":
            assistant_response_content = "Hold on! Files are still processing for this chat. Please wait."
        elif not st.session_state.api_key_configured:
             assistant_response_content = "I need a valid Google AI API Key configured in the sidebar."
        else:
            # 3. Determine query type: Data Analysis vs. RAG
            chat_has_database = current_chat_state.get("chat_db_path") is not None
            looks_like_data_query = is_data_query(actual_prompt)
            looks_like_summary_query = is_summary_query(actual_prompt) # <-- Check for summary

            if chat_has_database and looks_like_data_query and not looks_like_summary_query: # Only data query if NOT summary
                log_message(f"[{current_chat_id_for_query}] Query routed to Data Analysis Engine.", "info")
                with st.spinner("Analyzing data and thinking..."):
                     assistant_response_content, used_metadata, response_df = answer_data_query(
                         actual_prompt, selected_model
                     )

            else:
                # 4. Handle RAG (including summary queries which use RAG but with filtering)
                log_message(f"[{current_chat_id_for_query}] Query routed to Document Q&A (RAG). Summary Query: {looks_like_summary_query}", "info")
                current_collection = get_current_collection()
                if current_collection is None:
                      files_were_processed = bool(current_chat_state.get("processed_files"))
                      if not files_were_processed:
                          assistant_response_content = f"It looks like you're asking about specific information, but no documents or data files have been processed in this chat ('{current_chat_id_for_query[-4:]}') yet. Please upload the relevant file(s)."
                      else:
                          assistant_response_content = f"I don't have text content to search for this chat ('{current_chat_id_for_query[-4:]}'). If you uploaded data files (CSV/Excel/JSON), try asking specific questions about the data (e.g., 'how many rows?', 'what is the total amount?'). If you uploaded documents, please try uploading them again."
                          log_message(f"[{current_chat_id_for_query}] RAG query, but collection is None or potentially empty.", "warning")
                else:
                    with st.spinner("Searching documents and thinking..."):
                        # Retrieve context
                        context_docs, context_metadatas = get_relevant_context(
                            actual_prompt, current_collection
                        )

                        # --- Filter context IF it's a summary query ---
                        if looks_like_summary_query and context_metadatas:
                             log_message(f"[{current_chat_id_for_query}] Filtering context for summary query.", "debug")
                             filtered_docs = []
                             filtered_metadatas = []
                             # Define content types prioritized for summaries
                             summary_content_types = {'text', 'slide_text', 'ocr_page', 'crawled_web'}
                             for doc, meta in zip(context_docs, context_metadatas):
                                 if meta and meta.get('content_type') in summary_content_types:
                                      filtered_docs.append(doc)
                                      filtered_metadatas.append(meta)
                             if not filtered_docs:
                                 log_message(f"[{current_chat_id_for_query}] No primary text content found after filtering for summary. Using original context.", "warning")
                                 # Optionally: Fallback to using original context if filtering removes everything
                                 # Or let it proceed with empty context below
                             else:
                                 context_docs = filtered_docs
                                 context_metadatas = filtered_metadatas
                        # --- End Filtering ---

                        if not context_docs:
                             # Provide a more specific message if summary filtering led to empty context
                             if looks_like_summary_query:
                                  assistant_response_content = "I couldn't find enough text content in the uploaded documents to provide a general summary."
                             else:
                                 assistant_response_content = "I couldn't find relevant information in the uploaded documents to answer that specific question."
                             used_metadata = []
                        else:
                             assistant_response_content, used_metadata = generate_answer_from_context(
                                 actual_prompt, context_docs, context_metadatas, selected_model
                             )
                             # Generate follow-ups only for non-summary RAG answers?
                             if not looks_like_summary_query and assistant_response_content and isinstance(assistant_response_content, str) and \
                                "cannot answer" not in assistant_response_content.lower() and \
                                "error" not in assistant_response_content.lower() and \
                                "blocked" not in assistant_response_content.lower():
                                  follow_ups = generate_followup_questions(actual_prompt, assistant_response_content, selected_model)


        # --- Add Assistant Response ---
        if assistant_response_content is not None:
            assistant_message = {
                "role": "assistant",
                "content": assistant_response_content,
                "follow_ups": follow_ups,
                "metadata": used_metadata
            }
            if response_df is not None and isinstance(response_df, pd.DataFrame):
                assistant_message["dataframe_result"] = response_df
            current_chat_state["messages"].append(assistant_message)

        st.rerun()


# --- Footer / Disclaimer ---
st.divider()
st.caption("AI responses based on documents/data in the current chat. Verify critical info. Use 'New Chat' for separate contexts.")

# --- END OF FILE main_app.py ---
