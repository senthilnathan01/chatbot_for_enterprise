# main_app.py
"""
Main Streamlit application file for the Multi-Modal Q&A Chatbot.
Handles UI, state management, and orchestrates calls to other modules.
V5.3 Changes:
- Added LLM model selection dropdown in sidebar.
- Removed Processing Log display from sidebar.
- Passes selected LLM model name to qa_engine functions.
- Added delete button (🗑️) for each chat session in the sidebar.
- Implemented logic to delete chat state and associated ChromaDB collection.
- Handles switching to another chat if the current one is deleted.
- Confirmed removal of direct assignment to st.session_state.doc_uploader_key.
- Improved logging and state handling in handle_file_upload callback.
- Removed st.rerun() from within handle_file_upload.
- Ensured processing status display below uploader reflects current chat state.
- Ensured conversational logic handles greetings/help correctly post-upload.
- Fixed TypeError by removing argument from generate_unique_id() calls.
"""

import streamlit as st
import google.generativeai as genai
import chromadb
import os
import time
import random
import string
from datetime import datetime

# Import functions from our modules
from config import (
    DEFAULT_COLLECTION_NAME_PREFIX, ALL_SUPPORTED_TYPES,
    VECTOR_DB_PERSIST_PATH, CHAT_DB_DIRECTORY, 
    # Import default text model name for initialization
    TEXT_MODEL_NAME as DEFAULT_TEXT_MODEL_NAME
)
# Ensure generate_unique_id is imported correctly
from utils import log_message, generate_unique_id
from vector_store import initialize_embedding_function, process_and_embed, get_relevant_context
# Import specific qa_engine functions
from qa_engine import is_data_query, answer_data_query, generate_answer_from_context, generate_followup_questions

# --- Page Configuration ---
st.set_page_config(
    page_title="Persistent Multi-Modal Q&A",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Persistent Multi-Modal Q&A")
st.caption("Switch between chats, upload documents per chat, and ask questions or analyze your Excels!!")

# --- Constants for Model Selection ---
# Define available text models compatible with the library/API key
# Ensure these names are valid according to genai.list_models() output
AVAILABLE_TEXT_MODELS = ["gemini-1.0-pro", "gemini-1.5-pro-latest", "gemini-pro", "gemini-2.0-flash"]

# --- Global Variables / Initializations ---
@st.cache_resource
def initialize_chroma_client():
    """Initializes ChromaDB client, cached for the session."""
    # Use log_message defined in utils
    log_message("Initializing ChromaDB client...", "debug")
    if VECTOR_DB_PERSIST_PATH:
        log_message(f"Using persistent ChromaDB storage at: {VECTOR_DB_PERSIST_PATH}", "info")
        client = chromadb.PersistentClient(path=VECTOR_DB_PERSIST_PATH)
    else:
        log_message("Using in-memory ChromaDB storage.", "info")
        client = chromadb.Client()
    return client

# --- Session State Management for Multi-Chat ---
def create_new_chat_state(chat_id):
    """Creates the default state dictionary for a new chat."""
    return {
        "chat_id": chat_id,
        "collection_name": DEFAULT_COLLECTION_NAME_PREFIX + chat_id,
        "messages": [{"role": "assistant", "content": "Hi there! Upload documents or ask me about my functions to get started."}],
        "processed_files": {},
        #"data_frames": {},
        "chat_db_path": None,
        "imported_tables": [],
        "crawled_data": {},
        "processing_status": "idle",
        "data_import_status": "idle", # idle, processing, success, error
        "created_at": datetime.now().isoformat()
        # Add selected model for this chat? Or keep global? Global for now.
    }

def initialize_multi_chat_state():
    """Initializes the main chat structure if it doesn't exist."""
    if "chats" not in st.session_state:
        st.session_state.chats = {}
        if "processing_log" not in st.session_state: st.session_state.processing_log = [] # Global log for now
        log_message("Initializing multi-chat state.", "info")
        # *** Use generate_unique_id() without arguments ***
        first_chat_id = generate_unique_id()
        st.session_state.chats[first_chat_id] = create_new_chat_state(first_chat_id)
        st.session_state.current_chat_id = first_chat_id
        log_message(f"Created initial chat: {first_chat_id}", "info")
    elif "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
        if st.session_state.chats:
            most_recent_chat_id = max(st.session_state.chats.keys(), key=lambda k: st.session_state.chats[k].get('created_at', ''))
            st.session_state.current_chat_id = most_recent_chat_id
            log_message(f"Set current chat to most recent: {most_recent_chat_id}", "debug")
        else:
            # *** Use generate_unique_id() without arguments ***
            new_chat_id = generate_unique_id()
            st.session_state.chats[new_chat_id] = create_new_chat_state(new_chat_id)
            st.session_state.current_chat_id = new_chat_id
            log_message(f"No valid current chat found, created new one: {new_chat_id}", "info")

    # Ensure other necessary global states exist
    if "api_key_configured" not in st.session_state: st.session_state.api_key_configured = False
    if "embedding_func" not in st.session_state: st.session_state.embedding_func = None
    if "rerun_query" not in st.session_state: st.session_state.rerun_query = None
    # Initialize selected text model (use default from config)
    if "selected_text_model" not in st.session_state:
        # Ensure default model is in the available list, otherwise pick the first available
        if DEFAULT_TEXT_MODEL_NAME in AVAILABLE_TEXT_MODELS:
             st.session_state.selected_text_model = DEFAULT_TEXT_MODEL_NAME
        elif AVAILABLE_TEXT_MODELS:
             st.session_state.selected_text_model = AVAILABLE_TEXT_MODELS[0]
        else:
             st.session_state.selected_text_model = "gemini-1.0-pro" # Fallback if list is empty
             log_message("Warning: AVAILABLE_TEXT_MODELS list is empty in config.", "warning")


initialize_multi_chat_state() # Ensure state is ready

# --- Helper to get current chat state ---
def get_current_chat_state():
    """Safely retrieves the state dictionary for the currently active chat."""
    if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
        initialize_multi_chat_state() # Try to fix state
        if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
             log_message("Critical Error: Cannot determine current chat state.", "error")
             return None
    # Return the actual dictionary reference to allow modifications
    return st.session_state.chats[st.session_state.current_chat_id]

# --- API Key Handling & Model Selection (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- LLM Model Selection ---
    # Ensure the current selection is valid, otherwise default
    current_selection = st.session_state.get("selected_text_model", DEFAULT_TEXT_MODEL_NAME)
    if current_selection not in AVAILABLE_TEXT_MODELS:
        current_selection = AVAILABLE_TEXT_MODELS[0] if AVAILABLE_TEXT_MODELS else DEFAULT_TEXT_MODEL_NAME

    selected_model = st.selectbox(
        "Choose Text Model:",
        options=AVAILABLE_TEXT_MODELS,
        key="selected_text_model", # Automatically stores selection in session state
        index=AVAILABLE_TEXT_MODELS.index(current_selection), # Set default index safely
        help="Select the language model for answering questions."
    )

    # --- API Key Input ---
    st.header("🔑 API Configuration")
    try:
        env_api_key = os.environ.get("GOOGLE_API_KEY")
        secrets_api_key = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, 'secrets') else None
        current_api_key = st.session_state.get("google_api_key")
        google_api_key_input = st.text_input(
            "Enter Google AI API Key:", type="password",
            key="api_key_input_sidebar", value=current_api_key or env_api_key or secrets_api_key or "",
            help="Get key from Google AI Studio. Stored for session."
        )
        if google_api_key_input: st.session_state.google_api_key = google_api_key_input
        else:
            if "google_api_key" in st.session_state: del st.session_state.google_api_key
        final_google_api_key = st.session_state.get("google_api_key") or env_api_key or secrets_api_key
    except Exception as e:
        st.error("Error accessing secrets/env vars.")
        final_google_api_key = None

# --- Configure Google AI API and Embedding Function ---
# This needs to run early to enable other components
if final_google_api_key and not st.session_state.api_key_configured:
    try:
        genai.configure(api_key=final_google_api_key)
        # Pass the key explicitly to the initializer function
        st.session_state.embedding_func = initialize_embedding_function(final_google_api_key)
        if st.session_state.embedding_func:
             st.session_state.api_key_configured = True
             log_message("Google AI API configured and embedding function ready.", "info")
             # Success message moved below
        else:
             # Error message moved below
             st.session_state.api_key_configured = False
    except Exception as e:
        # Error message moved below
        log_message(f"Error configuring Google AI API: {e}", "error")
        st.session_state.api_key_configured = False
        st.session_state.embedding_func = None

# Display API status consistently in sidebar
with st.sidebar:
    if st.session_state.api_key_configured:
        st.success("API Key Configured.")
    elif final_google_api_key: # Key entered but config failed
         st.error("API Key Invalid or Configuration Failed.")
    else: # No key provided
         st.warning("API Key Needed.")


# --- Initialize ChromaDB Client ---
chroma_client = initialize_chroma_client()

# --- File Processing Callback ---
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

    if not st.session_state.api_key_configured or not st.session_state.embedding_func:
         st.toast("⚠️ Please configure a valid Google AI API Key first.", icon="🔑")
         chat_state["processing_status"] = "idle"
         return

    uploaded_files = st.session_state.get("doc_uploader_key", [])
    if not uploaded_files:
        log_message("Upload callback: No files found in uploader state.", "warning")
        return

    # Set status to processing *for the current chat*
    chat_state["processing_status"] = "processing"
    log_message(f"Callback: Set status to 'processing' for chat {current_chat_id}", "debug")
    # Don't rerun here, let the main loop show status

    # --- Start processing ---
    log_message(f"Callback: Starting processing for {len(uploaded_files)} files in chat {current_chat_id}...", "info")
    chat_state["processed_files"] = {} # Reset status for this batch
    chat_state["data_frames"] = {}
    chat_state["crawled_data"] = {}
    log_message(f"[{current_chat_id}] Resetting state for new upload...")
    log_message(f"[{current_chat_id}] Processing {len(uploaded_files)} files...")

    client = initialize_chroma_client()
    emb_func = st.session_state.embedding_func
    collection = None

    if not client or not emb_func:
         log_message(f"[{current_chat_id}] Cannot process files: Client/Embedding func unavailable.", "error")
         chat_state["processing_status"] = "error"
         return # Exit callback

    collection_name = chat_state["collection_name"]
    try:
        log_message(f"[{current_chat_id}] Attempting to delete old collection: '{collection_name}'.", "info")
        client.delete_collection(name=collection_name)
    except Exception as e:
        log_message(f"[{current_chat_id}] Old collection '{collection_name}' not found or delete failed (this is often ok): {e}", "warning")

    try:
         log_message(f"[{current_chat_id}] Creating/getting collection: '{collection_name}'.", "info")
         collection = client.get_or_create_collection(
             name=collection_name, embedding_function=emb_func,
             metadata={"hnsw:space": "cosine"}
         )
         log_message(f"[{current_chat_id}] Collection ready: '{collection_name}'.", "info")
    except Exception as e:
         log_message(f"[{current_chat_id}] Fatal Error creating ChromaDB collection: {e}", "error")
         st.exception(e)
         chat_state["processing_status"] = "error"
         return # Exit callback

    processing_successful = False
    if collection and emb_func:
         log_message(f"[{current_chat_id}] Calling process_and_embed...", "debug")
         # Pass the list of uploaded file objects directly
         success = process_and_embed(
             uploaded_files, collection, emb_func
             # Pass chat_state if process_and_embed needs to update it directly
             # e.g., process_and_embed(..., chat_state=chat_state)
         )
         processing_successful = success
         log_message(f"[{current_chat_id}] process_and_embed returned: {success}", "debug")
    else:
         log_message(f"[{current_chat_id}] Processing aborted: Collection/Embedding func unavailable.", "error")
         processing_successful = False

    # Update final status based on processing result
    chat_state["processing_status"] = "success" if processing_successful else "error"
    log_message(f"Callback: Set status to '{chat_state['processing_status']}' for chat {current_chat_id}", "debug")

    # Do NOT programmatically clear the file uploader state here
    # Do NOT call st.rerun() here - let Streamlit handle rerun after callback finishes.


# --- Helper to get Chroma Collection for Current Chat ---
def get_current_collection():
    """Gets the ChromaDB collection object for the current chat."""
    chat_state = get_current_chat_state()
    if not chat_state or not st.session_state.api_key_configured or not st.session_state.embedding_func:
        return None
    collection_name = chat_state.get("collection_name")
    if not collection_name: return None
    try:
        client = initialize_chroma_client()
        # Use get_collection - it should exist if processing was successful
        collection = client.get_collection(name=collection_name, embedding_function=st.session_state.embedding_func)
        return collection
    except Exception as e:
        # This error might mean the collection just hasn't been created yet for this chat_id
        log_message(f"Collection '{collection_name}' not found for current chat (may need upload). Error: {e}", "debug")
        return None


# --- Sidebar UI ---
with st.sidebar:
    # --- Chat Management ---
    st.header("📄 Chat Management")
    if st.button("➕ New Chat", key="new_chat_button", use_container_width=True):
        # *** Use generate_unique_id() without arguments ***
        new_chat_id = generate_unique_id()
        st.session_state.chats[new_chat_id] = create_new_chat_state(new_chat_id)
        st.session_state.current_chat_id = new_chat_id
        log_message(f"Created and switched to new chat: {new_chat_id}", "info")
        st.rerun()

    st.divider()

    # --- Chat History List ---
    st.header("🗂️ Chat Sessions")
    sorted_chat_ids = sorted(
        st.session_state.chats.keys(),
        key=lambda k: st.session_state.chats[k].get('created_at', ''),
        reverse=True
    )

    chat_list_container = st.container(height=300) # Adjust height as needed
    with chat_list_container:
        if not sorted_chat_ids:
            st.caption("No chat sessions yet.")
        else:
            current_chat_id_local = st.session_state.current_chat_id
            chat_to_delete = None # Flag to store which chat to delete after iterating

            for chat_id in sorted_chat_ids:
                chat_state = st.session_state.chats[chat_id]
                # Generate label for chat button
                first_user_message = next((msg['content'] for msg in chat_state['messages'] if msg['role'] == 'user'), None)
                label = first_user_message[:30] + "..." if first_user_message else f"Chat {chat_id[-4:]}"
                timestamp = datetime.fromisoformat(chat_state.get('created_at', datetime.min.isoformat())).strftime('%b %d, %H:%M')
                button_label = f"{label} ({timestamp})"
                button_key = f"switch_chat_{chat_id}"
                delete_button_key = f"delete_chat_{chat_id}"
                button_type = "primary" if chat_id == current_chat_id_local else "secondary"

                # Use columns for layout: Chat Button | Delete Button
                col1, col2 = st.columns([0.85, 0.15])

                with col1:
                    if st.button(button_label, key=button_key, use_container_width=True, type=button_type):
                        if st.session_state.current_chat_id != chat_id:
                            log_message(f"Switching to chat: {chat_id}", "info")
                            st.session_state.current_chat_id = chat_id
                            st.rerun() # Rerun to load the selected chat's state

                with col2:
                    # Add delete button - use a simple emoji
                    if st.button("🗑️", key=delete_button_key, help=f"Delete chat '{label}'"):
                        # Flag for deletion after loop
                        chat_to_delete = chat_id
                        log_message(f"Delete button clicked for chat: {chat_id}", "info")

            # --- Perform Deletion After Iteration ---
            if chat_to_delete:
                log_message(f"Performing deletion of chat: {chat_to_delete}", "info")
                chat_state_to_delete = st.session_state.chats.get(chat_to_delete)

                # 1. Delete Chroma Collection
                if chat_state_to_delete:
                    collection_name_to_delete = chat_state_to_delete['collection_name']
                    try:
                        log_message(f"Deleting collection: {collection_name_to_delete}", "info")
                        chroma_client.delete_collection(name=collection_name_to_delete)
                    except Exception as e:
                        log_message(f"Collection '{collection_name_to_delete}' not found or delete failed: {e}", "warning")

                # 2. Delete Chat State from Session
                if chat_to_delete in st.session_state.chats:
                    del st.session_state.chats[chat_to_delete]
                    log_message(f"Deleted chat state for: {chat_to_delete}", "info")

                # 3. Handle if the deleted chat was the current one
                if st.session_state.current_chat_id == chat_to_delete:
                    log_message(f"Current chat ({chat_to_delete}) was deleted. Switching...", "info")
                    remaining_chats = st.session_state.chats
                    if remaining_chats:
                        new_current_chat_id = max(remaining_chats.keys(), key=lambda k: remaining_chats[k].get('created_at', ''))
                        st.session_state.current_chat_id = new_current_chat_id
                        log_message(f"Switched current chat to: {new_current_chat_id}", "info")
                    else:
                        log_message("No chats remaining, creating a new initial chat.", "info")
                        initialize_multi_chat_state() # Creates and sets a new current chat

                # 4. Rerun to update the UI
                st.rerun()

    # --- REMOVED Global Processing Log Display ---
    # st.divider()
    # st.header("📜 Global Processing Log") # Removed Header
    # log_container = st.container(height=250) # Removed Container
    # log_placeholder = log_container.empty() # Removed Placeholder
    # log_placeholder.text_area(...) # Removed Text Area


# --- Main Chat Area UI ---
st.header("💬 Chat Interface")

# Get state for the *currently selected* chat
current_chat_state = get_current_chat_state()

# Display chat messages for the current chat
message_container = st.container()
with message_container:
    if not current_chat_state:
         # Handle state not ready (e.g., during initial load or after error)
         st.info("Select a chat session or start a new one from the sidebar.")
         # Optionally disable input if no chat state?
    else:
        # Display messages for the current chat
        for i, message in enumerate(current_chat_state["messages"]):
            with st.chat_message(message["role"]):
                st.markdown(message["content"]) # Render markdown content
                # Display sources
                if message["role"] == "assistant" and message.get("metadata"):
                     with st.expander("Sources Used", expanded=False):
                          sources_display = []
                          processed_sources = set()
                          for meta in message["metadata"]:
                               if not meta: continue
                               source = meta.get('source', 'Unknown')
                               page_num = meta.get('page_number')
                               slide_num = meta.get('slide_number')
                               crawled = meta.get('crawled_url')
                               content_type = meta.get('content_type','unknown')
                               source_key = f"{source}"
                               if page_num: source_key += f"_p{page_num}"
                               elif slide_num: source_key += f"_s{slide_num}"
                               elif crawled: source_key += f"_u{hash(crawled)}"
                               if source_key not in processed_sources:
                                    display_str = f"- **{source}**"
                                    if page_num: display_str += f" (Page {page_num})"
                                    elif slide_num: display_str += f" (Slide {slide_num})"
                                    elif crawled: display_str += f" (From URL)"
                                    elif content_type == 'data_summary': display_str += " (Data Summary)"
                                    sources_display.append(display_str)
                                    processed_sources.add(source_key)
                          st.markdown("\n".join(sorted(sources_display)))

                # Display follow-up buttons
                if message["role"] == "assistant" and message.get("follow_ups"):
                     st.markdown("**Suggested Questions:**")
                     cols = st.columns(len(message["follow_ups"]))
                     for j, q in enumerate(message["follow_ups"]):
                          if cols[j].button(q, key=f"followup_{current_chat_state['chat_id']}_{i}_{j}"):
                               st.session_state.rerun_query = q
                               st.rerun()

# --- Input Area (Chat Input + File Uploader + Status) ---
input_container = st.container()
with input_container:
    # Handle follow-up query state
    query_input_value = ""
    if st.session_state.rerun_query:
        query_input_value = st.session_state.rerun_query
        st.session_state.rerun_query = None

    # Chat input field
    chat_input_disabled = not st.session_state.api_key_configured
    chat_input_placeholder = "Ask a question..." if st.session_state.api_key_configured else "Please configure API Key in sidebar"
    prompt = st.chat_input(
        chat_input_placeholder,
        key="main_query_input", # Keep key consistent for input value persistence
        disabled=chat_input_disabled
    )

    # File uploader
    st.file_uploader(
            "Attach Files for **this chat** (Processing starts automatically)",
            accept_multiple_files=True, type=ALL_SUPPORTED_TYPES,
            key="doc_uploader_key", # Key is essential for state access
            on_change=handle_file_upload,
            label_visibility="collapsed", disabled=chat_input_disabled
        )

    # --- Processing Status Display Area (Below Uploader) ---
    status_placeholder = st.empty()
    # Read status from the *current* chat state, handle None case gracefully
    current_processing_status = "idle" # Default if no state
    if current_chat_state:
        current_processing_status = current_chat_state.get("processing_status", "idle")

    if current_processing_status == "processing":
        status_placeholder.info("⏳ Processing uploaded files for this chat... Please wait.")
    elif current_processing_status == "success":
        status_placeholder.success("✅ Files processed successfully for this chat!")
        # Optional: Reset status after showing success? Requires care with reruns.
        # if current_chat_state: current_chat_state["processing_status"] = "idle"
    elif current_processing_status == "error":
        status_placeholder.error("⚠️ File processing failed for this chat.")


    # --- Query Handling Logic ---
    actual_prompt = prompt or query_input_value

    if actual_prompt and current_chat_state: # Ensure we have a prompt and valid chat state
        current_chat_id_for_query = current_chat_state["chat_id"]
        # Get the currently selected text model from session state
        selected_model = st.session_state.selected_text_model

        # Add user message to current chat's history
        current_chat_state["messages"].append({"role": "user", "content": actual_prompt})

        # --- Determine how to respond ---
        assistant_response_content = None
        follow_ups = []
        used_metadata = []
        prompt_lower = actual_prompt.lower().strip()
        greetings = ["hi", "hello", "hey", "yo", "greetings", "good morning", "good afternoon", "good evening"]
        help_keywords = ["help", "what can you do", "capabilities", "features", "functions", "what are your functions"]

        # 1. Handle conversational prompts FIRST
        is_greeting = any(greet == prompt_lower for greet in greetings)
        is_help_request = any(keyword in prompt_lower for keyword in help_keywords)

        if is_greeting:
            assistant_response_content = random.choice(["Hello!", "Hi there!", "Hey!"]) + " How can I help you with this chat?"
        elif is_help_request:
            assistant_response_content = f"""For this chat session, I can:
1.  **Process Files:** Upload documents using the attach button below. They will be associated only with this chat using ChromaDB.
2.  **Answer Questions:** Ask me questions based on the documents uploaded *in this chat*. I use Google's embedding model and the selected text model (`{selected_model}`) for this.
3.  **Basic Data Queries:** Ask simple questions about data files uploaded *in this chat*.

You can start a 'New Chat' from the sidebar to work with a different set of documents."""

        # 2. If not conversational, check state before proceeding
        elif current_processing_status == "processing":
            assistant_response_content = "Hold on! Files are still processing for this chat. Please wait."
        elif not st.session_state.api_key_configured:
             assistant_response_content = "I need a valid Google AI API Key configured in the sidebar."
        else:
            # Try to get the collection for the current chat
            current_collection = get_current_collection()
            if current_collection is None:
                 # Check if any files *were* processed successfully in this chat state
                 files_were_processed = current_chat_state.get("processed_files") and any(s == 'success' for s in current_chat_state["processed_files"].values())
                 if not files_were_processed:
                      assistant_response_content = f"It looks like you're asking about specific information, but no documents have been successfully processed in this chat ('{current_chat_id_for_query[-4:]}') yet. Please upload the relevant document(s)."
                 else:
                      assistant_response_content = f"I seem to be having trouble accessing the documents for this chat ('{current_chat_id_for_query[-4:]}'). Please try uploading them again or starting a new chat."
                      log_message(f"[{current_chat_id_for_query}] Collection object is None, but files were previously processed. Potential issue.", "error")

            else:
                # Collection exists - Proceed with document Q&A
                with st.spinner("Searching documents and thinking..."):
                    context_docs, context_metadatas = get_relevant_context(
                        actual_prompt, current_collection
                    )

                    # Pass the selected model name to QA functions
                    if is_data_query(actual_prompt):
                        log_message(f"[{current_chat_id_for_query}] Query classified as data-related.", "info")
                        # Pass selected model here
                        # main_app.py (around line 579 - corrected call)
                        data_answer_content, data_metadata = answer_data_query(
                            actual_prompt,
                            selected_model # Pass model name
                        )
                        assistant_response_content = data_answer_content # Assign the string part to the response content
                        used_metadata = data_metadata # Assign the returned metadata
                        if data_answer_content:
                            assistant_response_content = data_answer_content
                            used_metadata = context_metadatas
                        else:
                            log_message(f"[{current_chat_id_for_query}] Data query failed, falling back to general.", "info")
                            assistant_response_content, used_metadata = generate_answer_from_context(actual_prompt, context_docs, context_metadatas, selected_model)
                    else:
                        log_message(f"[{current_chat_id_for_query}] Query classified as general text-based.", "info")
                         # Pass selected model here
                        assistant_response_content, used_metadata = generate_answer_from_context(actual_prompt, context_docs, context_metadatas, selected_model)

                    # Generate follow-ups
                    if assistant_response_content and isinstance(assistant_response_content, str) and \
                       "cannot answer" not in assistant_response_content.lower() and \
                       "error" not in assistant_response_content.lower() and \
                       "blocked" not in assistant_response_content.lower():
                        # Pass selected model here
                        follow_ups = generate_followup_questions(actual_prompt, assistant_response_content, selected_model)
                    elif not assistant_response_content:
                         assistant_response_content = "Sorry, I encountered an issue generating a response."
                         used_metadata = []

        # Add assistant response to the *current* chat's history
        if assistant_response_content is not None:
            current_chat_state["messages"].append({
                "role": "assistant",
                "content": assistant_response_content,
                "follow_ups": follow_ups,
                "metadata": used_metadata
            })

        # Rerun to update the display
        st.rerun()


# --- Footer / Disclaimer ---
st.divider()
st.caption("AI responses based on documents in the current chat. Verify critical info. Use 'New Chat' for separate contexts.")

# --- Log Display Update (End of Script) ---
# Update sidebar log display one last time in case state changed during the run
if 'log_placeholder' in locals():
    log_container = st.container(height=250)
    log_placeholder = log_container.empty()
    log_placeholder.text_area("Logs", "\n".join(st.session_state.processing_log[::-1]), height=250, key="log_display_sidebar_final", disabled=True)

