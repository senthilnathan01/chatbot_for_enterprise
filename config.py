# config.py
"""Stores configuration variables and constants."""
import os
# --- ChromaDB Settings ---
VECTOR_DB_PERSIST_PATH = "/mount/src/chatbot_for_enterprise/chroma_db_data" # Use an explicit path within your project/mount
# Ensure the directory exists and has permissions
if not os.path.exists(VECTOR_DB_PERSIST_PATH):
     try:
          os.makedirs(VECTOR_DB_PERSIST_PATH)
          print(f"Created ChromaDB persistence directory: {VECTOR_DB_PERSIST_PATH}")
     except OSError as e:
          print(f"ERROR: Could not create ChromaDB persistence directory {VECTOR_DB_PERSIST_PATH}: {e}")
          # Decide how to handle - maybe fall back to in-memory or raise error?
          VECTOR_DB_PERSIST_PATH = None # Fallback if creation fails

DEFAULT_COLLECTION_NAME_PREFIX = "qa_coll_"
# VECTOR_DB_PERSIST_PATH = None # Set to a directory path for persistence, None for in-memory
# --- Model Names ---
TEXT_MODEL_NAME = "gemini-2.0-flash" # Use the updated model name
VISION_MODEL_NAME = "gemini-2.0-flash" # Keep vision model name consistent
EMBEDDING_MODEL_NAME = "models/embedding-001" # Or a newer embedding model like "models/text-embedding-004"

# --- Processing Constants ---
CRAWLING_TIMEOUT = 10 # Seconds for web requests
MAX_CRAWL_TEXT_LENGTH = 5000 # Limit crawled text size per URL
CHUNK_SIZE = 5000 # Characters per chunk
CHUNK_OVERLAP = 1000 # Characters overlap between chunks
MAX_CONTEXT_RESULTS = 7 # Number of chunks to retrieve for context

# --- ChromaDB Settings ---
DEFAULT_COLLECTION_NAME_PREFIX = "qa_coll_"
VECTOR_DB_PERSIST_PATH = None # Set to a directory path for persistence, None for in-memory

# --- Data Analysis DB Settings --- # 
CHAT_DB_DIRECTORY = "chat_databases" # Directory to store individual chat SQLite files
# Ensure this directory exists or is created at startup
if not os.path.exists(CHAT_DB_DIRECTORY):
    os.makedirs(CHAT_DB_DIRECTORY)
    print(f"Created directory for chat databases: {CHAT_DB_DIRECTORY}")

# --- File Type Handling ---
SUPPORTED_TEXT_TYPES = ['pdf', 'docx', 'pptx', 'txt']
SUPPORTED_DATA_TYPES = ['csv', 'xlsx', 'json']
SUPPORTED_IMAGE_TYPES = ['png', 'jpg', 'jpeg']
ALL_SUPPORTED_TYPES = SUPPORTED_TEXT_TYPES + SUPPORTED_DATA_TYPES + SUPPORTED_IMAGE_TYPES
