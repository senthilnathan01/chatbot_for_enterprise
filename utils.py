# utils.py
"""General utility functions."""

import streamlit as st
import time
import random
import string
import re
import uuid # Import the uuid module

def log_message(message, level="info"):
    """Adds message to the session's processing log."""
    if "processing_log" not in st.session_state:
        st.session_state.processing_log = []
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}][{level.upper()}] {message}"
    st.session_state.processing_log.append(log_entry)

def generate_unique_id(): # Removed prefix argument as UUID is unique
    """Generates a unique ID using UUID4."""
    # Returns a unique string like 'f47ac10b-58cc-4372-a567-0e02b2c3d479'
    return str(uuid.uuid4())

def find_urls(text):
    """Finds potential URLs in text using regex."""
    if not isinstance(text, str):
        return []
    try:
        url_pattern = re.compile(r'https?://[^\s<>"\']+|(?:www|ftp)\.[^\s<>"\']+')
        return url_pattern.findall(text)
    except Exception as e:
        log_message(f"Error finding URLs: {e}", "warning")
        return []

def chunk_text(text, chunk_size, chunk_overlap):
    """Splits text into overlapping chunks."""
    if not isinstance(text, str) or not text: return []
    chunks = []
    start_index = 0
    text_length = len(text)
    while start_index < text_length:
        end_index = start_index + chunk_size
        chunks.append(text[start_index:end_index])
        start_index += chunk_size - chunk_overlap
        if start_index >= text_length: break
        if start_index < 0:
            log_message("Chunk overlap is too large, adjusting.", "warning")
            start_index = end_index
    return chunks

