# image_processor.py
"""Handles image processing and interaction with Vision Language Models."""

import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
from utils import log_message
from config import VISION_MODEL_NAME

# This function assumes genai is already configured in the main app scope

def get_gemini_vision_description_ocr(image_bytes, filename="image"):
    """
    Gets description and attempts OCR using Gemini Vision.
    Assumes genai API is configured.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Ensure model is initialized correctly (this might need API key if not globally configured)
        model = genai.GenerativeModel(VISION_MODEL_NAME)
        # Ask for both description and text extraction
        prompt = "Describe this image in detail and extract any text visible within it. Format the output clearly, perhaps using sections like 'Description:' and 'Extracted Text:'."
        response = model.generate_content([prompt, img])

        # Handle potential API response variations
        if hasattr(response, 'text'):
            return response.text
        elif response.parts:
             # If response has parts, concatenate them
             return "".join(part.text for part in response.parts if hasattr(part, 'text'))
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            log_message(f"Image processing for {filename} blocked: {block_reason}", "warning")
            return f"[Image: {filename} - Processing Blocked: {block_reason}]"
        else:
            log_message(f"Could not generate description/OCR for {filename}. Response format might be unexpected.", "warning")
            # Try to log the raw response for debugging if possible
            # log_message(f"Unexpected Vision Response: {response}", "debug")
            return f"[Image: {filename} - No description or text extracted]"

    except Exception as e:
        # Log specific GenAI errors if possible, e.g., API key issues, model access
        log_message(f"Error processing image {filename} with Gemini Vision: {e}", "error")
        # Optionally re-raise or handle specific exceptions if needed
        # st.exception(e) # Show detailed traceback in Streamlit console
        return f"[Image: {filename} - Error processing image: {e}]"
