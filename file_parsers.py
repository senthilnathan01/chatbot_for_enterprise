# --- START OF FILE file_parsers.py ---
"""Functions to parse different file types.
Each parser should return a list of tuples:
[(text_segment_1, metadata_1), (text_segment_2, metadata_2), ...]
where metadata includes at least 'source' and potentially 'page_number' etc.
Also returns the list of found URLs.

Data files (CSV/XLSX/JSON) are handled differently: they are imported into
a chat-specific database via DataImporter_Gemini, and a marker segment
is returned for ChromaDB indexing.

MODIFIED: Removed embedded image extraction/analysis from PDF and DOCX parsers
to focus solely on text content (including page-level OCR).
"""

import streamlit as st
import io
import fitz # PyMuPDF
import docx
from pptx import Presentation
import pandas as pd
import json
import os
from pathlib import Path

# Import specific analysis tool classes
from DataImporter_Gemini import DataImporter_Gemini

from utils import log_message, find_urls
from image_processor import get_gemini_vision_description_ocr # Still needed for page OCR and image files
from config import CHAT_DB_DIRECTORY, SUPPORTED_DATA_TYPES, SUPPORTED_TEXT_TYPES

# --- PDF Parser (REMOVED EMBEDDED IMAGE ANALYSIS) ---
def parse_pdf(file_content, filename):
    parsed_segments = [] # List of (text_segment, metadata)
    urls = set()
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        log_message(f"Processing PDF '{filename}': {len(doc)} pages (Text/OCR Only).", "info") # Updated log
        for page_num, page in enumerate(doc):
            page_num_actual = page_num + 1 # 1-based page number
            page_text = ""
            page_meta = {"source": filename, "page_number": page_num_actual}
            ocr_text_content = "" # Store OCR result separately

            try:
                # Extract standard text
                page_text = page.get_text("text", sort=True)
                if page_text:
                    urls.update(find_urls(page_text))
                    page_meta_text = page_meta.copy()
                    page_meta_text["content_type"] = "text" # Add content type
                    parsed_segments.append((page_text, page_meta_text))
                else:
                    log_message(f"No extractable text found on page {page_num_actual} of {filename}. Checking for images/OCR.", "debug")

                # --- OCR Fallback for image-based pages ---
                # Check if OCR is needed (little or no text extracted)
                ocr_needed = not page_text or len(page_text.strip()) < 50 # Adjust heuristic if needed
                if ocr_needed:
                    log_message(f"Attempting OCR for page {page_num_actual} in '{filename}'.", "info")
                    try:
                        # Increase DPI slightly for potentially better OCR
                        pix = page.get_pixmap(dpi=200)
                        img_bytes = pix.tobytes("png")
                        ocr_text_content = get_gemini_vision_description_ocr(img_bytes, f"{filename}_page{page_num_actual}_ocr")
                        if ocr_text_content and "No description or text extracted" not in ocr_text_content and len(ocr_text_content) > 20: # Check length
                             ocr_meta = page_meta.copy()
                             ocr_meta["content_type"] = "ocr_page"
                             # Add the OCR text as a segment
                             parsed_segments.append((f"[OCR Result for Page {page_num_actual}]\n{ocr_text_content}", ocr_meta))
                             urls.update(find_urls(ocr_text_content)) # Extract URLs from OCR text too
                        else:
                             log_message(f"OCR found no significant text on page {page_num_actual}.", "debug")
                    except Exception as ocr_err:
                        log_message(f"Error during OCR fallback for {filename} page {page_num_actual}: {ocr_err}", "warning")
                # --- End OCR Fallback ---

                # --- REMOVED EMBEDDED IMAGE EXTRACTION BLOCK ---
                # The following block was removed:
                # try:
                #     image_list = page.get_images(full=True)
                #     for img_index, img_info in enumerate(image_list):
                #         # ... code to extract image bytes ...
                #         # ... code to call get_gemini_vision_description_ocr ...
                #         # ... code to append image analysis segment ...
                # except Exception as img_err:
                #     log_message(...)
                # --- END REMOVED BLOCK ---

            except Exception as page_err:
                 log_message(f"Error processing page {page_num_actual} of {filename}: {page_err}", "warning")

        log_message(f"Finished processing PDF (Text/OCR Only) '{filename}'. Found URLs: {len(urls)}")
    except Exception as e:
        log_message(f"Critical error processing PDF '{filename}': {e}", "error")
        return [], []
    return parsed_segments, list(urls)


# --- DOCX Parser (REMOVED IMAGE ANALYSIS) ---
def parse_docx(file_content, filename):
    parsed_segments = []
    urls = set()
    base_meta = {"source": filename, "content_type": "text"}
    # img_meta = {"source": filename, "content_type": "embedded_image"} # No longer needed

    try:
        doc_stream = io.BytesIO(file_content)
        doc = docx.Document(doc_stream)
        log_message(f"Processing DOCX '{filename}' (Text Only).", "info") # Updated log

        # Extract text from paragraphs
        para_texts = []
        for para in doc.paragraphs:
            para_text = para.text
            if para_text.strip(): # Only add non-empty paragraphs
                 para_texts.append(para_text)
                 urls.update(find_urls(para_text))
        if para_texts: # Check if any text was found
             text_content = '\n\n'.join(para_texts) # Use double newline for better separation
             parsed_segments.append((text_content, base_meta.copy()))


        # Extract text from tables (basic)
        table_texts = []
        for i, table in enumerate(doc.tables):
             table_str = f"\n[Table {i+1} Content:]\n"
             try:
                  rows_data = []
                  for row in table.rows:
                       row_text = [cell.text.strip() for cell in row.cells]
                       # Filter out empty rows potentially
                       if any(row_text):
                           rows_data.append("| " + " | ".join(row_text) + " |")
                  if rows_data:
                      table_str += "\n".join(rows_data) + "\n"
                      table_texts.append(table_str)
                      urls.update(find_urls(table_str)) # Extract URLs from tables
             except Exception as table_err:
                  log_message(f"Error reading table {i+1} in {filename}: {table_err}", "warning")
        if table_texts:
            table_meta = base_meta.copy()
            table_meta["content_type"] = "table_text" # Keep specific type for tables
            parsed_segments.append(("\n".join(table_texts), table_meta))


        # Handle hyperlinks explicitly
        try:
            # Accessing relationships might require careful handling depending on docx structure
            if hasattr(doc.part, 'rels'):
                 for rel in doc.part.rels.values():
                     if rel.reltype == docx.opc.constants.RELATIONSHIP_TYPE.HYPERLINK and rel.is_external:
                         urls.add(rel.target_ref)
        except Exception as rel_err:
             log_message(f"Could not process relationships for hyperlinks in {filename}: {rel_err}", "debug")


        # --- REMOVED IMAGE EXTRACTION BLOCK ---
        # The following block was removed:
        # image_parts = {}
        # # ... code to populate image_parts ...
        # img_count = 0
        # processed_rel_ids = set()
        # for shape in all_shapes: # Primarily doc.inline_shapes
        #      try:
        #         # ... code to check if shape is image ...
        #         # ... code to get embed_id ...
        #         # ... code to call get_gemini_vision_description_ocr ...
        #         # ... code to append image analysis segment ...
        #      except Exception as img_ex:
        #          log_message(...)
        # --- END REMOVED BLOCK ---

        log_message(f"Finished processing DOCX (Text Only) '{filename}'. Found URLs: {len(urls)}")
    except Exception as e:
        log_message(f"Critical error processing DOCX '{filename}': {e}", "error")
        return [], []
    return parsed_segments, list(urls)

# --- PPTX Parser ---
# NOTE: Keep PPTX image analysis for now, as slides are often visual.
# Remove if you want PPTX to be text-only as well.
def parse_pptx(file_content, filename):
    parsed_segments = []
    urls = set()
    try:
        ppt_stream = io.BytesIO(file_content)
        prs = Presentation(ppt_stream)
        log_message(f"Processing PPTX '{filename}': {len(prs.slides)} slides.", "info")

        img_count = 0 # Keep track of images across slides for unique naming
        for i, slide in enumerate(prs.slides):
            slide_num = i + 1
            slide_text_content = ""
            slide_meta = {"source": filename, "slide_number": slide_num, "content_type": "slide_text"}
            slide_urls = set()

            try:
                notes_slide = slide.notes_slide
                if notes_slide and notes_slide.notes_text_frame:
                     notes_text = notes_slide.notes_text_frame.text
                     if notes_text:
                          slide_text_content += f"\n--- Speaker Notes (Slide {slide_num}) ---\n{notes_text}\n"
                          slide_urls.update(find_urls(notes_text))
            except Exception as notes_err:
                 log_message(f"Could not extract speaker notes for slide {slide_num}: {notes_err}", "warning")

            for shape in slide.shapes:
                shape_text = ""
                # Extract text from shapes
                if shape.has_text_frame:
                    try: shape_text = shape.text_frame.text
                    except AttributeError: pass
                    except Exception as text_err:
                        log_message(f"Error getting text from shape {shape.shape_id} on slide {slide_num}: {text_err}", "debug")
                    if shape_text:
                         slide_text_content += shape_text + "\n"
                         slide_urls.update(find_urls(shape_text))

                # Extract text from tables
                if shape.has_table:
                    table_text = f"\n[Table on Slide {slide_num}:]\n"
                    try:
                        for row in shape.table.rows:
                            row_data = [cell.text_frame.text for cell in row.cells]
                            table_text += "| " + " | ".join(row_data) + " |\n"
                        slide_text_content += table_text
                        slide_urls.update(find_urls(table_text))
                    except Exception as table_err:
                         log_message(f"Error reading table on slide {slide_num}: {table_err}", "warning")

                # Handle hyperlinks
                try:
                    if shape.click_action and shape.click_action.hyperlink and shape.click_action.hyperlink.address:
                        slide_urls.add(shape.click_action.hyperlink.address)
                    if hasattr(shape, 'text_frame'):
                        for para in shape.text_frame.paragraphs:
                            for run in para.runs:
                                if run.hyperlink and run.hyperlink.address:
                                    slide_urls.add(run.hyperlink.address)
                except Exception as link_err:
                    log_message(f"Minor error reading hyperlink on slide {slide_num}: {link_err}", "debug")

                # Handle images (Keep this section for PPTX unless requested otherwise)
                if hasattr(shape, 'image') or shape.shape_type == 13: # Picture or has image fill
                    try:
                        image = shape.image
                        image_bytes = image.blob
                        img_meta_filename = f"{Path(filename).stem}_slide{slide_num}_img{img_count}_{shape.shape_id}"
                        img_desc_ocr = get_gemini_vision_description_ocr(image_bytes, img_meta_filename)
                        if img_desc_ocr and "No description or text extracted" not in img_desc_ocr:
                             img_meta = {"source": filename, "slide_number": slide_num, "content_type": "embedded_image", "image_ref": f"shape_{shape.shape_id}"}
                             parsed_segments.append((f"[Analysis of Image {img_count} on Slide {slide_num}]\n{img_desc_ocr}", img_meta))
                             img_count += 1
                    except Exception as img_err:
                         log_message(f"Could not process shape as image on slide {slide_num} (ID {shape.shape_id}): {img_err}", "warning")


            slide_text_content_stripped = slide_text_content.strip()
            if slide_text_content_stripped:
                 parsed_segments.append((slide_text_content_stripped, slide_meta))
            urls.update(slide_urls)

        log_message(f"Finished processing PPTX '{filename}'. Found URLs: {len(urls)}, Images Analyzed: {img_count}")
    except Exception as e:
        log_message(f"Critical error processing PPTX '{filename}': {e}", "error")
        return [], []
    return parsed_segments, list(urls)


# --- Data File Parser (Integrates DataImporter_Gemini) ---
# ... (keep as is) ...
def parse_data_file(file_content, filename):
    """
    Parses CSV, XLSX, or JSON by importing it into a chat-specific SQLite DB.
    Returns a single "marker" text segment indicating success/failure for ChromaDB.
    The actual data interaction happens via VannaDataAnalyzer_Gemini in qa_engine.
    """
    parsed_segments = []
    file_type = filename.split('.')[-1].lower()
    chat_id = st.session_state.get("current_chat_id")
    chat_state = st.session_state.chats.get(chat_id)

    if not chat_id or not chat_state:
        log_message(f"Error parsing data file '{filename}': Could not determine current chat state.", "error")
        base_meta = {"source": filename, "content_type": "data_import_failed", "reason": "Chat state unavailable"}
        marker_text = f"[Data File Import Failed: {filename} - Chat state unavailable]"
        parsed_segments.append((marker_text, base_meta))
        return parsed_segments, []

    chat_db_filename = f"chat_{chat_id}.db"
    chat_db_path = os.path.join(CHAT_DB_DIRECTORY, chat_db_filename)
    api_key = st.session_state.get("google_api_key")
    if not api_key:
         log_message(f"Warning during data file import '{filename}': API Key not configured. Analysis will fail later.", "warning")

    importer = None
    imported_tables = []
    import_success = False
    try:
        log_message(f"Processing Data file '{filename}' for chat '{chat_id}' into DB '{chat_db_path}'.", "info")
        importer = DataImporter_Gemini(db_path=chat_db_path, verbose=True)
        import_kwargs = {'if_exists': 'replace'}

        if file_type == "csv":
            table_name = importer._clean_column_name(Path(filename).stem)
            stream = io.BytesIO(file_content)
            try:
                imported_tables = importer.import_csv_from_stream(
                    stream=stream, table_name=table_name, encoding='utf-8', if_exists=import_kwargs['if_exists']
                 )
            except UnicodeDecodeError:
                log_message(f"UTF-8 decode failed for CSV '{filename}', trying latin1.", "warning")
                stream.seek(0)
                imported_tables = importer.import_csv_from_stream(
                    stream=stream, table_name=table_name, encoding='latin1', if_exists=import_kwargs['if_exists']
                )
            except Exception as csv_err:
                raise RuntimeError(f"CSV stream import failed: {csv_err}")

        elif file_type == "xlsx":
            excel_file_stream = io.BytesIO(file_content)
            try:
                 xls = pd.ExcelFile(excel_file_stream)
                 if not xls.sheet_names: raise RuntimeError("Excel file has no sheets.")
                 sheet_name_to_import = xls.sheet_names[0]
                 table_name = importer._clean_column_name(Path(filename).stem)
                 import_kwargs['sheet_name'] = sheet_name_to_import
                 import_kwargs['table_names'] = {sheet_name_to_import: table_name}
                 imported_tables = importer.import_excel(excel_file_stream, **import_kwargs)
            except Exception as excel_err:
                raise RuntimeError(f"Excel import failed: {excel_err}")

        elif file_type == "json":
             table_name = importer._clean_column_name(Path(filename).stem)
             import_kwargs['table_name'] = table_name
             try:
                 stream = io.BytesIO(file_content)
                 df = pd.read_json(stream)
                 importer._import_dataframe(df, table_name, if_exists=import_kwargs['if_exists'])
                 imported_tables = [table_name]
             except Exception as json_err:
                 raise RuntimeError(f"JSON import/parsing failed: {json_err}")

        if imported_tables:
             import_success = True
             chat_state["chat_db_path"] = chat_db_path
             if "imported_tables" not in chat_state: chat_state["imported_tables"] = []
             chat_state["imported_tables"] = list(set(chat_state["imported_tables"] + imported_tables))

             log_message(f"Successfully imported '{filename}' to table(s) {imported_tables} in DB '{chat_db_path}'.")
             tables_string = ", ".join(imported_tables)
             base_meta = {
                 "source": filename,
                 "content_type": "data_import_success",
                 "database": chat_db_path,
                 "tables": tables_string
             }
             marker_text = f"[Data File Imported: {filename} ({tables_string}) - Ready for querying in chat database.]"
             parsed_segments.append((marker_text, base_meta))
             chat_state["processed_files"][filename] = 'success'

        else:
             log_message(f"Data import process for '{filename}' completed but reported no imported tables (empty file?).", "warning")
             base_meta = {"source": filename, "content_type": "data_import_failed", "reason": "No tables reported by importer"}
             marker_text = f"[Data File Import Failed: {filename} - No tables were imported (file empty or error).]"
             parsed_segments.append((marker_text, base_meta))
             chat_state["processed_files"][filename] = 'failed'

    except Exception as e:
        st.exception(e)
        log_message(f"Error importing data file '{filename}' into database: {e}", "error")
        base_meta = {"source": filename, "content_type": "data_import_failed", "reason": str(e)}
        marker_text = f"[Data File Import Failed: {filename} - Error: {e}]"
        parsed_segments.append((marker_text, base_meta))
        if filename in chat_state.get("processed_files", {}):
             chat_state["processed_files"][filename] = 'failed'
        return parsed_segments, []
    finally:
        if importer:
            importer.close()

    return parsed_segments, []


# --- TXT Parser ---
# ... (keep as is) ...
def parse_txt(file_content, filename):
    text = ""
    urls = set()
    base_meta = {"source": filename, "content_type": "text"}
    parsed_segments = []
    try:
        log_message(f"Processing TXT file '{filename}'.", "info")
        try: text = file_content.decode("utf-8")
        except UnicodeDecodeError: text = file_content.decode("latin1", errors='ignore')

        if text:
             urls.update(find_urls(text))
             parsed_segments.append((text, base_meta))
             log_message(f"Finished processing TXT '{filename}'. Found URLs: {len(urls)}")
        else:
             log_message(f"TXT file '{filename}' is empty.", "warning")

    except Exception as e:
        log_message(f"Error processing TXT file '{filename}': {e}", "error")
        return [], []
    return parsed_segments, list(urls)

# --- Image File Parser ---
# This is for standalone image files (PNG, JPG) - KEEP this functionality.
def parse_image(file_content, filename):
    text = ""
    base_meta = {"source": filename, "content_type": "image_analysis"}
    parsed_segments = []
    try:
        log_message(f"Processing Image file '{filename}'.", "info")
        img_desc_ocr = get_gemini_vision_description_ocr(file_content, filename)
        if img_desc_ocr and "No description or text extracted" not in img_desc_ocr:
             text = f"[Analysis of Image File: {filename}]\n{img_desc_ocr}\n"
             parsed_segments.append((text, base_meta))
             log_message(f"Finished processing image file: {filename}")
        else:
             log_message(f"Image analysis did not return significant content for {filename}.", "warning")

    except Exception as e:
        log_message(f"Error processing image file '{filename}': {e}", "error")
        return [], []
    return parsed_segments, [] # No URLs expected
# --- END OF FILE file_parsers.py ---
