# --- START OF FILE qa_engine.py ---
"""
Handles the core question answering logic, including data queries and follow-ups.
V3.5 Changes:
- Simplified RAG prompt in generate_answer_from_context, removing references
  to embedded image descriptions, reflecting changes in file_parsers.
"""

import streamlit as st
import google.generativeai as genai
import pandas as pd
from typing import Optional, Tuple, List, Any
import json
import re
import io
from utils import log_message
from VannaDataAnalyzer_Gemini import VannaDataAnalyzer_Gemini

# --- Helper to get chat state ---
# ... (keep as is) ...
def get_current_chat_state():
    """Safely retrieves the state dictionary for the currently active chat."""
    if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.get("chats", {}):
        log_message("Helper get_current_chat_state: Cannot determine current chat state.", "error")
        return None
    return st.session_state.chats.get(st.session_state.current_chat_id)

# --- is_data_query ---
# ... (keep as is) ...
def is_data_query(query):
    """Basic keyword-based check if a query seems related to structured data."""
    data_keywords = [
        "total", "average", "mean", "sum", "count", "list", "show me data",
        "maximum", "minimum", "value of", "in table", "from csv", "in excel",
        "json record", "how many rows", "column names", "what are the columns",
        "calculate", "statistics for", "summarize the data", "plot", "chart", "graph",
        "data for", "records where", "entries matching", "find rows", "summarise",
        "what is the", "give me the", "tell me the" # More question starters
    ]
    query_lower = query.lower()
    if re.match(r'^(what|how|list|show|give|tell|calculate|find|get)\b', query_lower):
         if any(keyword in query_lower for keyword in data_keywords):
              return True
    if any(keyword in query_lower for keyword in ["rows", "columns", "records", "entries"]):
        return True
    if any(agg in query_lower for agg in ['average', 'mean', 'total', 'sum', 'count', 'maximum', 'minimum']):
        return True
    return False


# --- answer_data_query ---
# ... (keep as is) ...
def answer_data_query(query: str, text_model_name: str) -> Tuple[str, list, Optional[pd.DataFrame]]:
    """
    Answers a query using the VannaDataAnalyzer_Gemini against the chat's specific database.
    Formats scalar results directly, provides limited table preview, and returns full DataFrame.

    Args:
        query: The user's natural language question about the data.
        text_model_name: The name of the Gemini model to use for NL-to-SQL.

    Returns:
        A tuple containing:
        - answer_string (str): Formatted text answer (scalar, preview, or error).
        - metadata_list (list): Metadata indicating the database source.
        - results_df (Optional[pd.DataFrame]): The full DataFrame result, or None if error/no results.
    """
    log_message(f"Attempting to answer data query using Vanna Engine: '{query}'", "info")
    analyzer = None
    final_answer_string = f"Okay, analyzing the data based on your query: '{query}'...\n\n"
    metadata = []
    results_df_full: Optional[pd.DataFrame] = None # Initialize df to None

    try:
        # 1. Get Current Chat State and Necessary Info
        current_chat_state = get_current_chat_state()
        if not current_chat_state:
             log_message("Data Query Error: Could not get current chat state.", "error")
             return "Error: Could not access the current chat session.", [], None

        chat_db_path = current_chat_state.get("chat_db_path")
        api_key = st.session_state.get("google_api_key")
        imported_tables = current_chat_state.get("imported_tables", [])

        if not chat_db_path:
             log_message("Data Query Info: No database found for this chat session.", "info")
             return "It looks like no data files (CSV, Excel, JSON) have been successfully imported in this chat session yet. Please upload a data file first.", [], None
        if not api_key:
             log_message("Data Query Error: API Key not found.", "error")
             return "Error: API Key is not configured, cannot analyze data.", [], None
        if not imported_tables:
            log_message("Data Query Info: Database exists but no imported tables recorded.", "warning")
            return "A database exists for this chat, but I don't have a record of which tables were imported. Please try re-uploading the data file.", [], None

        # 2. Initialize VannaDataAnalyzer
        log_message(f"Initializing VannaDataAnalyzer for DB: {chat_db_path}", "debug")
        analyzer = VannaDataAnalyzer_Gemini(
            api_key=api_key, model=text_model_name, db_path=chat_db_path, verbose=True
        )

        # 3. Ask Vanna
        vanna_result = analyzer.ask(question=query, explain=True) # Still get explanation

        # 4. Process and Format the Result
        if vanna_result.get("error"):
            log_message(f"Vanna Error: {vanna_result['error']}", "error")
            final_answer_string += f"Sorry, I encountered an error trying to analyze the data: {vanna_result['error']}"
            if vanna_result.get("sql_query"):
                final_answer_string += f"\n\n(Attempted SQL:\n```sql\n{vanna_result['sql_query']}\n```)"

        else:
            sql_query = vanna_result.get("sql_query", "N/A")
            # Store the full DataFrame result
            results_df_full = vanna_result.get("results")
            explanation = vanna_result.get("explanation", "No explanation generated.")

            # --- Result Formatting for Text Response ---
            final_answer_string += "**Result:**\n"
            if isinstance(results_df_full, pd.DataFrame):
                if results_df_full.empty:
                    if "status" in results_df_full.columns and "Execution successful" in results_df_full["status"].values:
                         rows_aff = results_df_full["rows_affected"].iloc[0]
                         final_answer_string += f"(Query executed successfully. Rows affected: {rows_aff})\n"
                    else:
                         final_answer_string += "(Query returned no results)\n"
                # Check for SCALAR result (1 row, 1 column)
                elif results_df_full.shape == (1, 1):
                    scalar_value = results_df_full.iloc[0, 0]
                    if pd.isna(scalar_value): formatted_value = "(No value)"
                    elif isinstance(scalar_value, float): formatted_value = f"{scalar_value:,.2f}".rstrip('0').rstrip('.') if '.' in f"{scalar_value:,.2f}" else f"{scalar_value:,.0f}"
                    elif isinstance(scalar_value, int): formatted_value = f"{scalar_value:,}"
                    else: formatted_value = str(scalar_value)
                    final_answer_string += f"**{formatted_value}**\n"
                    log_message(f"Displayed scalar result: {formatted_value}", "debug")
                # Format as limited table preview for non-scalar results
                else:
                    try:
                        max_rows_preview = 5
                        max_cols_preview = 5
                        df_preview = results_df_full.head(max_rows_preview)
                        cols_omitted_preview = False
                        rows_omitted_preview = len(results_df_full) > max_rows_preview

                        if len(df_preview.columns) > max_cols_preview:
                            df_preview = df_preview.iloc[:, :max_cols_preview]
                            cols_omitted_preview = True

                        final_answer_string += "(Showing preview)\n"
                        final_answer_string += df_preview.to_markdown(index=False) + "\n"

                        omitted_parts = []
                        if cols_omitted_preview: omitted_parts.append(f"first {max_cols_preview} cols")
                        if rows_omitted_preview: omitted_parts.append(f"first {max_rows_preview} rows")
                        if omitted_parts: final_answer_string += f"\n*({', '.join(omitted_parts)} of {len(results_df_full)} total rows shown)*"

                        final_answer_string += "\n\n*(Use the button below to view the full data table.)*"

                    except Exception as fmt_err:
                         log_message(f"Error formatting DataFrame preview to markdown: {fmt_err}", "warning")
                         final_answer_string += "(Could not display preview. Full table available via button below.)\n"
            else:
                final_answer_string += "(No DataFrame results returned)\n"

            # Add metadata
            metadata.append({
                "source": f"database_{current_chat_state['chat_id']}",
                "database": chat_db_path,
                "tables": ", ".join(imported_tables),
                "content_type": "database_query_result"
            })

            # Store SQL and Explanation for expander in main_app
            # We don't add them to final_answer_string here anymore
            # But they are available in vanna_result if needed elsewhere

    except Exception as e:
        st.exception(e)
        log_message(f"Critical error in answer_data_query: {e}", "error")
        final_answer_string += f"\n\nAn unexpected error occurred while trying to answer the data query: {e}"
        results_df_full = None # Ensure DF is None on error
    finally:
        if analyzer:
            analyzer.close()

    return final_answer_string.strip(), metadata, results_df_full


# --- generate_answer_from_context (RAG - SIMPLIFIED PROMPT) ---
def generate_answer_from_context(query, context_docs, context_metadatas, text_model_name):
    """Generates answer using the specified LLM based purely on retrieved text context (RAG)."""
    if not context_docs:
        log_message("No relevant context found in documents for the query.", "warning")
        data_markers = [m for m in context_metadatas if m and m.get('content_type', '').startswith('data_import')]
        if data_markers:
            imported_files = list(set(m.get('source') for m in data_markers if m.get('source')))
            return f"I couldn't find relevant information in the text documents for '{query}'. However, I see that data file(s) ({', '.join(imported_files)}) were imported. Try asking a specific question about the data content (e.g., 'how many rows in {imported_files[0]}?', 'what is the total sales?').", []
        else:
            return "I cannot answer this question based on the provided documents.", []

    context_items = []
    source_files = set()
    meta_list_for_response = []

    for doc, meta in zip(context_docs, context_metadatas):
         if not meta or not doc: continue
         source = meta.get('source', 'Unknown')
         source_files.add(source)
         citation = f"Source: {source}"

         if meta.get('page_number'): citation += f" (Page {meta.get('page_number')})"
         elif meta.get('slide_number'): citation += f" (Slide {meta.get('slide_number')})"
         elif meta.get('crawled_url'): citation += f" (From URL: {meta.get('crawled_url')})"

         content_type = meta.get("content_type", "text")
         content_prefix = "Content"
         actual_content = doc

         # Handle data markers explicitly
         if content_type == 'data_import_success':
              tables_str = meta.get('tables', 'unknown tables')
              citation += f" (Data File - Imported [{tables_str}], query directly)"
              actual_content = f"[Marker for imported data file '{source}'. Query its content directly.]"
              content_prefix = "Marker"
         elif content_type == 'data_import_failed':
               citation += " (Data File - Import Failed)"
               actual_content = f"[Marker for failed data import '{source}'.]"
               content_prefix = "Marker"
         # Note: 'embedded_image' and 'image_analysis' types should no longer appear for PDF/DOCX

         context_items.append(f"{citation}\n{content_prefix}:\n{actual_content}")
         meta_list_for_response.append(meta)

    context_str = "\n\n---\n\n".join(context_items)

    # --- *** SIMPLIFIED RAG PROMPT *** ---
    prompt = f"""You are a helpful assistant answering questions based ONLY on the text content provided in the context below.

    **Provided Context:**
    ```
    {context_str}
    ```

    **Instructions:**
    1.  Examine the user's question.
    2.  Carefully read the **Provided Context**. Context items contain citations and content which might be text excerpts or specific markers for data files.
    3.  **Data File Markers:** If a context item's 'Content:' or 'Marker:' section explicitly starts with `[Marker for imported data file...` or `[Marker for failed data import...]`, that item refers to a data file that cannot be directly read here. If the user's question seems related *only* to such a file marker, explain that the data must be queried separately using specific questions about its contents (e.g., "how many rows?", "what is the total revenue?"). Do *not* try to answer using the marker text itself.
    4.  **Answering from Text:** If the user's question can be answered using information from items that are *not* data file markers, synthesize a comprehensive answer based *only* on the text provided in these items.
    5.  **Summarization:** If the user asks for a general summary (e.g., "what is this document about?", "summarize this"), generate a concise overview based on the main topics found in the provided text excerpts.
    6.  **Citations:** When using information from the text, mention the source file and specific location (e.g., Page number, Slide number) provided in the citation line (e.g., "Source: my_doc.pdf (Page 3)") for the relevant context item(s).
    7.  **If No Answer:** If the provided text context (excluding data file markers) does not contain the information needed to answer the question, state that clearly (e.g., "Based on the provided documents, I cannot find information about X.").

    **User Question:**
    ```
    {query}
    ```

    **Answer:**
    """
    # --- *** END SIMPLIFIED RAG PROMPT *** ---

    try:
        model = genai.GenerativeModel(text_model_name)
        generation_config = genai.types.GenerationConfig(temperature=0.3)
        response = model.generate_content(prompt, generation_config=generation_config)

        if hasattr(response, 'text'):
            answer_text = response.text
            # Add citation hint only if non-data sources were potentially used
            unique_sources = set(m.get("source", "Unknown") for m in meta_list_for_response if m and not m.get("content_type", "").startswith("data_import"))
            contains_citation = any(f in answer_text for f in unique_sources)
            if not contains_citation and len(unique_sources) < 4 and unique_sources:
                 answer_text += f"\n\n(Sources possibly consulted: {', '.join(sorted(list(unique_sources)))})"
            return answer_text, meta_list_for_response
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             log_message(f"General answer generation blocked: {response.prompt_feedback.block_reason}", "warning")
             return f"Answer blocked due to safety settings: {response.prompt_feedback.block_reason}", meta_list_for_response
        else:
             log_message("LLM did not generate a general answer.", "warning")
             return "Sorry, I couldn't generate an answer based on the context.", meta_list_for_response
    except Exception as e:
        st.exception(e)
        log_message(f"Error generating general answer with LLM {text_model_name}: {e}", "error")
        return f"An error occurred while generating the answer: {e}", meta_list_for_response


# --- generate_followup_questions ---
# ... (keep as is) ...
def generate_followup_questions(query, answer, text_model_name):
    """Generates relevant follow-up questions using the specified LLM."""
    if "[Data File Imported:" in answer or "Generated SQL Query:" in answer or "Sorry, I encountered an error" in answer or "Result:" in answer:
         return []

    prompt = f"""Based on the user's question and the provided answer, suggest exactly 3 concise and relevant follow-up questions that naturally extend the conversation or explore related aspects mentioned in the answer.

    Original Question: {query}
    Provided Answer: {answer}

    Rules:
    - Focus on questions related to the *content* of the answer.
    - Ensure questions are distinct from the original question.
    - Do not suggest questions like "Can you tell me more?". Be specific.
    - Output *only* a Python list of strings, like ["Question 1?", "Question 2?", "Question 3?"].

    Suggested Follow-up Questions (Python list format ONLY):
    """
    try:
        model = genai.GenerativeModel(text_model_name)
        generation_config = genai.types.GenerationConfig(temperature=0.5)
        response = model.generate_content(prompt, generation_config=generation_config)

        if hasattr(response, 'text'):
            response_text = response.text.strip()
            log_message(f"Raw follow-up suggestions: {response_text}", "debug")
            match = re.search(r'\[\s*".*?"\s*(?:,\s*".*?"\s*)*\]', response_text, re.DOTALL)
            if match:
                 list_str = match.group()
                 try:
                      import ast
                      followups = ast.literal_eval(list_str)
                      if isinstance(followups, list) and all(isinstance(q, str) for q in followups):
                           log_message(f"Parsed follow-ups via ast: {followups}", "debug")
                           return followups[:3]
                 except Exception as eval_err:
                      log_message(f"ast.literal_eval failed for follow-up list '{list_str}': {eval_err}", "warning")

            lines = [line.strip(' -*\'" ') for line in response_text.split('\n')]
            questions = [line for line in lines if line.endswith('?') and len(line) > 10]
            if questions:
                 log_message(f"Fallback parsed follow-ups (lines): {questions[:3]}", "debug")
                 return questions[:3]

            log_message("Could not reliably parse follow-up suggestions into a list.", "warning")
            return []
        else:
             log_message("LLM response for follow-ups has no text part.", "warning")
             return []
    except Exception as e:
        log_message(f"Error generating follow-up questions with LLM {text_model_name}: {e}", "error")
        return []

# --- END OF FILE qa_engine.py ---
