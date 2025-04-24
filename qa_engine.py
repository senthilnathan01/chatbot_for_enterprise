# --- START OF FILE qa_engine.py ---
"""
Handles the core question answering logic, including data queries and follow-ups.
V3.1 Changes:
- Improved display of scalar results (e.g., single numbers like AVG, SUM, COUNT) in answer_data_query.
- Made explanation less prominent by default when a scalar result is shown.
"""

import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import re
import io
from utils import log_message
# Import Vanna Analyzer
from VannaDataAnalyzer_Gemini import VannaDataAnalyzer_Gemini

# --- Helper to get chat state ---
def get_current_chat_state():
    """Safely retrieves the state dictionary for the currently active chat."""
    if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.get("chats", {}):
        log_message("Helper get_current_chat_state: Cannot determine current chat state.", "error")
        return None
    return st.session_state.chats.get(st.session_state.current_chat_id)
# --- End Helper ---


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
    # Simple check for common question structures about data
    query_lower = query.lower()
    if re.match(r'^(what|how|list|show|give|tell|calculate|find|get)\b', query_lower):
         if any(keyword in query_lower for keyword in data_keywords):
              return True
    # Catch simple column/row requests
    if any(keyword in query_lower for keyword in ["rows", "columns", "records", "entries"]):
        return True
    # Catch direct table references if user knows them (less likely)
    # Be careful with this one, could overlap with RAG
    # if "table" in query_lower and any(kw in query_lower for kw in ['show', 'list', 'data']): return True

    # Check specifically for aggregate functions explicitly mentioned
    if any(agg in query_lower for agg in ['average', 'mean', 'total', 'sum', 'count', 'maximum', 'minimum']):
        return True

    return False


def answer_data_query(query: str, text_model_name: str) -> tuple[str, list]:
    """
    Answers a query using the VannaDataAnalyzer_Gemini against the chat's specific database.
    Formats scalar results (single numbers) directly.

    Args:
        query: The user's natural language question about the data.
        text_model_name: The name of the Gemini model to use for NL-to-SQL.

    Returns:
        A tuple containing:
        - answer_string (str): Formatted answer including SQL, results, explanation.
        - metadata_list (list): Metadata indicating the database source.
    """
    log_message(f"Attempting to answer data query using Vanna Engine: '{query}'", "info")
    analyzer = None
    # Start with a less verbose intro
    final_answer_string = f"Okay, analyzing the data based on your query: '{query}'...\n\n"
    metadata = []
    result_was_scalar = False # Flag to check if we handled a single value

    try:
        # 1. Get Current Chat State and Necessary Info
        current_chat_state = get_current_chat_state()
        if not current_chat_state:
             log_message("Data Query Error: Could not get current chat state.", "error")
             return "Error: Could not access the current chat session.", []

        chat_db_path = current_chat_state.get("chat_db_path")
        api_key = st.session_state.get("google_api_key")
        imported_tables = current_chat_state.get("imported_tables", [])

        if not chat_db_path:
             log_message("Data Query Info: No database found for this chat session.", "info")
             return "It looks like no data files (CSV, Excel, JSON) have been successfully imported in this chat session yet. Please upload a data file first.", []
        if not api_key:
             log_message("Data Query Error: API Key not found.", "error")
             return "Error: API Key is not configured, cannot analyze data.", []
        if not imported_tables:
            log_message("Data Query Info: Database exists but no imported tables recorded.", "warning")
            return "A database exists for this chat, but I don't have a record of which tables were imported. Please try re-uploading the data file.", []

        # 2. Initialize VannaDataAnalyzer
        log_message(f"Initializing VannaDataAnalyzer for DB: {chat_db_path}", "debug")
        analyzer = VannaDataAnalyzer_Gemini(
            api_key=api_key, model=text_model_name, db_path=chat_db_path, verbose=True
        )

        # 3. Ask Vanna
        vanna_result = analyzer.ask(question=query, explain=True) # Get explanation even if not shown by default

        # 4. Process and Format the Result
        if vanna_result.get("error"):
            log_message(f"Vanna Error: {vanna_result['error']}", "error")
            final_answer_string += f"Sorry, I encountered an error trying to analyze the data: {vanna_result['error']}"
            if vanna_result.get("sql_query"):
                final_answer_string += f"\n\n(Attempted SQL:\n```sql\n{vanna_result['sql_query']}\n```)" # Make SQL less prominent on error

        else:
            sql_query = vanna_result.get("sql_query", "N/A")
            results_df = vanna_result.get("results")
            explanation = vanna_result.get("explanation", "No explanation generated.")

            # --- Result Formatting ---
            final_answer_string += "**Result:**\n" # Changed header
            if isinstance(results_df, pd.DataFrame):
                if results_df.empty:
                    if "status" in results_df.columns and "Execution successful" in results_df["status"].values:
                         rows_aff = results_df["rows_affected"].iloc[0]
                         final_answer_string += f"(Query executed successfully. Rows affected: {rows_aff})\n"
                    else:
                         final_answer_string += "(Query returned no results)\n"
                # --- Check for SCALAR result (1 row, 1 column) ---
                elif results_df.shape == (1, 1):
                    scalar_value = results_df.iloc[0, 0]
                    result_was_scalar = True # Set flag
                    # Format the scalar value nicely
                    if pd.isna(scalar_value):
                         formatted_value = "(No value)"
                    elif isinstance(scalar_value, float):
                         # Format float to appropriate precision, avoid excessive zeros
                         formatted_value = f"{scalar_value:,.2f}".rstrip('0').rstrip('.') if '.' in f"{scalar_value:,.2f}" else f"{scalar_value:,.0f}"
                    elif isinstance(scalar_value, int):
                         formatted_value = f"{scalar_value:,}" # Add comma separators for integers
                    else:
                         formatted_value = str(scalar_value)
                    final_answer_string += f"**{formatted_value}**\n" # Display the single value directly and bolded
                    log_message(f"Displayed scalar result: {formatted_value}", "debug")
                # --- End Scalar Check ---
                else:
                    # Format as table for non-scalar results
                    try:
                        max_rows_display = 20
                        max_cols_display = 10
                        df_display = results_df.head(max_rows_display)
                        cols_omitted = False

                        if len(df_display.columns) > max_cols_display:
                            df_display = df_display.iloc[:, :max_cols_display]
                            cols_omitted = True

                        # Use a simpler table format or ensure markdown rendering works
                        final_answer_string += df_display.to_markdown(index=False) + "\n"

                        if cols_omitted:
                            final_answer_string += f"\n*(Displaying first {max_cols_display} of {len(results_df.columns)} columns)*"
                        if len(results_df) > max_rows_display:
                            final_answer_string += f"\n*(Showing first {max_rows_display} of {len(results_df)} total rows)*"
                        # Add a newline after table info
                        final_answer_string += "\n"

                    except Exception as fmt_err:
                         log_message(f"Error formatting DataFrame to markdown: {fmt_err}", "warning")
                         final_answer_string += "(Could not display results table nicely. Data was retrieved.)\n"
            else:
                final_answer_string += "(No DataFrame results returned)\n"

            # --- Optional SQL and Explanation ---
            # Add these within an expander or conditionally
            with st.expander("Query Details (SQL & Explanation)", expanded=False):
                 st.markdown(f"**Generated SQL Query:**\n```sql\n{sql_query}\n```")
                 st.markdown(f"**Explanation:**\n{explanation}")

            # Alternatively, add directly but less prominently if result was scalar:
            # if not result_was_scalar:
            #      final_answer_string += f"\n**Generated SQL Query:**\n```sql\n{sql_query}\n```"
            #      final_answer_string += f"\n**Explanation:**\n{explanation}"
            # else:
            #      # Maybe add a note that details are available?
            #      final_answer_string += "\n*(SQL query and explanation are available)*"


            # Add metadata
            metadata.append({
                "source": f"database_{current_chat_state['chat_id']}",
                "database": chat_db_path,
                "tables": ", ".join(imported_tables), # Keep as string
                "content_type": "database_query_result"
            })

    except Exception as e:
        st.exception(e)
        log_message(f"Critical error in answer_data_query: {e}", "error")
        final_answer_string += f"\n\nAn unexpected error occurred while trying to answer the data query: {e}"
    finally:
        if analyzer:
            analyzer.close()

    return final_answer_string.strip(), metadata # Remove trailing whitespace


# --- generate_answer_from_context (RAG - Keep As Is) ---
# ... (rest of the function is unchanged) ...
def generate_answer_from_context(query, context_docs, context_metadatas, text_model_name): # Added text_model_name
    """Generates answer using the specified LLM based purely on retrieved text context (RAG)."""
    if not context_docs:
        log_message("No relevant context found in documents for the query.", "warning")
        # Check if the context contains markers for data files
        data_markers = [m for m in context_metadatas if m and m.get('content_type', '').startswith('data_import')]
        if data_markers:
            imported_files = list(set(m.get('source') for m in data_markers if m.get('source')))
            return f"I couldn't find relevant information in the text documents for '{query}'. However, I see that data file(s) ({', '.join(imported_files)}) were imported. Try asking a specific question about the data content (e.g., 'how many rows in {imported_files[0]}?', 'what is the total sales?').", []
        else:
            return "I cannot answer this question based on the provided documents.", []


    context_items = []
    source_files = set()
    meta_list_for_response = [] # Store metadata associated with used context

    for doc, meta in zip(context_docs, context_metadatas):
         if not meta or not doc: continue # Skip if no metadata or empty doc
         source = meta.get('source', 'Unknown')
         source_files.add(source)
         citation = f"Source: {source}"

         # Add specific location info if available
         if meta.get('page_number'): citation += f" (Page {meta.get('page_number')})"
         elif meta.get('slide_number'): citation += f" (Slide {meta.get('slide_number')})"
         elif meta.get('crawled_url'): citation += f" (From URL: {meta.get('crawled_url')})"
         elif meta.get('content_type') == 'data_import_success':
              # Special citation for data markers - don't include the marker text itself in context
              citation += " (Data File - Imported, ask questions about content)"
              context_items.append(f"{citation}\nContent:\n[Marker for imported data file '{source}'. Query its content directly.]")
         elif meta.get('content_type') == 'data_import_failed':
               citation += " (Data File - Import Failed)"
               context_items.append(f"{citation}\nContent:\n[Marker for failed data import '{source}'.]")
         else: # Regular text chunk
             context_items.append(f"{citation}\nContent:\n{doc}")

         meta_list_for_response.append(meta) # Keep metadata for sources display


    context_str = "\n\n---\n\n".join(context_items)

    prompt = f"""Answer the following question based *only* on the provided context below. The context may include text excerpts, image descriptions, or markers indicating that data files were imported.

If the question relates to a data file indicated by a marker (e.g., "[Data File Imported: ...]"), explain that the data needs to be queried directly and suggest asking a specific question about its content (like "how many rows?", "what is the total revenue?"). Do not try to answer based on the marker text itself.

For all other questions, synthesize an answer from the text/image content provided. Cite the sources (filename, page number, slide number, URL, or image reference) mentioned in the context for the information used in your answer. If the context doesn't contain the answer, state that clearly.

**Provided Context:**
{context_str}
**User Question:**
{query}

**Answer (with citations where applicable):**
"""

    try:
        model = genai.GenerativeModel(text_model_name)
        generation_config = genai.types.GenerationConfig(temperature=0.3)
        response = model.generate_content(prompt, generation_config=generation_config)

        if hasattr(response, 'text'):
            answer_text = response.text
            # Add citation hint if LLM forgets and sources are few
            unique_sources = set(m.get("source", "Unknown") for m in meta_list_for_response if m and not m.get("content_type", "").startswith("data_import"))
            if not any(f in answer_text for f in unique_sources) and len(unique_sources) < 4 and unique_sources:
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
# ... (keep as is, potentially add logic to avoid follow-ups for scalar data results) ...
def generate_followup_questions(query, answer, text_model_name): # Added text_model_name
     """Generates relevant follow-up questions using the specified LLM."""
     # Avoid generating follow-ups for data query results or errors
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
         generation_config = genai.types.GenerationConfig(temperature=0.5) # Slightly more creative temp
         response = model.generate_content(prompt, generation_config=generation_config)

         if hasattr(response, 'text'):
             response_text = response.text.strip()
             log_message(f"Raw follow-up suggestions: {response_text}", "debug")
             # Try parsing as a Python list literal first
             match = re.search(r'\[\s*".*?"\s*(?:,\s*".*?"\s*)*\]', response_text, re.DOTALL)
             if match:
                  list_str = match.group()
                  try:
                       import ast
                       followups = ast.literal_eval(list_str)
                       if isinstance(followups, list) and all(isinstance(q, str) for q in followups):
                            log_message(f"Parsed follow-ups via ast: {followups}", "debug")
                            return followups[:3] # Take up to 3
                  except Exception as eval_err:
                       log_message(f"ast.literal_eval failed for follow-up list '{list_str}': {eval_err}", "warning")
                       # Fallback to line parsing if ast fails

             # Fallback: Extract lines ending with '?'
             lines = [line.strip(' -*\'" ') for line in response_text.split('\n')]
             questions = [line for line in lines if line.endswith('?') and len(line) > 10] # Basic sanity check
             if questions:
                  log_message(f"Fallback parsed follow-ups (lines): {questions[:3]}", "debug")
                  return questions[:3]

             log_message("Could not reliably parse follow-up suggestions into a list.", "warning")
             return []
         else:
              log_message("LLM response for follow-ups has no text part.", "warning")
              return []
     except Exception as e:
         # Don't show full exception to user, just log it
         log_message(f"Error generating follow-up questions with LLM {text_model_name}: {e}", "error")
         return []

# --- END OF FILE qa_engine.py ---