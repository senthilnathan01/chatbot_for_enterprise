# --- START OF FILE VannaReplica_Gemini.py ---

import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union
# Removed: import openai
import google.generativeai as genai # Gemini API
from sqlalchemy import create_engine, inspect, text
import json
import re
import logging
import time # For potential retries

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Logger name will be VannaReplica_Gemini

class VannaReplica_Gemini:
    """
    A simplified replica of Vanna.ai, using Google Gemini,
    that converts natural language questions to SQL queries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash", # Default to a capable and fast Gemini model
        db_uri: Optional[str] = None,
        verbose: bool = False,
        max_retries: int = 2,
        retry_delay: int = 5 # seconds
    ):
        """
        Initialize the VannaReplica_Gemini assistant.

        Args:
            api_key: Google Generative AI API key. If None, looks for GEMINI_API_KEY env var.
            model: Gemini model name to use (e.g., "gemini-1.5-pro", "gemini-1.5-flash").
            db_uri: Database connection URI (SQLAlchemy format).
            verbose: Whether to print verbose output.
            max_retries: Maximum number of retries for API calls.
            retry_delay: Delay between retries in seconds.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key must be provided via api_key argument or GEMINI_API_KEY environment variable")

        self.model_name = model
        self.db_uri = db_uri
        self.verbose = verbose
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.db_schema_info = None
        self.db_metadata_text = None
        self.engine = None
        self.genai_model = None

        try:
            # Configure Gemini client
            genai.configure(api_key=self.api_key)
            # Check available models (optional, for debugging)
            # for m in genai.list_models():
            #     if 'generateContent' in m.supported_generation_methods:
            #         logger.debug(m.name)
            self.genai_model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini client configured with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to configure Gemini client: {e}", exc_info=True)
            raise

        # Connect to database if URI is provided
        if db_uri:
            self.connect(db_uri)

    def connect(self, db_uri: str) -> None:
        """
        Connect to a database and extract its schema.

        Args:
            db_uri: Database connection URI (SQLAlchemy format)
        """
        try:
            self.engine = create_engine(db_uri)
            # Test connection
            with self.engine.connect() as connection:
                 logger.info("Database connection test successful.")
            self.db_uri = db_uri
            self._extract_db_metadata()
            logger.info(f"Successfully connected to database and extracted metadata from {db_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to database or extract metadata: {e}", exc_info=True)
            self.engine = None # Ensure engine is None if connection failed
            raise

    def _extract_db_metadata(self) -> None:
        """Extract database schema and metadata using SQLAlchemy Inspector."""
        if not self.engine:
            raise ValueError("Database connection not established. Call connect() first.")

        logger.info("Extracting database metadata...")
        inspector = inspect(self.engine)
        schema_info = {}
        schema_text_parts = []

        try:
             table_names = inspector.get_table_names()
             logger.info(f"Found tables: {table_names}")

             for table_name in table_names:
                 if table_name.startswith('sqlite_'): # Skip internal SQLite tables
                     continue

                 logger.debug(f"Processing table: {table_name}")
                 schema_text_parts.append(f"Table: {table_name}")

                 # Columns
                 columns = []
                 column_texts = []
                 try:
                     for column in inspector.get_columns(table_name):
                         col_info = {
                             "name": column["name"],
                             "type": str(column["type"]),
                             "nullable": column.get("nullable", True),
                             "primary_key": column.get("primary_key", False) # Added PK info here
                         }
                         columns.append(col_info)
                         pk_marker = " (PK)" if col_info["primary_key"] else ""
                         column_texts.append(f"  - {col_info['name']} ({col_info['type']}){pk_marker}")
                     schema_text_parts.append("Columns:")
                     schema_text_parts.extend(column_texts)
                 except Exception as e:
                      logger.warning(f"Could not get column info for table {table_name}: {e}")
                      schema_text_parts.append("  - Error retrieving columns")
                      columns = [{"error": str(e)}]

                 # Primary Keys (explicitly list constraints)
                 primary_keys = []
                 try:
                     pk_constraint = inspector.get_pk_constraint(table_name)
                     primary_keys = pk_constraint.get("constrained_columns", [])
                     if primary_keys:
                         schema_text_parts.append(f"Primary Key: ({', '.join(primary_keys)})")
                     # else: schema_text_parts.append("Primary Key: None") # Optional: be explicit
                 except Exception as e:
                      logger.warning(f"Could not get primary key info for table {table_name}: {e}")
                      schema_text_parts.append("Primary Key: Error retrieving")

                 # Foreign Keys
                 foreign_keys = []
                 fk_texts = []
                 try:
                     for fk in inspector.get_foreign_keys(table_name):
                         fk_info = {
                             "constrained_columns": fk["constrained_columns"],
                             "referred_table": fk["referred_table"],
                             "referred_columns": fk["referred_columns"]
                         }
                         foreign_keys.append(fk_info)
                         refs = f"({', '.join(fk_info['constrained_columns'])}) -> {fk_info['referred_table']}({', '.join(fk_info['referred_columns'])})"
                         fk_texts.append(f"  - {refs}")
                     if fk_texts:
                         schema_text_parts.append("Foreign Keys:")
                         schema_text_parts.extend(fk_texts)
                     # else: schema_text_parts.append("Foreign Keys: None") # Optional: be explicit
                 except Exception as e:
                      logger.warning(f"Could not get foreign key info for table {table_name}: {e}")
                      schema_text_parts.append("Foreign Keys: Error retrieving")
                      foreign_keys = [{"error": str(e)}]

                 # Sample data (first 3 rows for brevity in prompt)
                 sample_data = []
                 try:
                     # Use pandas for robust reading
                     sample_df = pd.read_sql(f"SELECT * FROM \"{table_name}\" LIMIT 3", self.engine)
                     # Convert to list of dicts, handling potential non-JSON serializable types
                     sample_data = sample_df.astype(object).where(pd.notnull(sample_df), None).to_dict('records')
                     if sample_data:
                          schema_text_parts.append("Sample rows:")
                          # Represent sample data concisely
                          sample_headers = list(sample_data[0].keys())
                          schema_text_parts.append(f"  Columns: {', '.join(sample_headers)}")
                          for row in sample_data:
                              row_values = [str(row.get(h, '')) for h in sample_headers]
                              schema_text_parts.append(f"  Data: {', '.join(row_values)}")

                 except Exception as e:
                     logger.warning(f"Could not fetch sample data for table {table_name}: {e}")
                     # Optionally add placeholder to schema text: schema_text_parts.append("Sample rows: Error retrieving")
                     sample_data = [{"error": f"Could not fetch sample data: {e}"}]

                 schema_info[table_name] = {
                     "columns": columns,
                     "foreign_keys": foreign_keys,
                     "primary_keys": primary_keys,
                     "sample_data": sample_data
                 }
                 schema_text_parts.append("---") # Separator between tables

             self.db_schema_info = schema_info
             self.db_metadata_text = "\n".join(schema_text_parts)

             if self.verbose:
                  logger.info("Database Metadata for Prompt:\n" + self.db_metadata_text)
             logger.info("Database metadata extracted successfully.")

        except Exception as e:
             logger.error(f"An error occurred during metadata extraction: {e}", exc_info=True)
             self.db_schema_info = None
             self.db_metadata_text = None
             raise # Re-raise the exception

    def _call_gemini_with_retry(self, prompt: str, generation_config: genai.types.GenerationConfig) -> str:
        """Calls the Gemini API with retry logic."""
        if not self.genai_model:
            raise ValueError("Gemini model not initialized.")

        attempts = 0
        while attempts <= self.max_retries:
            attempts += 1
            try:
                if self.verbose:
                    logger.info(f"--- Sending Prompt to Gemini (Attempt {attempts}) ---")
                    logger.info(prompt)
                    logger.info("--- End Prompt ---")

                response = self.genai_model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    # Add safety settings if needed, e.g.,
                    # safety_settings=[
                    #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    # ]
                )

                # Basic check for blocked content or lack of response text
                if not response.parts:
                    # Check finish_reason if available (might indicate safety blocking, etc.)
                    finish_reason = getattr(response, 'prompt_feedback', None)
                    block_reason = getattr(finish_reason, 'block_reason', 'Unknown') if finish_reason else 'Unknown'
                    error_message = f"Gemini response missing content. Finish/Block Reason: {block_reason}. Full response: {response}"
                    logger.warning(error_message)
                    # Depending on the block reason, you might not want to retry
                    if 'SAFETY' in str(block_reason).upper():
                         raise ValueError(f"Content blocked by Gemini safety filters: {block_reason}")
                    # Otherwise, treat as a potentially transient issue and retry if applicable
                    if attempts > self.max_retries:
                        raise ValueError(error_message) # Raise after final retry
                    else:
                        logger.info(f"Retrying after {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue # Go to next retry iteration


                result_text = response.text # Access generated text

                if self.verbose:
                    logger.info(f"--- Gemini Raw Response Text (Attempt {attempts}) ---")
                    logger.info(result_text)
                    logger.info("--- End Raw Response ---")
                return result_text # Success

            except Exception as e:
                logger.error(f"Gemini API call failed (Attempt {attempts}): {e}", exc_info=True)
                if attempts > self.max_retries:
                    logger.error("Max retries exceeded.")
                    raise # Re-raise the exception after final attempt
                logger.info(f"Retrying after {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)

        # Should not be reached if logic is correct, but as a safeguard:
        raise RuntimeError("Gemini API call failed after all retries.")


    def _generate_sql_prompt(self, question: str) -> str:
        """
        Generate a prompt for the Gemini model to convert natural language to SQL.

        Args:
            question: Natural language question to convert to SQL

        Returns:
            Formatted prompt for the LLM
        """
        if not self.db_metadata_text:
            raise ValueError("Database metadata text is not available. Connect first.")

        # Improved prompt structure for clarity
        return f"""You are an expert SQL query generator using the SQLite dialect. Your task is to convert natural language questions into valid SQL queries based on the provided database schema.

DATABASE SCHEMA INFORMATION:
--- Schema Start ---
{self.db_metadata_text}
--- Schema End ---

TASK:
Convert the following user question into a single, valid SQLite SQL query.

USER QUESTION: "{question}"

GUIDELINES:
1.  **Output ONLY the SQL query.** Do not include any explanations, comments, markdown formatting (like ```sql), or introductory text.
2.  Ensure the query is syntactically correct for **SQLite**.
3.  Use the exact table and column names provided in the schema. Pay attention to case sensitivity if applicable (though SQLite names are typically case-insensitive).
4.  Use JOIN clauses when the question requires data from multiple tables, based on the foreign key relationships shown in the schema.
5.  Include appropriate aggregate functions (e.g., COUNT, SUM, AVG, MAX, MIN) if the question implies aggregation.
6.  Use WHERE clauses to filter data as requested by the question.
7.  If the question is ambiguous, make reasonable assumptions based on the schema and common interpretations. If it's impossible to generate a query, output only the text "INVALID_QUESTION".
8.  Format the SQL query for readability if possible, but prioritize correctness.

SQL QUERY:
"""


    def _generate_explanation_prompt(self, question: str, sql_query: str) -> str:
        """
        Generate a prompt for the Gemini model to explain the SQL query.

        Args:
            question: Original natural language question
            sql_query: Generated SQL query

        Returns:
            Formatted prompt for the LLM
        """
        return f"""You are an expert at explaining SQL queries in simple terms to a non-technical user.

CONTEXT:
A user asked the following question about their data:
"{question}"

The following SQLite SQL query was generated to answer the question:
```sql
{sql_query}
TASK:
Provide a clear, concise, step-by-step explanation of what this SQL query does and how it answers the user's question.
GUIDELINES:
Start with a brief summary of the query's overall goal.
Explain each main clause (SELECT, FROM, WHERE, JOIN, GROUP BY, ORDER BY, LIMIT) present in the query and what it contributes.
Mention which tables and columns are used and why.
If there are JOINs, explain how the tables are linked.
If there are calculations or aggregate functions, explain what they compute.
Briefly state how the final result addresses the original question.
Keep the language simple and avoid technical jargon where possible.
Do not just repeat the SQL code. Explain the logic.
EXPLANATION:
"""
    def generate_sql(self, question: str) -> str:
        """
        Generate a SQL query from a natural language question using Gemini.

        Args:
            question: Natural language question

        Returns:
            SQL query string
        """
        if not self.db_metadata_text:
            self._extract_db_metadata() # Attempt to extract metadata if not already done
        if not self.db_metadata_text:
            raise ValueError("Database metadata not available. Connect to a database first.")


        prompt = self._generate_sql_prompt(question)
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,      # Low temperature for deterministic SQL
            max_output_tokens=1024 # Allow sufficient length for complex queries
            # top_p=, top_k= # Optional parameters
        )

        sql_query_raw = self._call_gemini_with_retry(prompt, generation_config)

        # Clean up the response to extract just the SQL
        # Remove potential markdown code blocks and leading/trailing whitespace
        sql_query = re.sub(r'^```(?:sql)?\s*', '', sql_query_raw, flags=re.IGNORECASE | re.MULTILINE)
        sql_query = re.sub(r'\s*```$', '', sql_query, flags=re.MULTILINE)
        sql_query = sql_query.strip()

        # Basic validation: check if it looks like SQL (e.g., starts with SELECT, UPDATE, etc.)
        if not sql_query or not re.match(r'^(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH)\b', sql_query, re.IGNORECASE):
            # Check if LLM explicitly returned failure marker
            if sql_query.upper() == "INVALID_QUESTION":
                logger.warning(f"LLM indicated question is invalid: '{question}'")
                raise ValueError("The question could not be converted to a valid SQL query.")
            else:
                logger.warning(f"Generated text doesn't look like SQL: '{sql_query_raw}'")
                raise ValueError(f"Failed to generate a valid SQL query. Raw LLM Output: {sql_query_raw}")


        if self.verbose:
            logger.info(f"Generated SQL query: {sql_query}")

        return sql_query

    def explain_query(self, question: str, sql_query: str) -> str:
        """
        Generate an explanation for a SQL query using Gemini.

        Args:
            question: Original natural language question
            sql_query: SQL query to explain

        Returns:
            Explanation of the query
        """
        prompt = self._generate_explanation_prompt(question, sql_query)
        generation_config = genai.types.GenerationConfig(
            temperature=0.6,       # Higher temperature for more natural explanation
            max_output_tokens=1500 # Allow ample space for explanation
        )

        explanation = self._call_gemini_with_retry(prompt, generation_config)
        return explanation.strip()

    def run_query(self, sql_query: str) -> pd.DataFrame:
        """
        Execute a SQL query against the connected database.

        Args:
            sql_query: SQL query to execute

        Returns:
            Pandas DataFrame with query results
        """
        if not self.engine:
            raise ValueError("Database connection not established. Call connect() first.")

        try:
            if self.verbose:
                logger.info(f"Executing query: {sql_query}")

            # Use pandas.read_sql for SELECT, handle others potentially
            # Basic check if it's likely a SELECT query
            if re.match(r'^\s*(SELECT|WITH)\b', sql_query, re.IGNORECASE):
                 result_df = pd.read_sql_query(sql_query, self.engine)
                 # Convert potential Pandas dtypes (like Int64) to standard types if needed
                 # result_df = result_df.astype(object).where(pd.notnull(result_df), None)
                 return result_df
            else:
                 # For non-SELECT statements (INSERT, UPDATE, DELETE, CREATE, etc.)
                 logger.info("Executing non-SELECT query.")
                 with self.engine.begin() as connection: # Use transaction
                      result_proxy = connection.execute(text(sql_query))
                      rowcount = result_proxy.rowcount
                      logger.info(f"Query executed successfully. Rows affected (if applicable): {rowcount}")
                      # Return an empty DataFrame or status message for non-SELECT
                      return pd.DataFrame({"status": ["Execution successful"], "rows_affected": [rowcount]})

        except Exception as e:
            logger.error(f"Query execution failed for SQL: {sql_query}\nError: {e}", exc_info=True)
            # Provide more specific error feedback if possible
            raise RuntimeError(f"Database query failed: {e}") from e # Re-raise with context

    def ask(self, question: str, explain: bool = True) -> Dict[str, Any]:
        """
        Process a natural language question, generate SQL using Gemini,
        execute it, and return results.

        Args:
            question: Natural language question about the data
            explain: Whether to include an explanation of the query

        Returns:
            Dictionary with question, generated SQL, results (DataFrame), and optional explanation/error.
        """
        response = {"question": question}
        try:
            sql_query = self.generate_sql(question)
            response["sql_query"] = sql_query

            results_df = self.run_query(sql_query)
            response["results"] = results_df

            if explain:
                try:
                    explanation = self.explain_query(question, sql_query)
                    response["explanation"] = explanation
                except Exception as explain_e:
                    logger.warning(f"Failed to generate explanation: {explain_e}")
                    response["explanation"] = f"Could not generate explanation: {explain_e}"

            return response

        except (ValueError, RuntimeError, genai.types.generation_types.StopCandidateException) as e:
            logger.error(f"Error processing question '{question}': {e}", exc_info=True)
            response["error"] = str(e)
            return response
        except Exception as e:
            logger.error(f"Unexpected error processing question '{question}': {e}", exc_info=True)
            response["error"] = f"An unexpected error occurred: {e}"
            return response