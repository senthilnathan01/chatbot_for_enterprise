# --- START OF FILE run_vanna_Gemini.py ---

import argparse
import os
import sys
import pandas as pd
from DataImporter_Gemini import DataImporter_Gemini # Use Gemini version
from VannaReplica_Gemini import VannaReplica_Gemini # Use Gemini version
import toml # For reading secrets.toml
import logging
from pathlib import Path
from typing import Optional # <--- *** ADDED THIS IMPORT ***

# Configure basic logging for the script itself
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Logger name will be run_vanna_Gemini

def load_api_key_from_secrets():
    """
    Load Gemini API key from .streamlit/secrets.toml.
    Returns None if file or key is not found.
    """
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        try:
            secrets = toml.load(secrets_path)
            # Look for GEMINI section, then API_KEY
            return secrets.get("GEMINI", {}).get("API_KEY")
        except toml.TomlDecodeError:
            logger.warning(f"Could not decode {secrets_path}. Ensure it's valid TOML.")
        except Exception as e:
            logger.warning(f"Error reading {secrets_path}: {e}")
    return None

def get_api_key(args_key: Optional[str]) -> str: # Now Optional is defined
    """
    Get Gemini API key following precedence: arguments > environment > secrets.toml.
    Raises ValueError if no key is found.
    """
    # 1. From command line argument
    if args_key:
        logger.info("Using API key from command line argument.")
        return args_key

    # 2. From environment variable
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        logger.info("Using API key from GEMINI_API_KEY environment variable.")
        return env_key

    # 3. From .streamlit/secrets.toml
    secrets_key = load_api_key_from_secrets()
    if secrets_key:
        logger.info("Using API key from .streamlit/secrets.toml.")
        return secrets_key

    # If no key found
    raise ValueError(
        "Gemini API key not found. Provide it via --api-key argument, "
        "set the GEMINI_API_KEY environment variable, or add it to "
        ".streamlit/secrets.toml under [GEMINI] section with key API_KEY."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Import data (CSV/Excel/JSON) into SQLite and run natural language queries using Gemini."
    )
    parser.add_argument(
        '--file',
        required=True,
        help='Path to the input data file (CSV, Excel .xlsx/.xls, or JSON).'
    )
    parser.add_argument(
        '--db',
        default='vanna_interactive.db',
        help='Path for the SQLite database file (default: vanna_interactive.db).'
    )
    parser.add_argument(
        '--api-key',
        help='Google Generative AI API key (overrides environment variable and secrets.toml).'
    )
    parser.add_argument(
        '--model',
        default='gemini-1.5-flash',
        help='Gemini model name to use (default: gemini-1.5-flash).'
    )
    parser.add_argument(
        '--table-name',
        help='Optional: Specify a table name for CSV/JSON import (defaults to filename stem).'
              ' For Excel, use DataImporter_Gemini directly for sheet->table mapping.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging for detailed output.'
    )
    parser.add_argument(
        '--if-exists',
        default='replace',
        choices=['fail', 'replace', 'append'],
        help="Action if the table already exists during import (default: replace)."
    )


    args = parser.parse_args()

    # Set logging level based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    # Set root logger level first
    logging.getLogger().setLevel(log_level)
    # Then set levels for our specific loggers if needed (optional, inherits from root)
    # logging.getLogger('DataImporter_Gemini').setLevel(log_level)
    # logging.getLogger('VannaReplica_Gemini').setLevel(log_level)


    importer = None # Ensure importer is defined for finally block

    try:
        # --- Get API Key ---
        api_key = get_api_key(args.api_key)

        # --- Detect File Type ---
        file_path = args.file
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Input file not found: {file_path}")

        file_ext = Path(file_path).suffix.lower()
        if file_ext not in ['.csv', '.xlsx', '.xls', '.json']:
            raise ValueError(f"Unsupported file type: '{file_ext}'. Use .csv, .xlsx, .xls, or .json.")

        # --- Initialize Importer ---
        logger.info(f"Initializing DataImporter for database: {args.db}")
        importer = DataImporter_Gemini(db_path=args.db, verbose=args.verbose)

        # --- Load Data into SQLite ---
        logger.info(f"Starting data import from: {file_path}")
        table_name_arg = args.table_name # Explicit table name from args (mainly for CSV/JSON)
        imported_tables = [] # Initialize list

        if file_ext == '.csv':
            imported_tables = importer.import_csv(
                file_path,
                table_name=table_name_arg,
                if_exists=args.if_exists
                # Add other CSV params like delimiter if needed
            )
        elif file_ext in ['.xlsx', '.xls']:
             # Basic Excel import: imports first sheet by default, uses filename stem if table_name not given
             # For more control (multiple sheets, specific naming), use DataImporter directly or VannaDataAnalyzer
             logger.warning("Importing only the first sheet from Excel file by default.")
             # Determine table name for the single sheet import
             # If table_name_arg is provided, use it. Otherwise, use filename stem.
             excel_table_name = table_name_arg if table_name_arg else Path(file_path).stem
             excel_table_name_cleaned = importer._clean_column_name(excel_table_name)

             imported_tables = importer.import_excel(
                 file_path,
                 sheet_name=0, # Import only the first sheet (index 0)
                 table_names={0: excel_table_name_cleaned}, # Map index 0 to the determined name
                 if_exists=args.if_exists
             )
        elif file_ext == '.json':
             # Basic JSON import using pandas helper within importer
             actual_table_name = table_name_arg if table_name_arg else Path(file_path).stem
             actual_table_name_cleaned = importer._clean_column_name(actual_table_name)
             logger.info(f"Importing JSON to table '{actual_table_name_cleaned}'")
             df = pd.read_json(file_path) # Add options like orient if necessary
             importer._import_dataframe(
                 df,
                 table_name=actual_table_name_cleaned,
                 if_exists=args.if_exists
                 # primary_key= can be added if needed
             )
             imported_tables = [actual_table_name_cleaned] # Assume one table for simple JSON import

        if not imported_tables:
             # Check if the table might exist due to 'append' or 'fail' mode but import method returned empty
             if args.if_exists != 'replace':
                  logger.warning(f"Import command finished, but no new tables reported (possibly due to if_exists='{args.if_exists}'). Checking DB...")
                  # Verify if the target table exists now
                  current_tables = importer.get_tables_info().keys()
                  target_table = importer._clean_column_name(table_name_arg if table_name_arg else Path(file_path).stem)
                  if target_table in current_tables:
                      logger.info(f"Target table '{target_table}' exists in the database.")
                  else:
                     raise RuntimeError(f"Data import failed. Target table '{target_table}' not found in DB after import.")
             else: # If 'replace' was used and nothing was imported, it's an error
                raise RuntimeError("Data import failed. No tables were created or updated. Check logs.")


        print(f"\n✅ Data source processed. Database '{args.db}' should contain table(s) like: {', '.join(imported_tables) if imported_tables else '(check logs)'}")

        # --- Initialize Vanna ---
        logger.info(f"Initializing VannaReplica with model '{args.model}'")
        db_connection_string = importer.get_connection_string()
        vanna = VannaReplica_Gemini(
            api_key=api_key,
            model=args.model,
            db_uri=db_connection_string,
            verbose=args.verbose
        )
        # Explicitly load metadata after import, before query loop
        logger.info("Loading database metadata for Vanna...")
        vanna.connect(db_connection_string) # This calls _extract_db_metadata


        # --- Interactive Query Loop ---
        print("\n💬 Database ready. Ask a question about your data.")
        print("   Type 'exit' or 'quit' to stop.")
        print("-" * 30)

        while True:
            try:
                user_input = input(">> ")
                if user_input.lower() in ['exit', 'quit']:
                    print("Exiting.")
                    break
                if not user_input.strip():
                    continue

                # Ask Vanna
                result = vanna.ask(user_input, explain=True) # Always explain in interactive mode

                print("\n--- Analysis ---")
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                    # Attempt to print SQL even if execution failed, if it was generated
                    if 'sql_query' in result and result['sql_query']:
                        print(f"Generated SQL (failed execution):\n```sql\n{result['sql_query']}\n```")
                    elif 'sql_query (raw)' in result: # Check for raw if main generation failed
                         print(f"Raw LLM output (failed generation):\n{result['sql_query (raw)']}")

                else:
                    print(f"🔍 SQL Query:\n```sql\n{result.get('sql_query', 'N/A')}\n```")

                    print("\n📊 Results:")
                    results_df = result.get('results')
                    if isinstance(results_df, pd.DataFrame):
                        if results_df.empty:
                             # Check if it was a non-SELECT query that succeeded
                             if "status" in results_df.columns and "Execution successful" in results_df["status"].values:
                                 rows_aff = results_df["rows_affected"].iloc[0]
                                 print(f"(Query executed successfully. Rows affected: {rows_aff})")
                             else:
                                print("(Query returned no results)")
                        else:
                             # Adjust display settings for console
                             pd.set_option('display.max_rows', 100)
                             pd.set_option('display.max_columns', 20)
                             pd.set_option('display.width', 100)
                             print(results_df.to_string(index=False))
                    else:
                        print("(No DataFrame results to display)")

                    if 'explanation' in result:
                        print("\n📘 Explanation:")
                        print(result['explanation'])
                    print("-" * 30)


            except KeyboardInterrupt:
                print("\nExiting.")
                break
            except Exception as e:
                print(f"\n⚠️ An error occurred in the query loop: {e}")
                logger.error("Error during interactive query", exc_info=True)
                print("-" * 30)


    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"\n❌ Configuration or Execution Error: {e}", file=sys.stderr)
        logger.error(f"Error: {e}", exc_info=True if args.verbose else False)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}", file=sys.stderr)
        logger.error("Unhandled exception in main script", exc_info=True)
        sys.exit(1)
    finally:
        # Ensure importer connection is closed if initialized
        if importer:
            importer.close()
        logger.info("Script finished.")


if __name__ == '__main__':
    main()
# --- END OF FILE run_vanna_Gemini.py ---