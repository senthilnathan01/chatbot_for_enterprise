import os
import argparse
from typing import List
from pathlib import Path
from DataImporter_Gemini import DataImporter_Gemini # Use Gemini version
from VannaReplica_Gemini import VannaReplica_Gemini # Use Gemini version
import pandas as pd
import logging
import sys
import json # For pretty printing info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Ensure logs go to console
    ]
)
logger = logging.getLogger(__name__) # Logger name will be VannaDataAnalyzer_Gemini

class VannaDataAnalyzer_Gemini:
    """
    Integration class combining DataImporter_Gemini and VannaReplica_Gemini
    for end-to-end data analysis from files to NL querying using Gemini.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-1.5-flash", # Default Gemini model
        db_path: str = "vanna_gemini_data.db", # Default DB name
        verbose: bool = False
    ):
        """
        Initialize the VannaDataAnalyzer_Gemini.

        Args:
            api_key: Google Generative AI API key (or set GEMINI_API_KEY env var).
            model: Gemini model name to use.
            db_path: Path to SQLite database file.
            verbose: Whether to print verbose output.
        """
        self.db_path = db_path
        self.verbose = verbose
        self.api_key = api_key # Store API key if needed later

        # Initialize importer
        self.importer = DataImporter_Gemini(db_path=db_path, verbose=verbose)

        # Ensure connection string is available *after* importer potentially creates the DB file/path
        db_connection_string = self.importer.get_connection_string()

        # Initialize VannaReplica_Gemini with connection to the same database
        # Pass API key explicitly
        self.vanna = VannaReplica_Gemini(
            api_key=self.api_key, # Pass the key
            model=model,
            db_uri=db_connection_string,
            verbose=verbose
        )

        logger.info(f"VannaDataAnalyzer_Gemini initialized with database at {db_path} using model {model}")

    def import_data(self, file_path: str, file_type: str, **kwargs) -> List[str]:
        """
        Import data from a file using the appropriate DataImporter method.

        Args:
            file_path: Path to the data file or glob pattern.
            file_type: Type of file ('excel', 'csv', 'json').
            **kwargs: Additional arguments for the specific importer method
                      (e.g., sheet_name, table_name, delimiter, pk, if_exists).

        Returns:
            List of table names created or updated.
        """
        logger.info(f"Attempting to import {file_type} data from: {file_path}")
        if file_type == 'excel':
            return self.importer.import_excel(file_path, **kwargs)
        elif file_type == 'csv':
            return self.importer.import_csv(file_path, **kwargs)
        elif file_type == 'json':
             # Basic JSON import - assumes list of records or similar pandas-compatible format
             logger.info("Using basic JSON import via pandas.")
             try:
                 df = pd.read_json(file_path) # Add kwargs like orient if needed via **kwargs
                 table_name = kwargs.get('table_name', Path(file_path).stem)
                 table_name = self.importer._clean_column_name(table_name)
                 pk = kwargs.get('pk')
                 if_exists = kwargs.get('if_exists', 'replace')
                 self.importer._import_dataframe(df, table_name, primary_key=pk, if_exists=if_exists)
                 logger.info(f"Successfully imported JSON to table '{table_name}'")
                 return [table_name]
             except Exception as e:
                  logger.error(f"Failed to import JSON file {file_path}: {e}", exc_info=True)
                  return [] # Return empty list on failure
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Use 'excel', 'csv', or 'json'.")


    def ask(self, question: str, explain: bool = True):
        """Process a natural language question with VannaReplica_Gemini"""
        logger.info(f"Processing question: {question}")
        # Make sure Vanna has metadata loaded, especially if DB was just created
        if not self.vanna.db_metadata_text:
             logger.info("Vanna metadata not loaded, attempting to extract...")
             try:
                  self.vanna._extract_db_metadata()
             except Exception as e:
                  logger.error(f"Failed to extract metadata before asking: {e}")
                  return {"question": question, "error": f"Failed to load database metadata: {e}"}

        return self.vanna.ask(question, explain=explain)

    def get_tables_info(self):
        """Get database schema information"""
        logger.info("Retrieving database schema information...")
        return self.importer.get_tables_info()

    def close(self):
        """Close connections"""
        logger.info("Closing VannaDataAnalyzer connections...")
        if self.importer:
            self.importer.close()
        # VannaReplica_Gemini doesn't hold persistent connections itself,
        # but the engine it uses is managed by the importer instance.
        logger.info("Connections closed.")

def main():
    """Command line interface for VannaDataAnalyzer_Gemini"""
    parser = argparse.ArgumentParser(description='Import data files and analyze with natural language queries using Gemini.')

    parser.add_argument('--db', default='vanna_gemini_data.db', help='Path to SQLite database file (default: vanna_gemini_data.db)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for detailed logging')
    parser.add_argument('--api-key', help='Google Generative AI API key (can also set GEMINI_API_KEY environment variable)')
    parser.add_argument('--model', default='gemini-1.5-flash', help='Gemini model name to use (default: gemini-1.5-flash)')

    subparsers = parser.add_subparsers(dest='command', help='Command to execute', required=True)

    # --- Import Command ---
    import_parser = subparsers.add_parser('import', help='Import data from a file into the database')
    import_parser.add_argument('--type', required=True, choices=['excel', 'csv', 'json'], help='Type of the input file')
    import_parser.add_argument('--file', required=True, help='Path to the data file or glob pattern (e.g., "data/*.csv")')
    # Common import arguments
    import_parser.add_argument('--table', help='Base table name for import. Defaults to filename stem or sheet name.')
    import_parser.add_argument('--pk', help='Primary key column name (original name before cleaning)')
    import_parser.add_argument('--if-exists', default='replace', choices=['fail', 'replace', 'append'], help='Action if table exists (default: replace)')
    # Excel specific
    import_parser.add_argument('--sheet', help='[Excel only] Sheet name/index to import, or "None" for all (default: 0)')
    # CSV specific
    import_parser.add_argument('--delimiter', default=',', help='[CSV only] Delimiter character (default: ",")')
    import_parser.add_argument('--encoding', default='utf-8', help='[CSV only] File encoding (default: utf-8)')
    # JSON specific (add more if needed, like orient)


    # --- Ask Command ---
    ask_parser = subparsers.add_parser('ask', help='Ask a natural language question about the data')
    ask_parser.add_argument('question', help='The natural language question to ask')
    ask_parser.add_argument('--no-explain', action='store_true', help='If set, do not generate an explanation for the SQL query')

    # --- Info Command ---
    info_parser = subparsers.add_parser('info', help='Display information about the database schema')

    # --- Query Command ---
    query_parser = subparsers.add_parser('query', help='Execute a raw SQL query against the database')
    query_parser.add_argument('sql', help='The SQL query to execute directly')


    args = parser.parse_args()

    # --- Initialize Analyzer ---
    analyzer = None # Initialize to None
    try:
        analyzer = VannaDataAnalyzer_Gemini(
            api_key=args.api_key, # Will check env var if None
            model=args.model,
            db_path=args.db,
            verbose=args.verbose
        )

        # --- Execute Command ---
        if args.command == 'import':
            import_args = {
                "if_exists": args.if_exists,
                "table_name": args.table,
                "pk": args.pk
            }
            if args.type == 'excel':
                 sheet_name_arg = args.sheet
                 if sheet_name_arg == "None": sheet_name = None
                 elif sheet_name_arg:
                     try: items = [int(s.strip()) if s.strip().isdigit() else s.strip() for s in sheet_name_arg.split(',')]
                     except ValueError: items = sheet_name_arg # Treat as single name if mixed/invalid
                     sheet_name = items[0] if len(items) == 1 else items
                 else: sheet_name = 0 # Default
                 import_args["sheet_name"] = sheet_name
                 # Simplified CLI handling for table/pk mapping with excel
                 if args.table and isinstance(sheet_name, (str, int)):
                      import_args["table_names"] = {sheet_name: args.table}
                      if args.pk: import_args["primary_keys"] = {args.table: args.pk}
                 else: # Use default naming if multiple sheets or no table name given
                      logger.info("Using default table naming (from sheet names). --table and --pk ignored for multiple sheets.")

            elif args.type == 'csv':
                import_args["delimiter"] = args.delimiter
                import_args["encoding"] = args.encoding
            # elif args.type == 'json':
                # Add specific JSON args if needed, e.g., import_args["orient"] = args.orient

            tables = analyzer.import_data(args.file, args.type, **import_args)

            if tables:
                print(f"\n✅ Successfully imported/updated tables: {', '.join(tables)}")
            else:
                print(f"\n⚠️ Import process completed, but no tables were created or updated (check logs for details).")


        elif args.command == 'ask':
            result = analyzer.ask(args.question, explain=not args.no_explain)

            print("\n--- Query Analysis ---")
            print(f"❓ Question: {result.get('question')}")

            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                if 'sql_query' in result: # Print SQL even if execution failed
                    print(f"\nGenerated SQL (failed execution):\n```sql\n{result['sql_query']}\n```")

            else:
                print(f"\n🔍 Generated SQL Query:\n```sql\n{result.get('sql_query', 'N/A')}\n```")

                print("\n📊 Results:")
                results_data = result.get('results')
                if isinstance(results_data, pd.DataFrame):
                    if results_data.empty:
                        print("(Query returned no results)")
                    else:
                         # Limit display for very wide tables in console
                        pd.set_option('display.max_columns', 15)
                        pd.set_option('display.width', 120)
                        print(results_data.to_string(index=False))
                else:
                    print("(No DataFrame results returned)")

                if 'explanation' in result:
                    print("\n📘 Explanation:")
                    print(result['explanation'])
                elif not args.no_explain:
                     print("\n📘 Explanation: (Not generated or failed)")


        elif args.command == 'info':
            tables_info = analyzer.get_tables_info()
            print("\n--- Database Schema Information ---")
            print(f"Database Path: {analyzer.db_path}")
            # Use json.dumps for pretty printing the schema dict
            print(json.dumps(tables_info, indent=2, default=str)) # default=str handles non-serializable types like datetime

        elif args.command == 'query':
            print(f"\nExecuting SQL: {args.sql}")
            try:
                # Use the importer's execute_query method
                results = analyzer.importer.execute_query(args.sql)
                print("\nQuery Results:")
                if isinstance(results, pd.DataFrame):
                    if results.empty:
                         print("(Query returned no results or was not a SELECT statement)")
                    else:
                         pd.set_option('display.max_columns', 15)
                         pd.set_option('display.width', 120)
                         print(results.to_string(index=False))
                elif isinstance(results, list): # Fallback for non-pandas results
                     print(json.dumps(results, indent=2, default=str))
                else:
                     print("(No results returned or unexpected format)")
            except Exception as e:
                print(f"\n❌ Error executing query: {e}")


        else:
            parser.print_help()

    except ValueError as ve:
         print(f"\nConfiguration Error: {ve}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        logger.error("Unhandled exception in VannaDataAnalyzer main", exc_info=True) # Log stack trace
        sys.exit(1)
    finally:
        if analyzer:
            analyzer.close()

if __name__ == "__main__":
    main()