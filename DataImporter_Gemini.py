# --- START OF FILE DataImporter_Gemini.py ---

# ... (Keep all imports and the DataImporter_Gemini class definition exactly as before) ...
import os
import pandas as pd
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
import re
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, MetaData, ForeignKey, Boolean, DateTime, Date, inspect, text
import datetime
import argparse # Keep the import
import glob
from pathlib import Path
import sys
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataImporter_Gemini:
    # --- Keep ALL class methods (__init__, connect, _clean_column_name, etc.) AS IS ---
    # ... (previous methods here) ...
    def __init__(self, db_path: str = None, verbose: bool = False):
        self.db_path = db_path
        self.verbose = verbose
        self.conn = None
        self.engine = None
        self.metadata = MetaData()
        if db_path:
            self.connect(db_path)

    def connect(self, db_path: str) -> None:
        try:
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                logger.info(f"Created directory: {db_dir}")
            sqlite_uri = f"sqlite:///{db_path}"
            self.engine = create_engine(sqlite_uri)
            self.db_path = db_path
            with self.engine.connect() as connection:
                logger.debug("SQLAlchemy engine connection test successful.")
            if self.verbose:
                logger.info(f"Connected to database engine for {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _clean_column_name(self, name: str) -> str:
        if not isinstance(name, str): name = str(name)
        name = re.sub(r'[^\w\s]', '_', name)
        name = re.sub(r'\s+', '_', name)
        name = name.strip('_')
        if name and name[0].isdigit(): name = f"col_{name}"
        name = name.lower()
        if not name: name = "unnamed_col"
        if len(name) > 63:
             logger.warning(f"Column name '{name[:10]}...' truncated to 63 characters.")
             name = name[:63]
        return name

    def _infer_column_type(self, column: pd.Series) -> Tuple[type, int, int]:
        non_null = column.dropna()
        if len(non_null) == 0: return String, 255, 0

        # --- Numeric Check ---
        try:
            numeric_col = pd.to_numeric(non_null, errors='raise')
            if pd.api.types.is_integer_dtype(numeric_col): return Integer, 0, 0
            elif pd.api.types.is_float_dtype(numeric_col):
                 max_val = numeric_col.abs().max()
                 if pd.isna(max_val) or max_val == 0: precision, scale = 10, 2
                 else:
                    # Use try-except for potentially non-numeric values after dropna
                    try: int_digits = non_null.apply(lambda x: len(str(int(x))) if pd.notna(x) and x != 0 else 0).max()
                    except (ValueError, TypeError): int_digits = 5 # Fallback int digits
                    try: decimal_places = non_null.apply(lambda x: len(str(x).split('.')[-1]) if pd.notna(x) and '.' in str(x) else 0).max()
                    except (ValueError, TypeError): decimal_places = 2 # Fallback decimal places

                    precision = min(int(int_digits + decimal_places), 15)
                    scale = min(int(decimal_places), 10)
                 return Float, precision, scale
        except (ValueError, TypeError): pass

        # --- Boolean Check ---
        if pd.api.types.is_bool_dtype(non_null) or non_null.astype(str).str.lower().isin(['true', 'false', '1', '0', 'yes', 'no', 't', 'f']).all():
             return Boolean, 0, 0

        # --- Datetime Check (with dayfirst=True attempt) ---
        try:
            # Try parsing with dayfirst=True first, as it's common outside US
            datetime_col_dayfirst = pd.to_datetime(non_null, errors='coerce', dayfirst=True)
            if not datetime_col_dayfirst.isnull().all(): # If successful with dayfirst=True
                logger.debug(f"Column '{column.name}' detected as datetime with dayfirst=True.")
                if (datetime_col_dayfirst.dt.floor('d') == datetime_col_dayfirst).all():
                    return Date, 0, 0
                else:
                    return DateTime, 0, 0
            else:
                 # Fallback to default parsing if dayfirst=True failed
                 datetime_col_default = pd.to_datetime(non_null, errors='coerce')
                 if not datetime_col_default.isnull().all():
                    logger.debug(f"Column '{column.name}' detected as datetime with default parsing.")
                    if (datetime_col_default.dt.floor('d') == datetime_col_default).all():
                        return Date, 0, 0
                    else:
                        return DateTime, 0, 0
        except (ValueError, TypeError, OverflowError): pass

        # --- Default to String ---
        max_len = 1
        if non_null.dtype == object or pd.api.types.is_string_dtype(non_null):
            max_len = non_null.astype(str).str.len().max()
            if pd.isna(max_len): max_len = 255
        max_len = max(1, max_len if pd.notna(max_len) else 0)
        # Return length as int
        return String, int(min(max_len * 1.5, 2048)), 0


    def import_excel(
        self, file_input: Union[str, io.BytesIO],
        sheet_name: Optional[Union[str, int, List[Union[str, int]]]] = 0,
        table_names: Optional[Dict[Union[str, int], str]] = None,
        primary_keys: Optional[Dict[str, str]] = None,
        relationships: Optional[List[Dict[str, str]]] = None,
        if_exists: str = 'replace'
    ) -> List[str]:
        if not self.engine: raise ValueError("Database connection not established.")
        processed_tables = []
        files_to_process = []
        if isinstance(file_input, str):
            files_to_process = glob.glob(file_input) if '*' in file_input or '?' in file_input else [file_input]
            if not files_to_process: logger.warning(f"No files found matching pattern: {file_input}"); return []
        elif isinstance(file_input, io.BytesIO): files_to_process = [file_input]; logger.info("Processing Excel stream.")
        else: raise TypeError("file_input must be a string (path/glob) or BytesIO stream.")

        for file_item in files_to_process:
            file_description = f"'{file_item}'" if isinstance(file_item, str) else "stream"
            if isinstance(file_item, str) and not os.path.exists(file_item): logger.warning(f"File not found: {file_item}. Skipping."); continue
            logger.info(f"Processing Excel input: {file_description}")
            try:
                excel_data = pd.read_excel(file_item, sheet_name=sheet_name, engine='openpyxl')
                if not isinstance(excel_data, dict):
                    try: actual_sheet_name = pd.ExcelFile(file_item).sheet_names[0] if sheet_name == 0 else sheet_name
                    except Exception: actual_sheet_name = "sheet_1"
                    excel_data = {actual_sheet_name: excel_data}
                for current_sheet_name, df in excel_data.items():
                    if table_names and current_sheet_name in table_names: table_name = self._clean_column_name(table_names[current_sheet_name])
                    else: table_name = self._clean_column_name(str(current_sheet_name))
                    pk_col_original = primary_keys.get(table_name) if primary_keys else None
                    self._import_dataframe(df, table_name, primary_key=pk_col_original, if_exists=if_exists)
                    processed_tables.append(table_name)
                    if self.verbose: logger.info(f"Imported sheet '{current_sheet_name}' from {file_description} as table '{table_name}' (if_exists='{if_exists}')")
            except FileNotFoundError: logger.error(f"Excel file not found during processing: {file_description}")
            except Exception as e: logger.error(f"Error importing Excel input {file_description}: {e}", exc_info=True); continue
        if relationships: logger.warning("Relationship creation is experimental."); self._create_relationships(relationships)
        return list(set(processed_tables))

    def import_csv(
        self, file_path: str, table_name: Optional[str] = None, delimiter: str = ',',
        encoding: str = 'utf-8', primary_key: Optional[str] = None,
        relationships: Optional[List[Dict[str, str]]] = None, if_exists: str = 'replace'
    ) -> List[str]:
        if not self.engine: raise ValueError("Database connection not established.")
        files = glob.glob(file_path) if '*' in file_path or '?' in file_path else [file_path]
        if not files: logger.warning(f"No files found matching pattern: {file_path}"); return []
        processed_tables = []; table_name_counts = {}
        for file in files:
            if not os.path.exists(file): logger.warning(f"File not found: {file}. Skipping."); continue
            logger.info(f"Processing CSV file path: {file}")
            try:
                base_table_name = table_name if table_name else Path(file).stem
                current_table_name = self._clean_column_name(base_table_name)
                if len(files) > 1 and not table_name:
                    if current_table_name in table_name_counts:
                        table_name_counts[current_table_name] += 1
                        current_table_name = f"{current_table_name}_{table_name_counts[current_table_name]}"
                    else: table_name_counts[current_table_name] = 0
                df = pd.read_csv(file, delimiter=delimiter, encoding=encoding, low_memory=False)
                self._import_dataframe(df, current_table_name, primary_key=primary_key, if_exists=if_exists)
                processed_tables.append(current_table_name)
                if self.verbose: logger.info(f"Imported CSV file '{file}' as table '{current_table_name}' (if_exists='{if_exists}')")
            except FileNotFoundError: logger.error(f"CSV file not found during processing: {file}")
            except pd.errors.EmptyDataError: logger.warning(f"CSV file '{file}' is empty. Skipping.")
            except Exception as e: logger.error(f"Error importing CSV file {file}: {e}", exc_info=True); continue
        if relationships: logger.warning("Relationship creation is experimental."); self._create_relationships(relationships)
        return list(set(processed_tables))

    def import_csv_from_stream(
        self, stream: io.BytesIO, table_name: str, delimiter: str = ',',
        encoding: str = 'utf-8', primary_key: Optional[str] = None, if_exists: str = 'replace'
    ) -> List[str]:
        if not self.engine: raise ValueError("Database connection not established.")
        if not table_name: raise ValueError("Table name must be provided when importing from a stream.")
        logger.info(f"Processing CSV stream into table '{table_name}'")
        try:
            stream.seek(0)
            df = pd.read_csv(stream, delimiter=delimiter, encoding=encoding, low_memory=False)
            self._import_dataframe(df, table_name, primary_key=primary_key, if_exists=if_exists)
            if self.verbose: logger.info(f"Imported CSV stream as table '{table_name}' (if_exists='{if_exists}')")
            return [table_name]
        except pd.errors.EmptyDataError: logger.warning(f"CSV stream for table '{table_name}' is empty. Skipping."); return []
        except Exception as e: logger.error(f"Error importing CSV stream into table '{table_name}': {e}", exc_info=True); raise

    def _import_dataframe(
        self, df: pd.DataFrame, table_name: str,
        primary_key: Optional[str] = None, if_exists: str = 'replace'
    ) -> None:
        if df.empty: logger.warning(f"DataFrame for table '{table_name}' is empty. Skipping import."); return
        if not self.engine: logger.error(f"Cannot import DataFrame for '{table_name}': Engine not initialized."); return

        original_columns = df.columns.tolist()
        df.columns = [self._clean_column_name(col) for col in original_columns]
        cleaned_column_map = dict(zip(original_columns, df.columns))
        cleaned_pk = None
        if primary_key:
            cleaned_pk = next((cleaned for orig, cleaned in cleaned_column_map.items() if orig == primary_key), None)
            if not cleaned_pk: logger.warning(f"Original PK '{primary_key}' not found. Ignoring PK for '{table_name}'.")
            elif cleaned_pk not in df.columns: logger.error(f"Cleaned PK '{cleaned_pk}' not in final columns for '{table_name}'. Ignoring PK."); cleaned_pk = None
            else:
                 if df[cleaned_pk].duplicated().any(): logger.warning(f"PK column '{cleaned_pk}' has duplicates in '{table_name}'. Ignoring PK."); cleaned_pk = None
                 elif df[cleaned_pk].isnull().any(): logger.warning(f"PK column '{cleaned_pk}' has NULLs in '{table_name}'. Ignoring PK."); cleaned_pk = None

        sqlalchemy_columns = []; dtype_mapping = {}
        for col_name in df.columns:
            sql_type_cls, precision, scale = self._infer_column_type(df[col_name])
            col_args = {}; current_sql_type_instance = None
            if col_name == cleaned_pk: col_args["primary_key"] = True; col_args["nullable"] = False
            if sql_type_cls is String and precision: current_sql_type_instance = sql_type_cls(precision)
            elif sql_type_cls is Float and precision and scale: current_sql_type_instance = sql_type_cls(precision=precision, decimal_return_scale=scale)
            elif sql_type_cls is DateTime: current_sql_type_instance = sql_type_cls(timezone=False)
            else: current_sql_type_instance = sql_type_cls()
            sqlalchemy_columns.append(Column(col_name, current_sql_type_instance, **col_args))
            dtype_mapping[col_name] = current_sql_type_instance

        local_metadata = MetaData()
        table = Table(table_name, local_metadata, *sqlalchemy_columns)
        try:
            insp = inspect(self.engine); table_exists = insp.has_table(table_name)
            if table_exists:
                if if_exists == 'fail': raise ValueError(f"Table '{table_name}' exists and if_exists='fail'.")
                elif if_exists == 'replace':
                    logger.info(f"Dropping existing table '{table_name}' (if_exists='replace').")
                    existing_table_meta = MetaData()
                    existing_table = Table(table_name, existing_table_meta, autoload_with=self.engine)
                    existing_table.drop(self.engine); logger.info(f"Dropped existing table '{table_name}'.")
                    table_exists = False
                elif if_exists == 'append': logger.info(f"Appending data to existing table '{table_name}'.")
                else: raise ValueError(f"Invalid value for if_exists: '{if_exists}'")
            if not table_exists:
                local_metadata.create_all(self.engine, tables=[table])
                if self.verbose: logger.info(f"Created table '{table_name}' structure.")

            if self.verbose: logger.info(f"Preparing to insert {len(df)} rows into '{table_name}'...")
            df_prepared = df.copy()

            # --- DataFrame Preparation (with explicit dayfirst=True) ---
            for col_name, sql_type in dtype_mapping.items():
                if col_name not in df_prepared: continue
                col_dtype = df_prepared[col_name].dtype
                needs_datetime_conversion = isinstance(sql_type, (DateTime, Date)) and not pd.api.types.is_datetime64_any_dtype(col_dtype)
                needs_bool_conversion = isinstance(sql_type, Boolean) and not pd.api.types.is_bool_dtype(col_dtype)

                if needs_datetime_conversion:
                    df_prepared[col_name] = pd.to_datetime(df_prepared[col_name], errors='coerce', dayfirst=True)
                    if pd.api.types.is_datetime64_any_dtype(df_prepared[col_name].dtype):
                        if isinstance(sql_type, Date):
                            df_prepared[col_name] = df_prepared[col_name].dt.date
                    else:
                         logger.warning(f"Could not reliably convert column '{col_name}' to datetime/date for table '{table_name}'.")

                elif needs_bool_conversion:
                    if pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_string_dtype(col_dtype):
                         bool_map = {'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False, 't': True, 'f': False}
                         lowered_series = df_prepared[col_name].astype(str).str.lower().str.strip()
                         df_prepared[col_name] = lowered_series.map(bool_map).combine_first(df_prepared[col_name])
                    elif pd.api.types.is_numeric_dtype(col_dtype):
                         df_prepared[col_name] = df_prepared[col_name].map({1: True, 0: False}).combine_first(df_prepared[col_name])
                    try:
                        df_prepared[col_name] = pd.to_numeric(df_prepared[col_name], errors='coerce')
                        df_prepared[col_name] = df_prepared[col_name].astype('boolean')
                    except (TypeError, ValueError) as bool_err:
                         logger.warning(f"Could not fully convert column '{col_name}' to Boolean for '{table_name}': {bool_err}")

            df_prepared = df_prepared.astype(object).replace({pd.NA: None, np.nan: None, pd.NaT: None})
            # --- End DataFrame Preparation ---

            try:
                df_prepared.to_sql(
                     name=table_name, con=self.engine, if_exists='append', index=False,
                     chunksize=500, dtype=dtype_mapping, method='multi'
                 )
            except Exception as insert_error:
                 logger.error(f"Error during pandas.to_sql insertion into '{table_name}': {insert_error}", exc_info=True)
                 raise insert_error

            if self.verbose:
                action_verb = "Appended data to" if if_exists == 'append' and table_exists else "Inserted data into"
                logger.info(f"Successfully {action_verb} '{table_name}'")

        except Exception as e:
            logger.error(f"Error during table creation or data insertion for '{table_name}': {e}", exc_info=True)
            raise

    def _create_relationships(self, relationships: List[Dict[str, str]]) -> None:
        logger.warning("Relationship creation via PRAGMA is experimental and limited.")
        temp_conn = None
        try:
            temp_conn = sqlite3.connect(self.db_path)
            temp_conn.execute("PRAGMA foreign_keys = ON;")
            for rel in relationships:
                try:
                    parent_table = rel.get('parent_table'); parent_column = rel.get('parent_column')
                    child_table = rel.get('child_table'); child_column = rel.get('child_column')
                    if not all([parent_table, parent_column, child_table, child_column]): logger.warning(f"Incomplete relationship: {rel}. Skipping."); continue
                    logger.warning(f"SQLite limitations prevent adding FK constraint directly for {child_table}.{child_column} -> {parent_table}.{parent_column}.")
                except Exception as e: logger.error(f"Error processing relationship {rel}: {e}", exc_info=True)
        except sqlite3.Error as e: logger.error(f"SQLite error during relationship processing: {e}")
        finally:
            if temp_conn: temp_conn.close()

    def get_connection_string(self) -> str:
        if not self.db_path: raise ValueError("No database connected or path specified")
        abs_db_path = os.path.abspath(self.db_path)
        return f"sqlite:///{abs_db_path}"

    def get_tables_info(self) -> Dict[str, Dict]:
        if not self.engine: raise ValueError("Database engine not initialized.")
        try:
            insp = inspect(self.engine); tables_info = {}; table_names = insp.get_table_names()
            for table_name in table_names:
                 if table_name.startswith('sqlite_'): continue
                 columns = []; foreign_keys = []; primary_keys = []; row_count = -1; sample_data = []
                 try:
                     cols_data = insp.get_columns(table_name)
                     for col in cols_data: columns.append({"name": col["name"], "type": str(col["type"]), "nullable": col.get("nullable", True), "default": col.get("default", None), "pk": col.get("primary_key", False)})
                 except Exception as e: logger.error(f"Could not get columns for {table_name}: {e}"); columns = [{"error": str(e)}]
                 try:
                     fks_data = insp.get_foreign_keys(table_name)
                     for fk in fks_data: foreign_keys.append({"constrained_columns": fk["constrained_columns"], "referred_table": fk["referred_table"], "referred_columns": fk["referred_columns"]})
                 except Exception as e: logger.error(f"Could not get FKs for {table_name}: {e}"); foreign_keys = [{"error": str(e)}]
                 try: pk_constraint = insp.get_pk_constraint(table_name); primary_keys = pk_constraint.get("constrained_columns", [])
                 except Exception as e: logger.error(f"Could not get PK for {table_name}: {e}"); primary_keys = ["error"]
                 try:
                     with self.engine.connect() as connection: result = connection.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')); row_count = result.scalar_one_or_none(); row_count = row_count if row_count is not None else 0
                 except Exception as e: logger.error(f"Could not get row count for {table_name}: {e}")
                 try: sample_df = pd.read_sql(f'SELECT * FROM "{table_name}" LIMIT 5', self.engine); sample_data = sample_df.astype(object).where(pd.notnull(sample_df), None).to_dict('records')
                 except Exception as e: logger.error(f"Could not get sample data for {table_name}: {e}"); sample_data = [{"error": f"Could not fetch: {e}"}]
                 tables_info[table_name] = {"columns": columns, "foreign_keys": foreign_keys, "primary_keys": primary_keys, "row_count": row_count, "sample_data": sample_data}
            return tables_info
        except Exception as inspect_e: logger.error(f"Failed to inspect DB schema: {inspect_e}", exc_info=True); raise RuntimeError(f"Failed to inspect DB schema: {inspect_e}") from inspect_e

    def execute_query(self, query: str) -> Union[pd.DataFrame, List[Dict]]:
        if not self.engine: raise ValueError("Database engine not initialized.")
        try:
            if self.verbose: logger.info(f"Executing query: {query}")
            if re.match(r'^\s*(SELECT|WITH)\b', query, re.IGNORECASE): df = pd.read_sql_query(query, self.engine); return df.astype(object).where(pd.notnull(df), None)
            else:
                logger.info(f"Executing non-SELECT query: {query}")
                with self.engine.begin() as connection: result_proxy = connection.execute(text(query)); rowcount = result_proxy.rowcount
                logger.info(f"Non-SELECT query executed. Rows affected: {rowcount}")
                return pd.DataFrame({"status": ["Execution successful"], "rows_affected": [rowcount]})
        except Exception as pd_e:
             logger.warning(f"Pandas read_sql failed: {pd_e}. Falling back.")
             try:
                 with self.engine.connect() as connection: cursor_result = connection.execute(text(query))
                 if cursor_result.returns_rows: columns = list(cursor_result.keys()); results = [dict(zip(columns, row)) for row in cursor_result.fetchall()]; return results
                 else: rowcount = cursor_result.rowcount; logger.info(f"Basic execution success. Rows affected: {rowcount}"); return pd.DataFrame({"status": ["Execution successful"], "rows_affected": [rowcount]})
             except Exception as e: logger.error(f"Query failed on both pandas and basic: {e}", exc_info=True); raise RuntimeError(f"Query execution failed: {e}") from e

    def close(self):
        if self.engine:
             try: self.engine.dispose(); self.engine = None; logger.info("SQLAlchemy engine disposed.")
             except Exception as e: logger.error(f"Error disposing SQLAlchemy engine: {e}")

# --- main function for CLI ---
def main():
    # *** ARGPARSE SETUP MOVED INSIDE MAIN ***
    parser = argparse.ArgumentParser(description='Import files into SQLite using DataImporter_Gemini')
    parser.add_argument('--db', required=True, help='Path to SQLite database file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    subparsers = parser.add_subparsers(dest='command', help='Command', required=True)
    base_import_parser = argparse.ArgumentParser(add_help=False)
    base_import_parser.add_argument('--file', required=True, help='Path/glob pattern')
    base_import_parser.add_argument('--table', help='Base table name')
    base_import_parser.add_argument('--pk', help='Primary key column name')
    base_import_parser.add_argument('--if-exists', default='replace', choices=['fail', 'replace', 'append'])
    excel_parser = subparsers.add_parser('excel', help='Import Excel', parents=[base_import_parser])
    excel_parser.add_argument('--sheet', help='Sheet name/index/None')
    csv_parser = subparsers.add_parser('csv', help='Import CSV', parents=[base_import_parser])
    csv_parser.add_argument('--delimiter', default=',')
    csv_parser.add_argument('--encoding', default='utf-8')
    json_parser = subparsers.add_parser('json', help='Import JSON', parents=[base_import_parser])
    json_parser.add_argument('--orient', default='records')
    json_parser.add_argument('--lines', action='store_true')
    info_parser = subparsers.add_parser('info', help='Display DB info')
    query_parser = subparsers.add_parser('query', help='Execute SQL query')
    query_parser.add_argument('sql', help='SQL query string')
    # *** END ARGPARSE SETUP ***

    args = parser.parse_args() # Now parse args inside main

    importer = None
    try:
        importer = DataImporter_Gemini(db_path=args.db, verbose=args.verbose)
        if args.command == 'excel':
            sheet_name_param = 0; table_names_map = None; primary_keys_map = None
            if args.sheet == "None": sheet_name_param = None
            elif args.sheet:
                 items = [s.strip() for s in args.sheet.split(',')]
                 parsed_items = []
                 for item in items:
                      try: parsed_items.append(int(item))
                      except ValueError: parsed_items.append(item)
                 sheet_name_param = parsed_items[0] if len(parsed_items) == 1 else parsed_items
            if args.table and isinstance(sheet_name_param, (str, int)):
                 clean_table_name = importer._clean_column_name(args.table); table_names_map = {sheet_name_param: clean_table_name}
                 if args.pk: primary_keys_map = {clean_table_name: args.pk}
            # --- COMMENT OUT THE PROBLEMATIC ELIF ---
            # elif args.table and sheet_name_param is None:
            #      logger.warning("--table ignored for all sheets.")
            #      if args.pk: logger.warning("--pk ignored for all sheets.")
            # --- END COMMENT OUT ---
            tables = importer.import_excel(file_input=args.file, sheet_name=sheet_name_param, table_names=table_names_map, primary_keys=primary_keys_map, if_exists=args.if_exists)
            print(f"Imported Excel tables: {', '.join(tables)}")
        elif args.command == 'csv':
            tables = importer.import_csv(file_path=args.file, table_name=args.table, delimiter=args.delimiter, encoding=args.encoding, primary_key=args.pk, if_exists=args.if_exists)
            print(f"Imported CSV tables: {', '.join(tables)}")
        elif args.command == 'json':
             try:
                 json_files = glob.glob(args.file) if '*' in args.file or '?' in args.file else [args.file]; imported_json_tables = []
                 if not json_files: print(f"No JSON files found: {args.file}")
                 else:
                      for json_file in json_files:
                          if not os.path.exists(json_file): logger.warning(f"JSON file not found: {json_file}. Skipping."); continue
                          logger.info(f"Processing JSON file: {json_file}")
                          df = pd.read_json(json_file, orient=args.orient, lines=args.lines)
                          table_name = args.table if args.table and len(json_files) == 1 else Path(json_file).stem
                          table_name = importer._clean_column_name(table_name)
                          importer._import_dataframe(df, table_name, primary_key=args.pk, if_exists=args.if_exists)
                          imported_json_tables.append(table_name)
                          print(f"Imported JSON '{json_file}' to table: {table_name}")
                      print(f"Completed JSON import. Tables: {', '.join(imported_json_tables)}")
             except Exception as e: print(f"Error importing JSON {args.file}: {e}")
        elif args.command == 'info':
            tables_info = importer.get_tables_info(); print("--- DB Schema Info ---"); print(f"DB: {args.db}"); print(f"Tables Found: {len(tables_info)}"); print("-" * 30)
            for table_name, info in tables_info.items():
                print(f"\nTable: {table_name}"); print(f"  Rows: {info.get('row_count', 'N/A')}"); print(f"  Columns: ({len(info.get('columns', []))})")
                for col in info.get('columns', []): pk_flag = " (PK)" if col.get('pk') else ""; nullable_flag = " NULL" if col.get('nullable') else " NOT NULL"; default_val = f" DEFAULT {col.get('default')}" if col.get('default') else ""; print(f"    - {col.get('name', '?')} ({col.get('type', '?')}){pk_flag}{nullable_flag}{default_val}")
                pks = info.get('primary_keys', []); fks = info.get('foreign_keys', [])
                if pks and isinstance(pks, list) and not any("error" in str(k) for k in pks): print(f"  PK Constraint: ({', '.join(pks)})")
                if fks and isinstance(fks, list) and not any("error" in str(k) for k in fks):
                    print(f"  FKs: ({len(fks)})");
                    for fk in fks: cols = ', '.join(fk.get('constrained_columns', ['?'])); ref_table = fk.get('referred_table', '?'); ref_cols = ', '.join(fk.get('referred_columns', ['?'])); print(f"    - ({cols}) -> {ref_table}({ref_cols})")
                elif not fks: print("  FKs: None"); else: print(f"  FKs: {fks}")
                print("  Sample Data (first 5 rows):"); sample_data = info.get('sample_data', [])
                if sample_data and isinstance(sample_data, list) and not any("error" in str(s) for s in sample_data):
                     try: sample_df = pd.DataFrame(sample_data); print(sample_df.to_string(index=False, max_rows=5, max_cols=10));
                     except Exception as display_e: print(f"    Could not display sample data: {display_e}")
                elif sample_data and any("error" in str(s) for s in sample_data): print(f"    Error retrieving sample data: {sample_data[0]}")
                else: print("    (No sample data or table empty)")
                print("-" * 20)
        elif args.command == 'query':
             print(f"Executing SQL: {args.sql}")
             try:
                  results = importer.execute_query(args.sql); print("\nQuery Results:")
                  if isinstance(results, pd.DataFrame):
                       if "status" in results.columns: print(f"Status: {results['status'].iloc[0]}, Rows affected: {results['rows_affected'].iloc[0]}")
                       elif results.empty: print("(Query returned no rows)")
                       else: print(results.to_string(index=False))
                  elif isinstance(results, list):
                       if results: import json; print(json.dumps(results, indent=2, default=str));
                       else: print("(Query executed successfully, no rows returned)")
                  else: print("(No results or unexpected format)")
             except Exception as e: print(f"\nError executing query: {e}")
        else: parser.print_help()
    except Exception as e: print(f"\nAn error occurred: {e}", file=sys.stderr); logger.error("Unhandled exception", exc_info=True); sys.exit(1)
    finally:
        if importer: importer.close()

# Guard the execution of main
if __name__ == "__main__":
    main()
# --- END OF FILE DataImporter_Gemini.py ---
