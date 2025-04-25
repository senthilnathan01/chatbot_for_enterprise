# --- START OF FILE DataImporter_Gemini.py ---

import os
import pandas as pd
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
import re
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, MetaData, ForeignKey, Boolean, DateTime, Date, inspect, text # Added text here
# Removed: from sqlalchemy.ext.declarative import declarative_base - Not used
import datetime
import argparse
import glob
from pathlib import Path
import sys # For stderr in main
import io # Import io for BytesIO type hint and usage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Logger name will be DataImporter_Gemini if run directly

class DataImporter_Gemini:
    """
    A tool for importing data from Excel, CSV or JSON files into SQL databases.
    Can be used with VannaReplica_Gemini to analyze the imported data.
    """

    def __init__(self, db_path: str = None, verbose: bool = False):
        """
        Initialize the DataImporter_Gemini.

        Args:
            db_path: Path to the SQLite database file to create or use
            verbose: Whether to print verbose output
        """
        self.db_path = db_path
        self.verbose = verbose
        self.conn = None
        self.engine = None
        self.metadata = MetaData()
        # Removed: self.base = declarative_base() - Not used

        if db_path:
            self.connect(db_path)

    def connect(self, db_path: str) -> None:
        """
        Connect to a SQLite database.

        Args:
            db_path: Path to the SQLite database file
        """
        try:
            # Ensure parent directory exists
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                logger.info(f"Created directory: {db_dir}")

            sqlite_uri = f"sqlite:///{db_path}"
            self.engine = create_engine(sqlite_uri)
            # Use check_same_thread=False only if necessary and understood.
            # For CLI usage, it's often not needed. Let's remove it for now.
            # Reconnect using sqlite3 only if needed for PRAGMA or direct cursor ops.
            # self.conn = sqlite3.connect(db_path) # Removed check_same_thread=False
            self.db_path = db_path

            # Test connection using SQLAlchemy engine
            with self.engine.connect() as connection:
                logger.debug("SQLAlchemy engine connection test successful.")


            if self.verbose:
                logger.info(f"Connected to database engine for {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _clean_column_name(self, name: str) -> str:
        """
        Clean column names to be SQL-friendly.

        Args:
            name: Original column name

        Returns:
            SQL-friendly column name
        """
        if not isinstance(name, str):
            name = str(name)

        # Replace spaces and special characters with underscores
        name = re.sub(r'[^\w\s]', '_', name) # Keep alphanumeric and underscore
        name = re.sub(r'\s+', '_', name) # Replace whitespace with underscore

        # Remove leading/trailing underscores
        name = name.strip('_')

        # Ensure name starts with a letter or underscore
        if name and name[0].isdigit():
            name = f"col_{name}"

        # Lowercase for consistency
        name = name.lower()

        # Ensure name is not empty
        if not name:
            name = "unnamed_col"

        # Ensure name is not too long for SQLite (less critical but good practice)
        if len(name) > 63:
             logger.warning(f"Column name '{name[:10]}...' truncated to 63 characters.")
             name = name[:63]

        return name

    def _infer_column_type(self, column: pd.Series) -> Tuple[type, int, int]:
        """
        Infer the appropriate SQL data type for a pandas Series.

        Args:
            column: Pandas Series to analyze

        Returns:
            Tuple of (SQLAlchemy type, precision, scale)
        """
        # Remove NaN values for type detection
        non_null = column.dropna()

        if len(non_null) == 0:
            return String, 255, 0 # Default to String if all null

        # Attempt numeric conversion first
        try:
            numeric_col = pd.to_numeric(non_null, errors='raise')
            if pd.api.types.is_integer_dtype(numeric_col):
                 # Check range for potential INTEGER overflow if needed (SQLite handles large integers)
                 return Integer, 0, 0
            elif pd.api.types.is_float_dtype(numeric_col):
                 # More robust Float detection
                 max_val = numeric_col.abs().max()
                 if pd.isna(max_val) or max_val == 0:
                     precision, scale = 10, 2
                 else:
                    # Estimate precision and scale based on actual data
                    int_digits = non_null.apply(lambda x: len(str(int(x))) if pd.notna(x) and x != 0 else 0).max()
                    decimal_places = non_null.apply(lambda x: len(str(x).split('.')[-1]) if pd.notna(x) and '.' in str(x) else 0).max()
                    precision = min(int_digits + decimal_places, 15) # Adjust precision cap if needed
                    scale = min(decimal_places, 10) # Adjust scale cap if needed
                 return Float, precision, scale
        except (ValueError, TypeError):
            # Not purely numeric, continue checks
            pass

        # Check if all values are boolean or can be interpreted as boolean
        if pd.api.types.is_bool_dtype(non_null) or non_null.astype(str).str.lower().isin(['true', 'false', '1', '0', 'yes', 'no', 't', 'f']).all():
             return Boolean, 0, 0

        # Check if values can be parsed as dates/datetimes
        try:
            # Attempt to convert to datetime, handling mixed types
            datetime_col = pd.to_datetime(non_null, errors='coerce')
            if not datetime_col.isnull().all(): # Check if at least one value converted successfully
                # Check if they are all dates (no time component)
                if (datetime_col.dt.floor('d') == datetime_col).all():
                    return Date, 0, 0
                else:
                    return DateTime, 0, 0
        except (ValueError, TypeError, OverflowError):
             pass # Continue if date conversion fails

        # Default to string
        max_len = 1 # Default minimum length
        if non_null.dtype == object or pd.api.types.is_string_dtype(non_null):
            # Calculate max string length more accurately
            max_len = non_null.astype(str).str.len().max()
            if pd.isna(max_len): max_len = 255 # Fallback if max_len calculation fails
        # Ensure max_len is at least 1 even if the only non-null value is empty string
        max_len = max(1, max_len if pd.notna(max_len) else 0)
        # Increase safety margin and max length for SQLite's dynamic typing flexibility
        return String, int(min(max_len * 1.5, 2048)), 0


    def import_excel(
        self,
        file_input: Union[str, io.BytesIO], # Can be path or stream
        sheet_name: Optional[Union[str, int, List[Union[str, int]]]] = 0, # Default to first sheet
        table_names: Optional[Dict[Union[str, int], str]] = None,
        primary_keys: Optional[Dict[str, str]] = None,
        relationships: Optional[List[Dict[str, str]]] = None, # Note: Relationship creation needs careful handling
        if_exists: str = 'replace' # *** ADDED if_exists parameter ***
    ) -> List[str]:
        """
        Import data from Excel file path or stream into SQLite database.

        Args:
            file_input: Path to Excel file, glob pattern, or BytesIO stream.
            sheet_name: Sheet name(s) or index(es) to import. 0 for first sheet, None for all sheets.
            table_names: Dictionary mapping sheet names/indexes to table names. Overrides default naming.
            primary_keys: Dictionary mapping table names to primary key column names
            relationships: List of relationship dictionaries (experimental, requires existing tables)
            if_exists: Action if table exists ('fail', 'replace', 'append'). Default 'replace'.

        Returns:
            List of table names created or updated
        """
        if not self.engine:
            raise ValueError("Database connection not established. Call connect() first.")

        processed_tables = []
        files_to_process = []

        if isinstance(file_input, str):
            # Handle path/glob input
            files_to_process = glob.glob(file_input) if '*' in file_input or '?' in file_input else [file_input]
            if not files_to_process:
                logger.warning(f"No files found matching pattern: {file_input}")
                return []
        elif isinstance(file_input, io.BytesIO):
            # Handle stream input (treat as single "file")
            files_to_process = [file_input]
            logger.info("Processing Excel stream.")
        else:
            raise TypeError("file_input must be a string (path/glob) or BytesIO stream.")


        for file_item in files_to_process:
            file_description = f"'{file_item}'" if isinstance(file_item, str) else "stream"

            if isinstance(file_item, str) and not os.path.exists(file_item):
                logger.warning(f"File not found: {file_item}. Skipping.")
                continue

            logger.info(f"Processing Excel input: {file_description}")

            try:
                # Read sheets using the file path or stream
                excel_data = pd.read_excel(file_item, sheet_name=sheet_name, engine='openpyxl')

                # Standardize excel_data to dict format
                if not isinstance(excel_data, dict):
                    # Get actual name of first sheet if index 0 used, or handle single sheet case
                    try:
                        actual_sheet_name = pd.ExcelFile(file_item).sheet_names[0] if sheet_name == 0 else sheet_name
                    except Exception:
                        actual_sheet_name = "sheet_1" # Fallback name
                    excel_data = {actual_sheet_name: excel_data}

                # Process each sheet
                for current_sheet_name, df in excel_data.items():
                    # Determine table name
                    # Use cleaned version of mapped name or cleaned sheet name
                    if table_names and current_sheet_name in table_names:
                        table_name = self._clean_column_name(table_names[current_sheet_name])
                    else:
                        table_name = self._clean_column_name(str(current_sheet_name))

                    # Get primary key for this specific table (use the cleaned table name for lookup)
                    pk_col_original = primary_keys.get(table_name) if primary_keys else None


                    # Import dataframe to SQL
                    self._import_dataframe(
                        df,
                        table_name,
                        primary_key=pk_col_original, # Pass original PK name
                        if_exists=if_exists         # *** Pass if_exists down ***
                    )

                    processed_tables.append(table_name)

                    if self.verbose:
                        logger.info(f"Imported sheet '{current_sheet_name}' from {file_description} as table '{table_name}' (if_exists='{if_exists}')")

            except FileNotFoundError:
                 logger.error(f"Excel file not found during processing: {file_description}")
            except Exception as e:
                logger.error(f"Error importing Excel input {file_description}: {e}", exc_info=True) # Log traceback
                continue # Log error and continue with next file if path/glob

        # Create relationships (Note: This simplistic approach might fail if tables need modification)
        if relationships:
            logger.warning("Relationship creation is experimental and assumes tables/columns exist.")
            self._create_relationships(relationships) # Consider moving this to a separate method or tool

        return list(set(processed_tables)) # Return unique table names

    def import_csv(
        self,
        file_path: str,
        table_name: Optional[str] = None,
        delimiter: str = ',',
        encoding: str = 'utf-8',
        primary_key: Optional[str] = None,
        relationships: Optional[List[Dict[str, str]]] = None,
        if_exists: str = 'replace'
    ) -> List[str]:
        """
        Import data from CSV file(s) into SQLite database. Uses file paths.

        Args:
            file_path: Path to CSV file or glob pattern.
            table_name: Base name for the imported table(s). If None, uses filename stem.
            delimiter: CSV delimiter character.
            encoding: File encoding.
            primary_key: Primary key column name (applied if table_name specified or for single file).
            relationships: List of relationship dictionaries (experimental).
            if_exists: Action if table exists ('fail', 'replace', 'append'). Default 'replace'.

        Returns:
            List of table names created or updated.
        """
        if not self.engine:
            raise ValueError("Database connection not established. Call connect() first.")

        # Expand glob pattern if provided
        files = glob.glob(file_path) if '*' in file_path or '?' in file_path else [file_path]

        if not files:
            logger.warning(f"No files found matching pattern: {file_path}")
            return []

        processed_tables = []
        table_name_counts = {} # Handle potential name collisions with globs

        for file in files:
            if not os.path.exists(file): # Check path existence
                logger.warning(f"File not found: {file}. Skipping.")
                continue

            logger.info(f"Processing CSV file path: {file}") # Log path

            try:
                # Determine table name
                base_table_name = table_name if table_name else Path(file).stem
                current_table_name = self._clean_column_name(base_table_name)

                # Handle potential name duplicates if using globs without explicit table names per file
                if len(files) > 1 and not table_name:
                    if current_table_name in table_name_counts:
                        table_name_counts[current_table_name] += 1
                        current_table_name = f"{current_table_name}_{table_name_counts[current_table_name]}"
                    else:
                         table_name_counts[current_table_name] = 0

                # Read CSV *from file path*
                df = pd.read_csv(file, delimiter=delimiter, encoding=encoding, low_memory=False)

                # Import dataframe to SQL
                self._import_dataframe(
                    df,
                    current_table_name,
                    primary_key=primary_key,
                    if_exists=if_exists
                )
                processed_tables.append(current_table_name)

                if self.verbose:
                    logger.info(f"Imported CSV file '{file}' as table '{current_table_name}' (if_exists='{if_exists}')")

            except FileNotFoundError:
                 logger.error(f"CSV file not found during processing: {file}")
            except pd.errors.EmptyDataError:
                logger.warning(f"CSV file '{file}' is empty. Skipping.")
            except Exception as e:
                logger.error(f"Error importing CSV file {file}: {e}", exc_info=True) # Log traceback
                continue

        # Create relationships (Note: Same warning as in import_excel)
        if relationships:
            logger.warning("Relationship creation is experimental and assumes tables/columns exist.")
            self._create_relationships(relationships)

        return list(set(processed_tables))

    # --- ADDED METHOD for streams ---
    def import_csv_from_stream(
        self,
        stream: io.BytesIO,
        table_name: str, # Require table name when using stream
        delimiter: str = ',',
        encoding: str = 'utf-8',
        primary_key: Optional[str] = None,
        if_exists: str = 'replace'
    ) -> List[str]:
        """
        Import data from a CSV file stream (e.g., BytesIO) into SQLite database.

        Args:
            stream: File-like object (BytesIO) containing CSV data.
            table_name: Name for the imported table (should already be cleaned).
            delimiter: CSV delimiter character.
            encoding: File encoding.
            primary_key: Primary key column name.
            if_exists: Action if table exists ('fail', 'replace', 'append').

        Returns:
            List containing the table name if successful, empty list otherwise.
        """
        if not self.engine:
            raise ValueError("Database connection not established. Call connect() first.")
        if not table_name:
             raise ValueError("Table name must be provided when importing from a stream.")

        logger.info(f"Processing CSV stream into table '{table_name}'")

        try:
            # Read CSV directly from the stream
            # Ensure the stream is reset if it has been read before
            stream.seek(0)
            df = pd.read_csv(stream, delimiter=delimiter, encoding=encoding, low_memory=False)

            # Import dataframe to SQL using the common internal method
            self._import_dataframe(
                df,
                table_name, # Use the provided (cleaned) table name
                primary_key=primary_key,
                if_exists=if_exists
            )

            if self.verbose:
                logger.info(f"Imported CSV stream as table '{table_name}' (if_exists='{if_exists}')")

            return [table_name] # Return list with the table name

        except pd.errors.EmptyDataError:
            logger.warning(f"CSV stream for table '{table_name}' is empty. Skipping.")
            return []
        except Exception as e:
            logger.error(f"Error importing CSV stream into table '{table_name}': {e}", exc_info=True)
            raise # Re-raise the exception to be caught by the caller

    def _import_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        primary_key: Optional[str] = None,
        if_exists: str = 'replace' # Default 'replace'
    ) -> None:
        """
        Import a pandas DataFrame into a SQLite table.

        Args:
            df: Pandas DataFrame to import
            table_name: Name for the table (already cleaned)
            primary_key: Name of the primary key column (original name before cleaning)
            if_exists: Action if table exists ('fail', 'replace', 'append'). Default 'replace'.
        """
        if df.empty:
            logger.warning(f"DataFrame for table '{table_name}' is empty. Skipping import.")
            return
        if not self.engine:
            logger.error(f"Cannot import DataFrame for '{table_name}': Database engine not initialized.")
            return

        # Clean column names first
        original_columns = df.columns.tolist()
        df.columns = [self._clean_column_name(col) for col in original_columns]
        cleaned_column_map = dict(zip(original_columns, df.columns))

        cleaned_pk = None
        if primary_key:
            # Find the cleaned version of the PK using the map
            cleaned_pk = next((cleaned for orig, cleaned in cleaned_column_map.items() if orig == primary_key), None)
            if not cleaned_pk:
                 logger.warning(f"Original primary key column '{primary_key}' not found in DataFrame's original columns. Ignoring PK constraint for table '{table_name}'.")
            elif cleaned_pk not in df.columns:
                 # This case should ideally not happen if the map is correct
                 logger.error(f"Consistency Error: Cleaned primary key column '{cleaned_pk}' (from '{primary_key}') not found in final DataFrame columns for table '{table_name}'. Ignoring PK constraint.")
                 cleaned_pk = None
            else:
                 # Verify primary key is unique *after* potential data type inference/cleaning
                 if df[cleaned_pk].duplicated().any():
                     logger.warning(f"Column '{cleaned_pk}' (cleaned from '{primary_key}') has duplicate values in table '{table_name}'. Cannot be used as primary key. Ignoring PK constraint.")
                     cleaned_pk = None
                 else:
                     # Ensure PK column does not contain NULLs
                     if df[cleaned_pk].isnull().any():
                         logger.warning(f"Primary key column '{cleaned_pk}' (cleaned from '{primary_key}') contains NULL values in table '{table_name}'. Ignoring PK constraint.")
                         cleaned_pk = None

        # Infer types and create SQLAlchemy Column definitions
        sqlalchemy_columns = []
        dtype_mapping = {} # For pandas.to_sql type mapping
        for col_name in df.columns:
            sql_type_cls, precision, scale = self._infer_column_type(df[col_name])
            col_args = {}
            if col_name == cleaned_pk:
                col_args["primary_key"] = True
                col_args["nullable"] = False # Primary keys cannot be null

            # Handle precision/scale for specific types
            current_sql_type_instance = None
            if sql_type_cls is String and precision:
                 current_sql_type_instance = sql_type_cls(precision)
            elif sql_type_cls is Float and precision and scale:
                # SQLite doesn't explicitly use precision/scale for FLOAT, but keep for mapping
                current_sql_type_instance = sql_type_cls(precision=precision, decimal_return_scale=scale)
            elif sql_type_cls is DateTime:
                 current_sql_type_instance = sql_type_cls(timezone=False) # Store naive datetimes
            else:
                 current_sql_type_instance = sql_type_cls()

            sqlalchemy_columns.append(Column(col_name, current_sql_type_instance, **col_args))
            dtype_mapping[col_name] = current_sql_type_instance


        # Create or replace table using SQLAlchemy metadata
        # Create a fresh metadata object for each table operation to avoid conflicts
        # This is simpler than managing reflections carefully across multiple calls.
        local_metadata = MetaData()
        table = Table(table_name, local_metadata, *sqlalchemy_columns)

        try:
            # Handle if_exists logic before creating table/inserting data
            insp = inspect(self.engine) # Use the main engine's inspector
            table_exists = insp.has_table(table_name)

            if table_exists:
                if if_exists == 'fail':
                    raise ValueError(f"Table '{table_name}' already exists and if_exists='fail'.")
                elif if_exists == 'replace':
                    logger.info(f"Table '{table_name}' exists and if_exists='replace'. Dropping existing table.")
                    # Reflect existing table to drop it correctly
                    existing_table_meta = MetaData()
                    existing_table = Table(table_name, existing_table_meta, autoload_with=self.engine)
                    existing_table.drop(self.engine)
                    logger.info(f"Dropped existing table '{table_name}'.")
                    table_exists = False # Now it doesn't exist for creation step below
                elif if_exists == 'append':
                    logger.info(f"Table '{table_name}' exists and if_exists='append'. Data will be appended.")
                    # We'll skip table creation but still insert data later
                else:
                    # Should not happen if choices are enforced earlier, but good failsafe
                    raise ValueError(f"Invalid value for if_exists: '{if_exists}'")

            # Create the table structure if it doesn't exist (or after dropping)
            if not table_exists:
                local_metadata.create_all(self.engine, tables=[table])
                if self.verbose:
                     logger.info(f"Created table '{table_name}' structure.")


            # --- Data Insertion using pandas.to_sql for better type handling ---
            if self.verbose:
                logger.info(f"Preparing to insert {len(df)} rows into '{table_name}'...")

            # Convert dtypes before insertion where necessary (especially dates/times/bools)
            df_prepared = df.copy()
            for col_name, sql_type in dtype_mapping.items():
                if col_name not in df_prepared: continue # Skip if column somehow missing

                col_dtype = df_prepared[col_name].dtype
                # Check if conversion is likely needed
                needs_datetime_conversion = isinstance(sql_type, (DateTime, Date)) and not pd.api.types.is_datetime64_any_dtype(col_dtype)
                needs_bool_conversion = isinstance(sql_type, Boolean) and not pd.api.types.is_bool_dtype(col_dtype)

                if needs_datetime_conversion:
                    # Convert to datetime objects, coercing errors to NaT
                    df_prepared[col_name] = pd.to_datetime(df_prepared[col_name], errors='coerce')
                    # For Date type, keep only the date part (apply only if conversion succeeded)
                    if isinstance(sql_type, Date) and pd.api.types.is_datetime64_any_dtype(df_prepared[col_name].dtype):
                        df_prepared[col_name] = df_prepared[col_name].dt.date
                elif needs_bool_conversion:
                    # Convert common boolean representations more robustly
                    if pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_string_dtype(col_dtype):
                         # Map common strings, keep original if no map
                         bool_map = {'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False, 't': True, 'f': False}
                         lowered_series = df_prepared[col_name].astype(str).str.lower().str.strip()
                         # Apply map, keep original if not in map (might become NaN -> None later)
                         df_prepared[col_name] = lowered_series.map(bool_map).combine_first(df_prepared[col_name])
                    # Attempt direct conversion for numeric types (1=True, 0=False)
                    elif pd.api.types.is_numeric_dtype(col_dtype):
                         df_prepared[col_name] = df_prepared[col_name].map({1: True, 0: False}).combine_first(df_prepared[col_name])
                    # Convert to pandas BooleanDtype AFTER mapping to handle None correctly
                    try:
                        # Coerce errors during final conversion
                        df_prepared[col_name] = pd.to_numeric(df_prepared[col_name], errors='coerce') # Try numeric first for 1/0
                        df_prepared[col_name] = df_prepared[col_name].astype('boolean')
                    except (TypeError, ValueError) as bool_err:
                         logger.warning(f"Could not fully convert column '{col_name}' to Boolean for table '{table_name}', may contain unmappable values. Error: {bool_err}")


            # Replace numpy NaN/NaT with None for SQL NULL *just before* insertion
            # This ensures calculations/conversions above worked correctly
            # Important: Use object dtype to allow None
            df_prepared = df_prepared.astype(object).replace({pd.NA: None, np.nan: None, pd.NaT: None})


            # Use pandas.to_sql for efficient insertion
            try:
                df_prepared.to_sql(
                     name=table_name,
                     con=self.engine,
                     if_exists='append', # Table structure handled above, always append data here
                     index=False, # Do not write pandas index as a column
                     chunksize=500, # Insert in chunks
                     dtype=dtype_mapping, # Provide explicit types to sqlalchemy
                     method='multi' # Often faster for SQLite with many rows
                 )
            except Exception as insert_error:
                 logger.error(f"Error during pandas.to_sql insertion into '{table_name}': {insert_error}", exc_info=True)
                 raise insert_error # Re-raise the original error


            if self.verbose:
                action_verb = "Appended data to" if if_exists == 'append' and table_exists else "Inserted data into"
                logger.info(f"Successfully {action_verb} '{table_name}'")

        except Exception as e:
            logger.error(f"Error during table creation or data insertion for '{table_name}': {e}", exc_info=True)
            raise # Re-raise the exception after logging


    def _create_relationships(self, relationships: List[Dict[str, str]]) -> None:
        """
        Create foreign key relationships between tables (EXPERIMENTAL).
        This basic version assumes tables and columns exist and doesn't modify them.
        It uses PRAGMA which might not be ideal or fully supported in all contexts.

        Args:
            relationships: List of relationship dictionaries with keys:
                          'parent_table', 'parent_column', 'child_table', 'child_column'
        """
        # Need a direct sqlite3 connection for PRAGMA manipulation if not using SQLAlchemy's schema tools
        temp_conn = None
        try:
            temp_conn = sqlite3.connect(self.db_path) # Connect directly for PRAGMA
            logger.warning("Attempting to add foreign key constraints using PRAGMA. This has limitations and might require manual schema adjustments.")

            # Ensure foreign keys are enabled
            temp_conn.execute("PRAGMA foreign_keys = ON;")


            for rel in relationships:
                try:
                    parent_table = rel.get('parent_table')
                    parent_column = rel.get('parent_column')
                    child_table = rel.get('child_table')
                    child_column = rel.get('child_column')

                    if not all([parent_table, parent_column, child_table, child_column]):
                        logger.warning(f"Incomplete relationship definition: {rel}. Skipping.")
                        continue

                    # Clean names (assuming they were cleaned during import)
                    parent_table_c = self._clean_column_name(parent_table)
                    parent_column_c = self._clean_column_name(parent_column)
                    child_table_c = self._clean_column_name(child_table)
                    child_column_c = self._clean_column_name(child_column)

                    # --- Limitation: Adding FK constraints to existing SQLite tables is complex ---
                    logger.warning(f"SQLite limitations prevent adding foreign key constraint directly for {child_table_c}.{child_column_c} -> {parent_table_c}.{parent_column_c}. "
                                f"Consider defining schema with constraints *before* inserting data, or use a database migration tool.")
                    # Placeholder for future (complex) implementation:
                    # 1. Read current CREATE TABLE statement for child table
                    # 2. Create a new temporary table with the FK constraint added
                    # 3. Copy data from old table to new table
                    # 4. Drop old table
                    # 5. Rename new table to old table name
                    # This is error-prone and complex, better handled by migration tools or defining schema upfront.

                except Exception as e:
                    logger.error(f"Unexpected error processing relationship {rel}: {e}", exc_info=True)

        except sqlite3.Error as e:
            logger.error(f"SQLite error during relationship processing: {e}")
        finally:
            if temp_conn:
                temp_conn.close()


    def get_connection_string(self) -> str:
        """
        Get the SQLAlchemy connection string for the database.

        Returns:
            SQLAlchemy connection string
        """
        if not self.db_path:
            raise ValueError("No database connected or path specified")

        # Ensure the path is absolute for consistency
        abs_db_path = os.path.abspath(self.db_path)
        return f"sqlite:///{abs_db_path}"


    def get_tables_info(self) -> Dict[str, Dict]:
        """
        Get information about tables in the database using SQLAlchemy Inspector.

        Returns:
            Dictionary with table info, including columns, PKs, FKs, row count, sample data.
        """
        if not self.engine:
            raise ValueError("Database engine not initialized. Call connect() first.")

        try: # Add try-except around inspection
            insp = inspect(self.engine)
            tables_info = {}
            table_names = insp.get_table_names()

            for table_name in table_names:
                 if table_name.startswith('sqlite_'): # Skip internal SQLite tables
                     continue

                 columns = []
                 try:
                     cols_data = insp.get_columns(table_name)
                     for col in cols_data:
                          columns.append({
                              "name": col["name"],
                              "type": str(col["type"]),
                              "nullable": col.get("nullable", True),
                              "default": col.get("default", None),
                              "pk": col.get("primary_key", False) # Use 'primary_key' flag
                          })
                 except Exception as e:
                      logger.error(f"Could not get column info for table {table_name}: {e}")
                      columns = [{"error": str(e)}]


                 foreign_keys = []
                 try:
                     fks_data = insp.get_foreign_keys(table_name)
                     for fk in fks_data:
                         foreign_keys.append({
                             "constrained_columns": fk["constrained_columns"],
                             "referred_table": fk["referred_table"],
                             "referred_columns": fk["referred_columns"],
                         })
                 except Exception as e:
                      logger.error(f"Could not get foreign key info for table {table_name}: {e}")
                      foreign_keys = [{"error": str(e)}]

                 primary_keys = []
                 try:
                      pk_constraint = insp.get_pk_constraint(table_name)
                      primary_keys = pk_constraint.get("constrained_columns", [])
                 except Exception as e:
                     logger.error(f"Could not get primary key info for table {table_name}: {e}")
                     primary_keys = ["error retrieving primary keys"]


                 row_count = -1 # Default if count fails
                 try:
                     with self.engine.connect() as connection:
                         # Use text() for literal SQL and ensure table name quoting
                         result = connection.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                         row_count = result.scalar_one_or_none() # Use scalar_one_or_none
                         if row_count is None: row_count = 0 # Handle case of empty table
                 except Exception as e:
                      logger.error(f"Could not get row count for table {table_name}: {e}")

                 sample_data = []
                 try:
                      # Use pandas for robust reading and type handling
                      sample_df = pd.read_sql(f'SELECT * FROM "{table_name}" LIMIT 5', self.engine)
                      # Convert to list of dicts, handling potential non-JSON serializable types
                      sample_data = sample_df.astype(object).where(pd.notnull(sample_df), None).to_dict('records')

                 except Exception as e:
                      logger.error(f"Could not get sample data for table {table_name}: {e}")
                      sample_data = [{"error": f"Could not fetch sample data: {e}"}]


                 tables_info[table_name] = {
                     "columns": columns,
                     "foreign_keys": foreign_keys,
                     "primary_keys": primary_keys, # Already retrieved
                     "row_count": row_count,
                     "sample_data": sample_data
                 }

            return tables_info
        except Exception as inspect_e:
             logger.error(f"Failed to inspect database schema: {inspect_e}", exc_info=True)
             raise RuntimeError(f"Failed to inspect database schema: {inspect_e}") from inspect_e


    def execute_query(self, query: str) -> Union[pd.DataFrame, List[Dict]]:
        """
        Execute a SQL query on the database and return results as a DataFrame.
        Falls back to list of dicts if DataFrame conversion fails.

        Args:
            query: SQL query string

        Returns:
            Pandas DataFrame with query results, or List of Dicts as fallback.
        """
        if not self.engine:
            raise ValueError("Database engine not initialized. Call connect() first.")

        try:
            if self.verbose:
                logger.info(f"Executing query: {query}")
            # Use pandas for robust query execution and type handling for SELECT
            if re.match(r'^\s*(SELECT|WITH)\b', query, re.IGNORECASE):
                df = pd.read_sql_query(query, self.engine)
                # Convert NaNs/NaTs to None for consistency before returning
                return df.astype(object).where(pd.notnull(df), None)
            else:
                # Handle non-SELECT statements
                logger.info(f"Executing non-SELECT query: {query}")
                with self.engine.begin() as connection: # Use transaction
                    result_proxy = connection.execute(text(query))
                    rowcount = result_proxy.rowcount
                    logger.info(f"Non-SELECT query executed. Rows affected (if applicable): {rowcount}")
                    return pd.DataFrame({"status": ["Execution successful"], "rows_affected": [rowcount]})

        except Exception as pd_e:
             # Log the pandas error and attempt fallback if appropriate
             logger.warning(f"Pandas read_sql failed (may be non-SELECT or error): {pd_e}. Falling back to basic execution for query: {query}")
             try:
                 with self.engine.connect() as connection:
                     # Use text() construct for safety and compatibility
                     cursor_result = connection.execute(text(query))
                     if cursor_result.returns_rows:
                         columns = list(cursor_result.keys()) # Get column names from ResultProxy
                         results = [dict(zip(columns, row)) for row in cursor_result.fetchall()]
                         return results
                     else:
                         rowcount = cursor_result.rowcount
                         logger.info(f"Query executed successfully using basic execution. Rows affected (if applicable): {rowcount}")
                         return pd.DataFrame({"status": ["Execution successful"], "rows_affected": [rowcount]})
             except Exception as e:
                 logger.error(f"Query execution failed using both pandas and basic execution: {e}", exc_info=True)
                 raise RuntimeError(f"Query execution failed: {e}") from e # Re-raise the final exception

    def close(self):
        """Close the database connection and dispose the engine"""
        # Removed self.conn handling as direct sqlite3 connection is optional
        # if self.conn:
        #     try:
        #         self.conn.close()
        #         self.conn = None
        #         logger.info("SQLite connection closed.")
        #     except Exception as e:
        #         logger.error(f"Error closing SQLite connection: {e}")
        if self.engine:
             try:
                 self.engine.dispose() # Release connection pool resources
                 self.engine = None
                 logger.info("SQLAlchemy engine disposed.")
             except Exception as e:
                 logger.error(f"Error disposing SQLAlchemy engine: {e}")


def main():
    """Command line interface for DataImporter_Gemini"""
    parser = argparse.ArgumentParser(description='Import Excel/CSV/JSON files into SQLite database using DataImporter_Gemini')

    parser.add_argument('--db', required=True, help='Path to SQLite database file to create/use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Command to execute', required=True)

    # Base import arguments (parent parser)
    base_import_parser = argparse.ArgumentParser(add_help=False)
    base_import_parser.add_argument('--file', required=True, help='Path to data file or glob pattern (e.g., "data/*.csv")')
    base_import_parser.add_argument('--table', help='Base table name for import. Defaults to filename stem or sheet name.')
    base_import_parser.add_argument('--pk', help='Primary key column name (original name before cleaning)')
    base_import_parser.add_argument('--if-exists', default='replace', choices=['fail', 'replace', 'append'], help='Action if table exists (default: replace)')


    # Excel import command
    excel_parser = subparsers.add_parser('excel', help='Import Excel file(s)', parents=[base_import_parser])
    excel_parser.add_argument('--sheet', help='Sheet name(s) or index(es) to import, comma-separated. Defaults to the first sheet (index 0). Use "None" (as a string) for all sheets.')

    # CSV import command
    csv_parser = subparsers.add_parser('csv', help='Import CSV file(s)', parents=[base_import_parser])
    csv_parser.add_argument('--delimiter', default=',', help='CSV delimiter character (default: ,)')
    csv_parser.add_argument('--encoding', default='utf-8', help='File encoding (default: utf-8)')

    # JSON import command
    json_parser = subparsers.add_parser('json', help='Import JSON file', parents=[base_import_parser])
    json_parser.add_argument('--orient', default='records', help='Pandas JSON orientation (e.g., records, split, index, columns, values, table)')
    json_parser.add_argument('--lines', action='store_true', help='Read the file as a JSON object per line.')


    # Info command
    info_parser = subparsers.add_parser('info', help='Display database schema information')

    # Query command
    query_parser = subparsers.add_parser('query', help='Execute a raw SQL query')
    query_parser.add_argument('sql', help='SQL query to execute')


    args = parser.parse_args()

    # Initialize importer
    importer = None # Initialize to None
    try:
        importer = DataImporter_Gemini(db_path=args.db, verbose=args.verbose)

        if args.command == 'excel':
            sheet_name_arg = args.sheet
            sheet_name_param = 0 # Default to first sheet
            if sheet_name_arg == "None":
                 sheet_name_param = None
            elif sheet_name_arg:
                 items = [s.strip() for s in sheet_name_arg.split(',')]
                 parsed_items = []
                 for item in items:
                      try: parsed_items.append(int(item))
                      except ValueError: parsed_items.append(item)
                 sheet_name_param = parsed_items[0] if len(parsed_items) == 1 else parsed_items

            # Simplified CLI table/pk mapping: Applies to first sheet if name provided
            table_names_map = None
            primary_keys_map = None
            if args.table and isinstance(sheet_name_param, (str, int)):
                 clean_table_name = importer._clean_column_name(args.table)
                 table_names_map = {sheet_name_param: clean_table_name}
                 if args.pk: primary_keys_map = {clean_table_name: args.pk} # Map using cleaned table name
            elif args.table and sheet_name_param is None:
                 logger.warning("--table name ignored when importing all sheets. Using sheet names as table names.")
                 if args.pk: logger.warning("--pk ignored when importing all sheets.")

            tables = importer.import_excel(
                file_input=args.file, # Use file_input to accept path/glob
                sheet_name=sheet_name_param,
                table_names=table_names_map,
                primary_keys=primary_keys_map,
                if_exists=args.if_exists
                # relationships - too complex for CLI, use programmatically
            )
            print(f"Imported/Updated tables from Excel: {', '.join(tables)}")

        elif args.command == 'csv':
            tables = importer.import_csv( # Use path-based import for CLI
                file_path=args.file,
                table_name=args.table,
                delimiter=args.delimiter,
                encoding=args.encoding,
                primary_key=args.pk,
                if_exists=args.if_exists
                 # relationships - too complex for CLI
            )
            print(f"Imported/Updated tables from CSV: {', '.join(tables)}")

        elif args.command == 'json':
             # Basic JSON import assuming a list of records or record-oriented structure
             try:
                 # Handle path/glob for JSON file(s)
                 json_files = glob.glob(args.file) if '*' in args.file or '?' in args.file else [args.file]
                 imported_json_tables = []
                 if not json_files:
                      print(f"No JSON files found matching pattern: {args.file}")
                 else:
                      for json_file in json_files:
                          if not os.path.exists(json_file):
                               logger.warning(f"JSON file not found: {json_file}. Skipping.")
                               continue
                          logger.info(f"Processing JSON file: {json_file}")
                          # Read JSON using pandas options
                          df = pd.read_json(json_file, orient=args.orient, lines=args.lines)
                          # Determine table name
                          table_name = args.table if args.table and len(json_files) == 1 else Path(json_file).stem
                          table_name = importer._clean_column_name(table_name)
                          # Use internal dataframe import method
                          importer._import_dataframe(df, table_name, primary_key=args.pk, if_exists=args.if_exists)
                          imported_json_tables.append(table_name)
                          print(f"Imported/Updated table from JSON '{json_file}': {table_name}")
                      print(f"Completed JSON import. Tables processed: {', '.join(imported_json_tables)}")

             except Exception as e:
                  print(f"Error importing JSON file {args.file}: {e}")


        elif args.command == 'info':
            tables_info = importer.get_tables_info()
            print(f"--- Database Schema Information ---")
            print(f"Database: {args.db}")
            print(f"Total Tables Found: {len(tables_info)}")
            print("-" * 30)

            for table_name, info in tables_info.items():
                print(f"\nTable: {table_name}")
                print(f"  Row count: {info.get('row_count', 'N/A')}")
                print(f"  Columns: ({len(info.get('columns', []))})")
                for col in info.get('columns', []):
                     pk_flag = " (PK)" if col.get('pk') else ""
                     nullable_flag = " NULL" if col.get('nullable') else " NOT NULL"
                     default_val = f" DEFAULT {col.get('default')}" if col.get('default') else ""
                     print(f"    - {col.get('name', '?')} ({col.get('type', '?')}){pk_flag}{nullable_flag}{default_val}")

                pks = info.get('primary_keys', [])
                if pks and isinstance(pks, list) and not any("error" in str(k) for k in pks):
                     print(f"  Primary Key Constraint: ({', '.join(pks)})")

                fks = info.get('foreign_keys', [])
                if fks and isinstance(fks, list) and not any("error" in str(k) for k in fks):
                    print(f"  Foreign Keys: ({len(fks)})")
                    for fk in fks:
                        cols = ', '.join(fk.get('constrained_columns', ['?']))
                        ref_table = fk.get('referred_table', '?')
                        ref_cols = ', '.join(fk.get('referred_columns', ['?']))
                        print(f"    - ({cols}) -> {ref_table}({ref_cols})")
                elif not fks:
                     print("  Foreign Keys: None")
                else: # Error case
                     print(f"  Foreign Keys: {fks}")

                print("  Sample Data (first 5 rows):")
                sample_data = info.get('sample_data', [])
                if sample_data and isinstance(sample_data, list) and not any("error" in str(s) for s in sample_data):
                     try:
                          sample_df = pd.DataFrame(sample_data)
                          print(sample_df.to_string(index=False, max_rows=5, max_cols=10))
                     except Exception as display_e:
                          print(f"    Could not display sample data nicely: {display_e}")
                elif sample_data and any("error" in str(s) for s in sample_data):
                     print(f"    Error retrieving sample data: {sample_data[0]}")
                else:
                    print("    (No sample data or table empty)")
                print("-" * 20)

        elif args.command == 'query':
             print(f"Executing SQL: {args.sql}")
             try:
                  results = importer.execute_query(args.sql)
                  print("\nQuery Results:")
                  if isinstance(results, pd.DataFrame):
                       # Check for status message from non-SELECT queries
                       if "status" in results.columns:
                            print(f"Status: {results['status'].iloc[0]}, Rows affected: {results['rows_affected'].iloc[0]}")
                       elif results.empty:
                            print("(Query returned no rows)")
                       else:
                           print(results.to_string(index=False))
                  elif isinstance(results, list):
                       if results: # Simple print for list of dicts
                            import json
                            print(json.dumps(results, indent=2, default=str))
                       else:
                            print("(Query executed successfully, no rows returned)")
                  else:
                       print("(No results returned or unexpected format)")
             except Exception as e:
                  print(f"\nError executing query: {e}")


        else:
            parser.print_help()

    except Exception as e:
         print(f"\nAn error occurred: {e}", file=sys.stderr)
         logger.error("Unhandled exception in DataImporter main", exc_info=True)
         sys.exit(1) # Exit with error code
    finally:
        if importer:
            importer.close()

if __name__ == "__main__":
    main()
# --- END OF FILE DataImporter_Gemini.py ---
