"""
Data Loader Module
=================

Handles loading and saving data from various sources including:
- CSV files
- Excel files
- JSON files
- Database connections
- API endpoints
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import json
import sqlite3
from sqlalchemy import create_engine
import requests


class DataLoader:
    """
    Data loader class for handling various data sources and formats.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Supported file formats
        self.supported_formats = {
            'csv': self._load_csv,
            'excel': self._load_excel,
            'xlsx': self._load_excel,
            'xls': self._load_excel,
            'json': self._load_json,
            'parquet': self._load_parquet,
            'sql': self._load_from_database,
            'sqlite': self._load_from_sqlite
        }
        
        # Default settings
        self.default_csv_settings = {
            'encoding': 'utf-8',
            'delimiter': ',',
            'quotechar': '"',
            'na_values': ['', 'NULL', 'null', 'N/A', 'n/a', 'NaN', 'nan'],
            'keep_default_na': True,
            'low_memory': False
        }
    
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from file with automatic format detection.
        
        Args:
            file_path: Path to data file
            **kwargs: Additional parameters for specific loaders
            
        Returns:
            Loaded DataFrame
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect file format
        file_extension = path.suffix.lower().lstrip('.')
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        self.logger.info(f"Loading data from {file_path} (format: {file_extension})")
        
        # Load data using appropriate method
        loader_func = self.supported_formats[file_extension]
        data = loader_func(file_path, **kwargs)
        
        self.logger.info(f"Loaded {len(data)} rows and {len(data.columns)} columns")
        
        # Log basic info about the loaded data
        self._log_data_info(data)
        
        return data
    
    def _load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional pandas.read_csv parameters
            
        Returns:
            DataFrame with loaded data
        """
        # Merge default settings with user provided kwargs
        csv_params = {**self.default_csv_settings, **kwargs}
        
        try:
            # Try to detect encoding if not specified
            if 'encoding' not in kwargs:
                csv_params['encoding'] = self._detect_encoding(file_path)
            
            data = pd.read_csv(file_path, **csv_params)
            
            # Handle empty DataFrame
            if data.empty:
                self.logger.warning("Loaded CSV file is empty")
            
            return data
            
        except UnicodeDecodeError as e:
            self.logger.warning(f"Encoding error: {e}. Trying with different encoding...")
            # Fallback to latin-1 encoding
            csv_params['encoding'] = 'latin-1'
            return pd.read_csv(file_path, **csv_params)
    
    def _load_excel(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Args:
            file_path: Path to Excel file
            **kwargs: Additional pandas.read_excel parameters
            
        Returns:
            DataFrame with loaded data
        """
        # Default Excel parameters
        excel_params = {
            'na_values': self.default_csv_settings['na_values'],
            **kwargs
        }
        
        try:
            data = pd.read_excel(file_path, **excel_params)
            
            if data.empty:
                self.logger.warning("Loaded Excel file is empty")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading Excel file: {e}")
            raise
    
    def _load_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            **kwargs: Additional pandas.read_json parameters
            
        Returns:
            DataFrame with loaded data
        """
        try:
            data = pd.read_json(file_path, **kwargs)
            
            if data.empty:
                self.logger.warning("Loaded JSON file is empty")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading JSON file: {e}")
            # Try loading as regular JSON and converting to DataFrame
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if isinstance(json_data, list):
                return pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                return pd.DataFrame([json_data])
            else:
                raise ValueError("JSON data cannot be converted to DataFrame")
    
    def _load_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from Parquet file.
        
        Args:
            file_path: Path to Parquet file
            **kwargs: Additional pandas.read_parquet parameters
            
        Returns:
            DataFrame with loaded data
        """
        try:
            data = pd.read_parquet(file_path, **kwargs)
            return data
        except Exception as e:
            self.logger.error(f"Error loading Parquet file: {e}")
            raise
    
    def _load_from_database(self, connection_string: str, query: str = None, 
                           table_name: str = None, **kwargs) -> pd.DataFrame:
        """
        Load data from database.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            table_name: Table name to load (if no query provided)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with loaded data
        """
        try:
            engine = create_engine(connection_string)
            
            if query:
                data = pd.read_sql(query, engine, **kwargs)
            elif table_name:
                data = pd.read_sql_table(table_name, engine, **kwargs)
            else:
                raise ValueError("Either query or table_name must be provided")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading from database: {e}")
            raise
    
    def _load_from_sqlite(self, file_path: str, query: str = None, 
                         table_name: str = None, **kwargs) -> pd.DataFrame:
        """
        Load data from SQLite database.
        
        Args:
            file_path: Path to SQLite file
            query: SQL query to execute
            table_name: Table name to load
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with loaded data
        """
        try:
            conn = sqlite3.connect(file_path)
            
            if query:
                data = pd.read_sql_query(query, conn, **kwargs)
            elif table_name:
                data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn, **kwargs)
            else:
                # List all tables
                tables = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table'", 
                    conn
                )
                self.logger.info(f"Available tables: {tables['name'].tolist()}")
                raise ValueError("Either query or table_name must be provided")
            
            conn.close()
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading from SQLite: {e}")
            raise
    
    def save_data(self, data: pd.DataFrame, file_path: str, **kwargs) -> None:
        """
        Save DataFrame to file with automatic format detection.
        
        Args:
            data: DataFrame to save
            file_path: Output file path
            **kwargs: Additional parameters for specific savers
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        file_extension = path.suffix.lower().lstrip('.')
        
        self.logger.info(f"Saving data to {file_path} (format: {file_extension})")
        
        save_methods = {
            'csv': self._save_csv,
            'excel': self._save_excel,
            'xlsx': self._save_excel,
            'json': self._save_json,
            'parquet': self._save_parquet
        }
        
        if file_extension not in save_methods:
            # Default to CSV
            file_extension = 'csv'
            file_path = str(path.with_suffix('.csv'))
        
        save_method = save_methods[file_extension]
        save_method(data, file_path, **kwargs)
        
        self.logger.info(f"Data saved successfully to {file_path}")
    
    def _save_csv(self, data: pd.DataFrame, file_path: str, **kwargs) -> None:
        """Save DataFrame to CSV file."""
        csv_params = {
            'index': False,
            'encoding': 'utf-8',
            **kwargs
        }
        data.to_csv(file_path, **csv_params)
    
    def _save_excel(self, data: pd.DataFrame, file_path: str, **kwargs) -> None:
        """Save DataFrame to Excel file."""
        excel_params = {
            'index': False,
            'engine': 'openpyxl',
            **kwargs
        }
        data.to_excel(file_path, **excel_params)
    
    def _save_json(self, data: pd.DataFrame, file_path: str, **kwargs) -> None:
        """Save DataFrame to JSON file."""
        json_params = {
            'orient': 'records',
            'indent': 2,
            **kwargs
        }
        data.to_json(file_path, **json_params)
    
    def _save_parquet(self, data: pd.DataFrame, file_path: str, **kwargs) -> None:
        """Save DataFrame to Parquet file."""
        data.to_parquet(file_path, **kwargs)
    
    def _detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding.
        
        Args:
            file_path: Path to file
            
        Returns:
            Detected encoding
        """
        try:
            import chardet
            
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                self.logger.debug(
                    f"Detected encoding: {encoding} (confidence: {confidence:.2f})"
                )
                
                return encoding if confidence > 0.7 else 'utf-8'
        
        except ImportError:
            self.logger.warning("chardet not available, using utf-8 encoding")
            return 'utf-8'
        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    
    def _log_data_info(self, data: pd.DataFrame) -> None:
        """
        Log basic information about loaded data.
        
        Args:
            data: DataFrame to analyze
        """
        self.logger.info(f"Data shape: {data.shape}")
        self.logger.info(f"Columns: {list(data.columns)}")
        
        # Memory usage
        memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        self.logger.info(f"Memory usage: {memory_mb:.2f} MB")
        
        # Missing values summary
        missing_counts = data.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            self.logger.info(f"Total missing values: {total_missing}")
            
            # Log columns with missing values
            missing_cols = missing_counts[missing_counts > 0]
            for col, count in missing_cols.items():
                percentage = (count / len(data)) * 100
                self.logger.info(f"  {col}: {count} ({percentage:.1f}%)")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file without loading it.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        info = {
            'file_path': str(path.absolute()),
            'file_name': path.name,
            'file_size': path.stat().st_size,
            'file_extension': path.suffix.lower().lstrip('.'),
            'modified_time': path.stat().st_mtime
        }
        
        # Add format-specific information
        if info['file_extension'] == 'csv':
            # For CSV, try to detect delimiter and encoding
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    
                # Simple delimiter detection
                delimiters = [',', ';', '\t', '|']
                delimiter_counts = {d: first_line.count(d) for d in delimiters}
                likely_delimiter = max(delimiter_counts, key=delimiter_counts.get)
                
                info['likely_delimiter'] = likely_delimiter
                info['encoding'] = self._detect_encoding(file_path)
                
            except Exception:
                pass
        
        elif info['file_extension'] in ['xlsx', 'xls']:
            # For Excel files, list sheet names
            try:
                excel_file = pd.ExcelFile(file_path)
                info['sheet_names'] = excel_file.sheet_names
                excel_file.close()
            except Exception:
                pass
        
        return info
    
    def preview_data(self, file_path: str, n_rows: int = 5) -> pd.DataFrame:
        """
        Preview first few rows of data without loading entire file.
        
        Args:
            file_path: Path to data file
            n_rows: Number of rows to preview
            
        Returns:
            DataFrame with preview data
        """
        path = Path(file_path)
        file_extension = path.suffix.lower().lstrip('.')
        
        if file_extension == 'csv':
            return pd.read_csv(file_path, nrows=n_rows, **self.default_csv_settings)
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(file_path, nrows=n_rows)
        elif file_extension == 'json':
            # For JSON, load and take first n_rows
            data = self._load_json(file_path)
            return data.head(n_rows)
        else:
            # Fallback: load entire file and take first rows
            data = self.load_data(file_path)
            return data.head(n_rows)
