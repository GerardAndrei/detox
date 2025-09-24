"""
Data Cleaning Module

Provides comprehensive data cleaning capabilities for pandas DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """
    A comprehensive data cleaning class for pandas DataFrames.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataCleaner with a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to clean
        """
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_log = []
        
    def handle_missing_values(self, strategy: str = "auto") -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Strategy for handling missing values
                          - "auto": Automatic strategy based on data type
                          - "drop": Drop rows with missing values
                          - "fill_mean": Fill with mean for numeric, mode for categorical
                          - "fill_median": Fill with median for numeric, mode for categorical
                          - "fill_mode": Fill with mode for all columns
                          - "forward_fill": Forward fill missing values
                          - "backward_fill": Backward fill missing values
        
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        df_cleaned = self.df.copy()
        missing_before = df_cleaned.isnull().sum().sum()
        
        if missing_before == 0:
            self.cleaning_log.append("No missing values found")
            return df_cleaned
        
        if strategy == "auto":
            # Automatic strategy based on data types and missing percentage
            for col in df_cleaned.columns:
                missing_pct = (df_cleaned[col].isnull().sum() / len(df_cleaned)) * 100
                
                if missing_pct > 50:
                    # Drop columns with >50% missing
                    df_cleaned = df_cleaned.drop(columns=[col])
                    self.cleaning_log.append(f"Dropped column '{col}' (>{missing_pct:.1f}% missing)")
                elif df_cleaned[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    median_val = df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median_val)
                    self.cleaning_log.append(f"Filled missing values in '{col}' with median ({median_val})")
                elif df_cleaned[col].dtype == 'object':
                    # Fill categorical columns with mode
                    mode_val = df_cleaned[col].mode()
                    if len(mode_val) > 0:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
                        self.cleaning_log.append(f"Filled missing values in '{col}' with mode ('{mode_val[0]}')")
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna("Unknown")
                        self.cleaning_log.append(f"Filled missing values in '{col}' with 'Unknown'")
        
        elif strategy == "drop":
            rows_before = len(df_cleaned)
            df_cleaned = df_cleaned.dropna()
            rows_dropped = rows_before - len(df_cleaned)
            self.cleaning_log.append(f"Dropped {rows_dropped} rows with missing values")
        
        elif strategy == "fill_mean":
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype in ['int64', 'float64']:
                    mean_val = df_cleaned[col].mean()
                    df_cleaned[col] = df_cleaned[col].fillna(mean_val)
                elif df_cleaned[col].dtype == 'object':
                    mode_val = df_cleaned[col].mode()
                    if len(mode_val) > 0:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
            self.cleaning_log.append("Filled missing values with mean/mode")
        
        elif strategy == "fill_median":
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype in ['int64', 'float64']:
                    median_val = df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median_val)
                elif df_cleaned[col].dtype == 'object':
                    mode_val = df_cleaned[col].mode()
                    if len(mode_val) > 0:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
            self.cleaning_log.append("Filled missing values with median/mode")
        
        elif strategy == "fill_mode":
            for col in df_cleaned.columns:
                mode_val = df_cleaned[col].mode()
                if len(mode_val) > 0:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
            self.cleaning_log.append("Filled missing values with mode")
        
        elif strategy == "forward_fill":
            df_cleaned = df_cleaned.fillna(method='ffill')
            self.cleaning_log.append("Applied forward fill for missing values")
        
        elif strategy == "backward_fill":
            df_cleaned = df_cleaned.fillna(method='bfill')
            self.cleaning_log.append("Applied backward fill for missing values")
        
        missing_after = df_cleaned.isnull().sum().sum()
        self.cleaning_log.append(f"Reduced missing values from {missing_before} to {missing_after}")
        
        self.df = df_cleaned
        return df_cleaned
    
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            subset (List[str], optional): Columns to consider for duplicate detection
            keep (str): Which duplicates to keep ('first', 'last', False)
        
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        df_cleaned = self.df.copy()
        duplicates_before = df_cleaned.duplicated(subset=subset).sum()
        
        if duplicates_before == 0:
            self.cleaning_log.append("No duplicate rows found")
            return df_cleaned
        
        df_cleaned = df_cleaned.drop_duplicates(subset=subset, keep=keep)
        duplicates_removed = duplicates_before
        
        self.cleaning_log.append(f"Removed {duplicates_removed} duplicate rows")
        
        self.df = df_cleaned
        return df_cleaned
    
    def standardize_text(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Standardize text in specified columns.
        
        Args:
            columns (List[str], optional): Columns to standardize. If None, all text columns.
        
        Returns:
            pd.DataFrame: DataFrame with standardized text
        """
        df_cleaned = self.df.copy()
        
        if columns is None:
            columns = df_cleaned.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
                # Remove leading/trailing whitespace
                df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
                
                # Convert to title case for better consistency
                df_cleaned[col] = df_cleaned[col].str.title()
                
                # Replace multiple spaces with single space
                df_cleaned[col] = df_cleaned[col].str.replace(r'\s+', ' ', regex=True)
                
                self.cleaning_log.append(f"Standardized text in column '{col}'")
        
        self.df = df_cleaned
        return df_cleaned
    
    def convert_data_types(self, type_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Convert data types of specified columns.
        
        Args:
            type_mapping (Dict[str, str]): Mapping of column names to target data types
        
        Returns:
            pd.DataFrame: DataFrame with converted data types
        """
        df_cleaned = self.df.copy()
        
        for col, target_type in type_mapping.items():
            if col not in df_cleaned.columns:
                continue
                
            try:
                if target_type == 'numeric':
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                elif target_type == 'datetime':
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                elif target_type == 'category':
                    df_cleaned[col] = df_cleaned[col].astype('category')
                elif target_type == 'boolean':
                    df_cleaned[col] = df_cleaned[col].astype('bool')
                elif target_type == 'string':
                    df_cleaned[col] = df_cleaned[col].astype(str)
                
                self.cleaning_log.append(f"Converted '{col}' to {target_type}")
                
            except Exception as e:
                self.cleaning_log.append(f"Failed to convert '{col}' to {target_type}: {str(e)}")
        
        self.df = df_cleaned
        return df_cleaned
    
    def remove_outliers(self, columns: Optional[List[str]] = None, method: str = "iqr", 
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from numeric columns.
        
        Args:
            columns (List[str], optional): Columns to process. If None, all numeric columns.
            method (str): Method for outlier detection ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection
        
        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        df_cleaned = self.df.copy()
        
        if columns is None:
            columns = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_removed = 0
        
        for col in columns:
            if col not in df_cleaned.columns:
                continue
                
            original_count = len(df_cleaned)
            
            if method == "iqr":
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                df_cleaned = df_cleaned[
                    (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)
                ]
                
            elif method == "zscore":
                z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
                df_cleaned = df_cleaned[z_scores <= threshold]
            
            outliers_removed += original_count - len(df_cleaned)
        
        if outliers_removed > 0:
            self.cleaning_log.append(f"Removed {outliers_removed} outlier rows using {method} method")
        else:
            self.cleaning_log.append("No outliers detected")
        
        self.df = df_cleaned
        return df_cleaned
    
    def clean_column_names(self) -> pd.DataFrame:
        """
        Clean and standardize column names.
        
        Returns:
            pd.DataFrame: DataFrame with cleaned column names
        """
        df_cleaned = self.df.copy()
        
        # Store original column names
        original_columns = df_cleaned.columns.tolist()
        
        # Clean column names
        new_columns = []
        for col in original_columns:
            # Convert to lowercase
            clean_col = str(col).lower()
            
            # Replace spaces and special characters with underscores
            clean_col = clean_col.replace(' ', '_').replace('-', '_')
            
            # Remove special characters except underscores
            clean_col = ''.join(c for c in clean_col if c.isalnum() or c == '_')
            
            # Remove multiple underscores
            clean_col = '_'.join(part for part in clean_col.split('_') if part)
            
            new_columns.append(clean_col)
        
        df_cleaned.columns = new_columns
        
        # Log changes
        changes = [(orig, new) for orig, new in zip(original_columns, new_columns) if orig != new]
        if changes:
            self.cleaning_log.append(f"Renamed {len(changes)} columns for consistency")
        
        self.df = df_cleaned
        return df_cleaned
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all cleaning operations performed.
        
        Returns:
            Dict containing cleaning summary
        """
        return {
            "original_shape": self.original_shape,
            "current_shape": self.df.shape,
            "rows_removed": self.original_shape[0] - self.df.shape[0],
            "columns_removed": self.original_shape[1] - self.df.shape[1],
            "cleaning_steps": self.cleaning_log.copy(),
            "data_reduction_percentage": (
                (self.original_shape[0] - self.df.shape[0]) / self.original_shape[0]
            ) * 100 if self.original_shape[0] > 0 else 0
        }