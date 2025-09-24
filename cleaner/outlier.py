"""
Outlier Detection Module

Provides comprehensive outlier detection capabilities for pandas DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class OutlierDetector:
    """
    A comprehensive outlier detection class for pandas DataFrames.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the OutlierDetector with a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to analyze for outliers
        """
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
    def detect_outliers_iqr(self, columns: Optional[List[str]] = None, 
                           threshold: float = 1.5) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers using the Interquartile Range (IQR) method.
        
        Args:
            columns (List[str], optional): Columns to analyze. If None, all numeric columns.
            threshold (float): IQR multiplier threshold (default: 1.5)
        
        Returns:
            Dict containing outlier information for each column
        """
        if columns is None:
            columns = self.numeric_columns
        
        outliers_info = {}
        
        for col in columns:
            if col not in self.df.columns or col not in self.numeric_columns:
                continue
                
            series = self.df[col].dropna()
            if len(series) == 0:
                continue
            
            # Calculate IQR
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            # Calculate bounds
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Find outliers
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_indices = self.df[outlier_mask].index.tolist()
            outlier_values = self.df.loc[outlier_mask, col].tolist()
            
            outliers_info[col] = {
                "method": "IQR",
                "threshold": threshold,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "Q1": float(Q1),
                "Q3": float(Q3),
                "IQR": float(IQR),
                "outlier_count": len(outlier_indices),
                "outlier_percentage": (len(outlier_indices) / len(self.df)) * 100,
                "outlier_indices": outlier_indices,
                "outlier_values": outlier_values,
                "outlier_stats": {
                    "min_outlier": float(min(outlier_values)) if outlier_values else None,
                    "max_outlier": float(max(outlier_values)) if outlier_values else None,
                    "mean_outlier": float(np.mean(outlier_values)) if outlier_values else None
                }
            }
        
        return outliers_info
    
    def detect_outliers_zscore(self, columns: Optional[List[str]] = None, 
                              threshold: float = 3.0) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers using the Z-Score method.
        
        Args:
            columns (List[str], optional): Columns to analyze. If None, all numeric columns.
            threshold (float): Z-score threshold (default: 3.0)
        
        Returns:
            Dict containing outlier information for each column
        """
        if columns is None:
            columns = self.numeric_columns
        
        outliers_info = {}
        
        for col in columns:
            if col not in self.df.columns or col not in self.numeric_columns:
                continue
                
            series = self.df[col].dropna()
            if len(series) == 0:
                continue
            
            # Calculate Z-scores
            mean_val = series.mean()
            std_val = series.std()
            
            if std_val == 0:  # No variation in data
                outliers_info[col] = {
                    "method": "Z-Score",
                    "threshold": threshold,
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "outlier_count": 0,
                    "outlier_percentage": 0.0,
                    "outlier_indices": [],
                    "outlier_values": [],
                    "outlier_stats": {
                        "min_outlier": None,
                        "max_outlier": None,
                        "mean_outlier": None
                    }
                }
                continue
            
            z_scores = np.abs((self.df[col] - mean_val) / std_val)
            
            # Find outliers
            outlier_mask = z_scores > threshold
            outlier_indices = self.df[outlier_mask].index.tolist()
            outlier_values = self.df.loc[outlier_mask, col].tolist()
            outlier_zscores = z_scores[outlier_mask].tolist()
            
            outliers_info[col] = {
                "method": "Z-Score",
                "threshold": threshold,
                "mean": float(mean_val),
                "std": float(std_val),
                "outlier_count": len(outlier_indices),
                "outlier_percentage": (len(outlier_indices) / len(self.df)) * 100,
                "outlier_indices": outlier_indices,
                "outlier_values": outlier_values,
                "outlier_zscores": outlier_zscores,
                "outlier_stats": {
                    "min_outlier": float(min(outlier_values)) if outlier_values else None,
                    "max_outlier": float(max(outlier_values)) if outlier_values else None,
                    "mean_outlier": float(np.mean(outlier_values)) if outlier_values else None,
                    "max_zscore": float(max(outlier_zscores)) if outlier_zscores else None
                }
            }
        
        return outliers_info
    
    def detect_outliers_modified_zscore(self, columns: Optional[List[str]] = None, 
                                       threshold: float = 3.5) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers using the Modified Z-Score method (using median).
        
        Args:
            columns (List[str], optional): Columns to analyze. If None, all numeric columns.
            threshold (float): Modified Z-score threshold (default: 3.5)
        
        Returns:
            Dict containing outlier information for each column
        """
        if columns is None:
            columns = self.numeric_columns
        
        outliers_info = {}
        
        for col in columns:
            if col not in self.df.columns or col not in self.numeric_columns:
                continue
                
            series = self.df[col].dropna()
            if len(series) == 0:
                continue
            
            # Calculate Modified Z-scores
            median_val = series.median()
            mad = np.median(np.abs(series - median_val))
            
            if mad == 0:  # No variation in data
                outliers_info[col] = {
                    "method": "Modified Z-Score",
                    "threshold": threshold,
                    "median": float(median_val),
                    "mad": float(mad),
                    "outlier_count": 0,
                    "outlier_percentage": 0.0,
                    "outlier_indices": [],
                    "outlier_values": [],
                    "outlier_stats": {
                        "min_outlier": None,
                        "max_outlier": None,
                        "mean_outlier": None
                    }
                }
                continue
            
            modified_z_scores = 0.6745 * (self.df[col] - median_val) / mad
            
            # Find outliers
            outlier_mask = np.abs(modified_z_scores) > threshold
            outlier_indices = self.df[outlier_mask].index.tolist()
            outlier_values = self.df.loc[outlier_mask, col].tolist()
            outlier_mod_zscores = modified_z_scores[outlier_mask].tolist()
            
            outliers_info[col] = {
                "method": "Modified Z-Score",
                "threshold": threshold,
                "median": float(median_val),
                "mad": float(mad),
                "outlier_count": len(outlier_indices),
                "outlier_percentage": (len(outlier_indices) / len(self.df)) * 100,
                "outlier_indices": outlier_indices,
                "outlier_values": outlier_values,
                "outlier_mod_zscores": outlier_mod_zscores,
                "outlier_stats": {
                    "min_outlier": float(min(outlier_values)) if outlier_values else None,
                    "max_outlier": float(max(outlier_values)) if outlier_values else None,
                    "mean_outlier": float(np.mean(outlier_values)) if outlier_values else None,
                    "max_mod_zscore": float(max(np.abs(outlier_mod_zscores))) if outlier_mod_zscores else None
                }
            }
        
        return outliers_info
    
    def get_outlier_summary(self, method: str = "iqr", **kwargs) -> Dict[str, Any]:
        """
        Get a summary of outliers across all numeric columns.
        
        Args:
            method (str): Detection method ('iqr', 'zscore', 'modified_zscore')
            **kwargs: Additional arguments for the detection method
        
        Returns:
            Dict containing outlier summary
        """
        if method == "iqr":
            outliers_info = self.detect_outliers_iqr(**kwargs)
        elif method == "zscore":
            outliers_info = self.detect_outliers_zscore(**kwargs)
        elif method == "modified_zscore":
            outliers_info = self.detect_outliers_modified_zscore(**kwargs)
        else:
            raise ValueError("Method must be 'iqr', 'zscore', or 'modified_zscore'")
        
        total_outliers = sum(info["outlier_count"] for info in outliers_info.values())
        columns_with_outliers = [col for col, info in outliers_info.items() if info["outlier_count"] > 0]
        
        return {
            "method": method,
            "total_outliers": total_outliers,
            "columns_analyzed": len(outliers_info),
            "columns_with_outliers": len(columns_with_outliers),
            "outlier_percentage": (total_outliers / len(self.df)) * 100 if len(self.df) > 0 else 0,
            "outliers_by_column": {
                col: info["outlier_count"] for col, info in outliers_info.items()
            },
            "detailed_info": outliers_info
        }
    
    def remove_outliers(self, method: str = "iqr", columns: Optional[List[str]] = None, 
                       **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove outliers from the DataFrame.
        
        Args:
            method (str): Detection method ('iqr', 'zscore', 'modified_zscore')
            columns (List[str], optional): Columns to process. If None, all numeric columns.
            **kwargs: Additional arguments for the detection method
        
        Returns:
            Tuple of (cleaned DataFrame, outlier summary)
        """
        if method == "iqr":
            outliers_info = self.detect_outliers_iqr(columns=columns, **kwargs)
        elif method == "zscore":
            outliers_info = self.detect_outliers_zscore(columns=columns, **kwargs)
        elif method == "modified_zscore":
            outliers_info = self.detect_outliers_modified_zscore(columns=columns, **kwargs)
        else:
            raise ValueError("Method must be 'iqr', 'zscore', or 'modified_zscore'")
        
        # Collect all outlier indices
        all_outlier_indices = set()
        for col_info in outliers_info.values():
            all_outlier_indices.update(col_info["outlier_indices"])
        
        # Remove outliers
        df_cleaned = self.df.drop(index=list(all_outlier_indices))
        
        # Create summary
        summary = {
            "method": method,
            "original_rows": len(self.df),
            "rows_removed": len(all_outlier_indices),
            "remaining_rows": len(df_cleaned),
            "removal_percentage": (len(all_outlier_indices) / len(self.df)) * 100 if len(self.df) > 0 else 0,
            "outliers_by_column": outliers_info
        }
        
        return df_cleaned, summary
    
    def visualize_outliers(self, column: str, method: str = "iqr", **kwargs) -> Dict[str, Any]:
        """
        Get data for visualizing outliers in a specific column.
        
        Args:
            column (str): Column to analyze
            method (str): Detection method ('iqr', 'zscore', 'modified_zscore')
            **kwargs: Additional arguments for the detection method
        
        Returns:
            Dict containing visualization data
        """
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric")
        
        if method == "iqr":
            outliers_info = self.detect_outliers_iqr(columns=[column], **kwargs)
        elif method == "zscore":
            outliers_info = self.detect_outliers_zscore(columns=[column], **kwargs)
        elif method == "modified_zscore":
            outliers_info = self.detect_outliers_modified_zscore(columns=[column], **kwargs)
        else:
            raise ValueError("Method must be 'iqr', 'zscore', or 'modified_zscore'")
        
        col_info = outliers_info.get(column, {})
        series = self.df[column].dropna()
        
        return {
            "column": column,
            "method": method,
            "data_points": series.tolist(),
            "outlier_info": col_info,
            "statistics": {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75))
            }
        }