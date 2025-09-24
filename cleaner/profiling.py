"""
AI-Powered Data Profiling Module

Provides comprehensive analysis and profiling capabilities for pandas DataFrames
with machine learning-based insights and intelligent data quality assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML and statistical libraries
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    from scipy import stats
    from scipy.stats import entropy
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML libraries not available. Installing scikit-learn is recommended for AI features.")

class AIDataProfiler:
    """
    An AI-powered comprehensive data profiling class for analyzing pandas DataFrames
    with machine learning-based insights and intelligent recommendations.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the AIDataProfiler with a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame to profile
        """
        self.df = df.copy()
        self.shape = df.shape
        self.columns = df.columns.tolist()
        self.ml_available = ML_AVAILABLE
        
    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset with AI-enhanced insights.
        
        Returns:
            Dict containing basic dataset information
        """
        # Count columns by data type with intelligent detection
        numeric_cols = len(self.df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(self.df.select_dtypes(include=['object', 'category']).columns)
        datetime_cols = len(self.df.select_dtypes(include=['datetime64']).columns)
        boolean_cols = len(self.df.select_dtypes(include=['bool']).columns)
        
        # AI-enhanced data type suggestions
        ai_suggestions = self._ai_suggest_data_types() if self.ml_available else {}
        
        # Calculate data complexity score
        complexity_score = self._calculate_complexity_score()
        
        return {
            "shape": list(self.shape),
            "total_columns": len(self.columns),
            "total_rows": int(self.shape[0]),
            "columns": len(self.columns),  # Keep for backward compatibility
            "rows": int(self.shape[0]),    # Keep for backward compatibility
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "boolean_columns": boolean_cols,
            "memory_usage": int(self.df.memory_usage(deep=True).sum()),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "encoding": "utf-8",  # Default encoding info
            "data_complexity_score": complexity_score,
            "ai_type_suggestions": ai_suggestions,
            "data_quality_summary": self._get_quick_quality_summary()
        }
    
    def _calculate_complexity_score(self) -> float:
        """
        Calculate a complexity score for the dataset (0-100).
        Higher scores indicate more complex data requiring careful handling.
        """
        score = 0.0
        
        # Factor 1: Missing value complexity (30%)
        missing_ratio = self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])
        score += min(missing_ratio * 100, 30)
        
        # Factor 2: Data type diversity (20%)
        type_diversity = len(self.df.dtypes.unique()) / 10  # Normalize to 0-1
        score += min(type_diversity * 20, 20)
        
        # Factor 3: Categorical cardinality (25%)
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            avg_cardinality = np.mean([self.df[col].nunique() / len(self.df) for col in cat_cols])
            score += min(avg_cardinality * 25, 25)
        
        # Factor 4: Numerical distribution skewness (25%)
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            try:
                skewness_values = [abs(self.df[col].skew()) for col in num_cols if self.df[col].notna().sum() > 1]
                if skewness_values:
                    avg_skewness = np.mean(skewness_values)
                    score += min(avg_skewness * 5, 25)  # Skewness > 5 is very high
            except:
                pass
        
        return min(score, 100.0)
    
    def _get_quick_quality_summary(self) -> Dict[str, Any]:
        """Get a quick AI assessment of data quality."""
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        duplicate_rows = self.df.duplicated().sum()
        
        # Calculate quality score (0-100)
        completeness = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 100
        uniqueness = (1 - duplicate_rows / self.df.shape[0]) * 100 if self.df.shape[0] > 0 else 100
        
        overall_quality = (completeness * 0.6 + uniqueness * 0.4)
        
        # AI-powered quality assessment
        if overall_quality >= 90:
            quality_level = "EXCELLENT"
            ai_recommendation = "Data is high quality and ready for analysis"
        elif overall_quality >= 75:
            quality_level = "GOOD"
            ai_recommendation = "Minor cleaning recommended before analysis"
        elif overall_quality >= 50:
            quality_level = "MODERATE"
            ai_recommendation = "Significant cleaning required - missing values and duplicates detected"
        else:
            quality_level = "POOR"
            ai_recommendation = "Extensive data cleaning required - high missing data or duplicates"
        
        return {
            "overall_quality_score": round(overall_quality, 2),
            "quality_level": quality_level,
            "completeness": round(completeness, 2),
            "uniqueness": round(uniqueness, 2),
            "ai_recommendation": ai_recommendation
        }
    
    def _ai_suggest_data_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Use AI/ML to suggest optimal data types for columns.
        """
        if not self.ml_available:
            return {}
        
        suggestions = {}
        
        for col in self.df.columns:
            current_type = str(self.df[col].dtype)
            suggestion = {"current_type": current_type, "suggested_type": current_type, "confidence": 1.0, "reason": "No change needed"}
            
            if current_type == 'object':
                # AI-powered type detection for object columns
                suggestion = self._analyze_object_column(col)
            elif current_type in ['int64', 'float64']:
                # AI-powered optimization for numeric columns
                suggestion = self._analyze_numeric_column(col)
            
            suggestions[col] = suggestion
        
        return suggestions
    
    def _analyze_object_column(self, col: str) -> Dict[str, Any]:
        """Analyze object column with AI to suggest optimal type."""
        series = self.df[col].dropna()
        if len(series) == 0:
            return {"current_type": "object", "suggested_type": "object", "confidence": 1.0, "reason": "No non-null values"}
        
        # Try to detect if it's actually numeric
        try:
            pd.to_numeric(series, errors='raise')
            return {
                "current_type": "object",
                "suggested_type": "numeric",
                "confidence": 0.95,
                "reason": "All values can be converted to numeric"
            }
        except:
            pass
        
        # Try to detect if it's datetime
        try:
            pd.to_datetime(series, errors='raise')
            return {
                "current_type": "object",
                "suggested_type": "datetime",
                "confidence": 0.90,
                "reason": "Values appear to be datetime format"
            }
        except:
            pass
        
        # Check if it should be categorical
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.5:
            return {
                "current_type": "object",
                "suggested_type": "category",
                "confidence": 0.8,
                "reason": f"Low unique ratio ({unique_ratio:.2f}) suggests categorical data"
            }
        
        return {"current_type": "object", "suggested_type": "object", "confidence": 1.0, "reason": "Best kept as text"}
    
    def _analyze_numeric_column(self, col: str) -> Dict[str, Any]:
        """Analyze numeric column for optimization opportunities."""
        series = self.df[col].dropna()
        if len(series) == 0:
            return {"current_type": str(self.df[col].dtype), "suggested_type": str(self.df[col].dtype), "confidence": 1.0, "reason": "No non-null values"}
        
        # Check if it's actually boolean (0/1 or True/False)
        unique_vals = set(series.unique())
        if unique_vals.issubset({0, 1}) or unique_vals.issubset({True, False}):
            return {
                "current_type": str(self.df[col].dtype),
                "suggested_type": "boolean",
                "confidence": 0.95,
                "reason": "Only binary values detected"
            }
        
        # Check if integers could be optimized to smaller int types
        if str(self.df[col].dtype) == 'int64':
            min_val, max_val = series.min(), series.max()
            if min_val >= 0 and max_val <= 255:
                return {
                    "current_type": "int64",
                    "suggested_type": "uint8",
                    "confidence": 0.9,
                    "reason": f"Values range {min_val}-{max_val} fits in uint8"
                }
            elif min_val >= -128 and max_val <= 127:
                return {
                    "current_type": "int64",
                    "suggested_type": "int8",
                    "confidence": 0.9,
                    "reason": f"Values range {min_val}-{max_val} fits in int8"
                }
        
        return {"current_type": str(self.df[col].dtype), "suggested_type": str(self.df[col].dtype), "confidence": 1.0, "reason": "Optimal type"}
    
    def get_ai_anomaly_detection(self) -> Dict[str, Any]:
        """
        Use AI/ML for advanced anomaly detection across the dataset.
        """
        if not self.ml_available:
            return {"error": "ML libraries not available"}
        
        anomalies = {}
        
        # Anomaly detection for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            anomalies["numeric_anomalies"] = self._detect_numeric_anomalies(numeric_cols)
        
        # Pattern anomalies for text columns
        text_cols = self.df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            anomalies["text_anomalies"] = self._detect_text_anomalies(text_cols)
        
        return anomalies
    
    def _detect_numeric_anomalies(self, numeric_cols: List[str]) -> Dict[str, Any]:
        """Use Isolation Forest for numeric anomaly detection."""
        try:
            # Prepare data for ML
            numeric_data = self.df[numeric_cols].select_dtypes(include=[np.number])
            numeric_data = numeric_data.fillna(numeric_data.median())
            
            if len(numeric_data) < 10:  # Need minimum samples
                return {"error": "Insufficient data for ML anomaly detection"}
            
            # Use Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(numeric_data)
            
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            anomaly_count = len(anomaly_indices)
            
            return {
                "method": "Isolation Forest",
                "total_anomalies": int(anomaly_count),
                "anomaly_percentage": round(anomaly_count / len(numeric_data) * 100, 2),
                "anomaly_indices": anomaly_indices.tolist()[:10],  # Return max 10 examples
                "confidence": 0.85
            }
        except Exception as e:
            return {"error": f"ML anomaly detection failed: {str(e)}"}
    
    def _detect_text_anomalies(self, text_cols: List[str]) -> Dict[str, Any]:
        """Detect text pattern anomalies."""
        anomalies = {}
        
        for col in text_cols[:3]:  # Limit to first 3 text columns
            try:
                series = self.df[col].dropna().astype(str)
                if len(series) == 0:
                    continue
                
                # Length-based anomalies
                lengths = series.str.len()
                q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                length_anomalies = series[(lengths < lower_bound) | (lengths > upper_bound)]
                
                # Pattern anomalies (simple)
                has_numbers = series.str.contains(r'\d', na=False)
                has_special = series.str.contains(r'[^a-zA-Z0-9\s]', na=False)
                
                number_ratio = has_numbers.mean()
                special_ratio = has_special.mean()
                
                anomalies[col] = {
                    "length_anomalies": len(length_anomalies),
                    "avg_length": lengths.mean(),
                    "length_std": lengths.std(),
                    "number_content_ratio": round(number_ratio, 3),
                    "special_char_ratio": round(special_ratio, 3)
                }
            except:
                continue
        
        return anomalies
    
    def get_column_types(self) -> Dict[str, str]:
        """
        Get data types for each column.
        
        Returns:
            Dict mapping column names to their data types
        """
        return self.df.dtypes.astype(str).to_dict()
    
    def get_missing_values(self) -> Dict[str, Any]:
        """
        Analyze missing values in the dataset.
        
        Returns:
            Dict containing missing value statistics
        """
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        total_cells = self.df.shape[0] * self.df.shape[1]
        total_missing = missing_counts.sum()
        total_missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        
        return {
            "total_missing": int(total_missing),
            "total_missing_percentage": float(total_missing_percentage),
            "column_missing": {k: float(v) for k, v in missing_percentages.to_dict().items()},
            "missing_by_column": {k: int(v) for k, v in missing_counts.to_dict().items()},
            "missing_percentages": {k: float(v) for k, v in missing_percentages.to_dict().items()},
            "columns_with_missing": missing_counts[missing_counts > 0].index.tolist()
        }
    
    def get_numeric_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summary for numeric columns.
        
        Returns:
            Dict containing statistics for each numeric column
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        stats = {}
        
        for col in numeric_cols:
            series = self.df[col].dropna()
            if len(series) > 0:
                mean_val = series.mean()
                median_val = series.median()
                std_val = series.std()
                min_val = series.min()
                max_val = series.max()
                q25_val = series.quantile(0.25)
                q75_val = series.quantile(0.75)
                
                # Convert NaN to None for JSON serialization
                stats[col] = {
                    "mean": float(mean_val) if not pd.isna(mean_val) else 0.0,
                    "median": float(median_val) if not pd.isna(median_val) else 0.0,
                    "std": float(std_val) if not pd.isna(std_val) else 0.0,
                    "min": float(min_val) if not pd.isna(min_val) else 0.0,
                    "max": float(max_val) if not pd.isna(max_val) else 0.0,
                    "q25": float(q25_val) if not pd.isna(q25_val) else 0.0,
                    "q75": float(q75_val) if not pd.isna(q75_val) else 0.0,
                    "unique_values": int(series.nunique()),
                    "zeros": int((series == 0).sum()),
                    "negatives": int((series < 0).sum())
                }
        
        return stats
    
    def get_categorical_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for categorical/text columns.
        
        Returns:
            Dict containing statistics for each categorical column
        """
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        stats = {}
        
        for col in categorical_cols:
            series = self.df[col].dropna()
            if len(series) > 0:
                value_counts = series.value_counts()
                stats[col] = {
                    "unique_values": int(series.nunique()),
                    "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "least_frequent": str(value_counts.index[-1]) if len(value_counts) > 0 else None,
                    "least_frequent_count": int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                    "top_5_values": {str(k): int(v) for k, v in value_counts.head(5).to_dict().items()},
                    "empty_strings": int((series == "").sum()),
                    "avg_length": float(series.astype(str).str.len().mean())
                }
        
        return stats
    
    def detect_data_quality_issues(self) -> Dict[str, List[str]]:
        """
        Detect potential data quality issues.
        
        Returns:
            Dict containing lists of issues by category
        """
        issues = {
            "high_missing_columns": [],
            "high_cardinality_columns": [],
            "potential_duplicates": [],
            "mixed_types": [],
            "outlier_candidates": []
        }
        
        # High missing value columns (>50%)
        missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
        issues["high_missing_columns"] = missing_pct[missing_pct > 50].index.tolist()
        
        # High cardinality columns (>95% unique for categorical)
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_pct = (self.df[col].nunique() / len(self.df)) * 100
            if unique_pct > 95:
                issues["high_cardinality_columns"].append(col)
        
        # Check for potential duplicates
        if self.df.duplicated().any():
            issues["potential_duplicates"].append(f"{self.df.duplicated().sum()} duplicate rows found")
        
        # Mixed types (basic check for numeric columns with objects)
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    pd.to_numeric(self.df[col], errors='raise')
                except:
                    # Check if it's mixed numeric/text
                    sample_values = self.df[col].dropna().astype(str).head(100)
                    numeric_count = sum(1 for val in sample_values if val.replace('.', '').replace('-', '').isdigit())
                    if 0.1 < numeric_count / len(sample_values) < 0.9:
                        issues["mixed_types"].append(col)
        
        return issues
    
    def generate_profile(self) -> Dict[str, Any]:
        """
        Generate a comprehensive profile of the dataset.
        
        Returns:
            Dict containing complete dataset profile
        """
        return {
            "basic_info": self.get_basic_info(),
            "column_types": self.get_column_types(),
            "missing_values": self.get_missing_values(),
            "numeric_statistics": self.get_numeric_statistics(),
            "categorical_statistics": self.get_categorical_statistics(),
            "data_quality_issues": self.detect_data_quality_issues(),
            "sample_data": self.df.head(5).to_dict('records')
        }
    
    def generate_detailed_profile(self) -> Dict[str, Any]:
        """
        Generate a detailed profile with additional metrics.
        
        Returns:
            Dict containing detailed dataset profile
        """
        profile = self.generate_profile()
        
        # Add correlation matrix for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = self.df[numeric_cols].corr()
            # Convert to native Python types
            correlations = {}
            for col1 in correlation_matrix.columns:
                correlations[col1] = {}
                for col2 in correlation_matrix.columns:
                    correlations[col1][col2] = float(correlation_matrix.loc[col1, col2])
            profile["correlations"] = correlations
        
        # Add data type recommendations
        profile["type_recommendations"] = self._suggest_data_types()
        
        return profile
    
    def _suggest_data_types(self) -> Dict[str, str]:
        """
        Suggest optimal data types for columns.
        
        Returns:
            Dict mapping column names to suggested data types
        """
        suggestions = {}
        
        for col in self.df.columns:
            current_type = str(self.df[col].dtype)
            
            if current_type == 'object':
                # Check if it could be numeric
                try:
                    pd.to_numeric(self.df[col], errors='raise')
                    suggestions[col] = "numeric"
                except:
                    # Check if it could be datetime
                    try:
                        pd.to_datetime(self.df[col], errors='raise')
                        suggestions[col] = "datetime"
                    except:
                        # Check if it could be category
                        unique_ratio = self.df[col].nunique() / len(self.df)
                        if unique_ratio < 0.5:
                            suggestions[col] = "category"
                        else:
                            suggestions[col] = "text"
            elif current_type in ['int64', 'float64']:
                # Check if integer could be boolean
                if self.df[col].nunique() == 2:
                    suggestions[col] = "boolean"
                else:
                    suggestions[col] = "keep_current"
            else:
                suggestions[col] = "keep_current"
        
        return suggestions

# Compatibility alias for backward compatibility
DataProfiler = AIDataProfiler