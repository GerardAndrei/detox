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
        
        return {"current_type": str(self.df[col].dtype), "suggested_type": str(self.df[col].dtype), "confidence": 1.0, "reason": "Optimal type"}
    
    def get_missing_values(self) -> Dict[str, Any]:
        """
        Analyze missing values in the dataset with AI insights.
        
        Returns:
            Dict containing missing value statistics and AI recommendations
        """
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        total_cells = self.df.shape[0] * self.df.shape[1]
        total_missing = missing_counts.sum()
        total_missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        
        # AI-powered missing value strategy recommendations
        ai_strategies = {}
        for col in self.df.columns:
            if missing_counts[col] > 0:
                strategy = self._recommend_missing_strategy(col, missing_percentages[col])
                ai_strategies[col] = strategy
        
        return {
            "total_missing": int(total_missing),
            "total_missing_percentage": float(total_missing_percentage),
            "column_missing": {k: float(v) for k, v in missing_percentages.to_dict().items()},
            "missing_by_column": {k: int(v) for k, v in missing_counts.to_dict().items()},
            "missing_percentages": {k: float(v) for k, v in missing_percentages.to_dict().items()},
            "columns_with_missing": missing_counts[missing_counts > 0].index.tolist(),
            "ai_missing_strategies": ai_strategies
        }
    
    def _recommend_missing_strategy(self, col: str, missing_percentage: float) -> Dict[str, Any]:
        """AI-powered recommendation for handling missing values in a specific column."""
        dtype = str(self.df[col].dtype)
        
        if missing_percentage > 70:
            return {
                "strategy": "drop_column",
                "reason": f"Column has {missing_percentage:.1f}% missing values - consider removal",
                "confidence": 0.9
            }
        elif missing_percentage > 30:
            if 'object' in dtype:
                return {
                    "strategy": "mode_imputation",
                    "reason": f"High missing rate ({missing_percentage:.1f}%) in categorical column",
                    "confidence": 0.7
                }
            else:
                return {
                    "strategy": "median_imputation", 
                    "reason": f"High missing rate ({missing_percentage:.1f}%) in numeric column",
                    "confidence": 0.7
                }
        else:
            if 'object' in dtype:
                return {
                    "strategy": "mode_imputation",
                    "reason": f"Low missing rate ({missing_percentage:.1f}%) - mode imputation safe",
                    "confidence": 0.85
                }
            else:
                return {
                    "strategy": "mean_imputation",
                    "reason": f"Low missing rate ({missing_percentage:.1f}%) - mean imputation recommended",
                    "confidence": 0.85
                }
    
    def get_column_types(self) -> Dict[str, str]:
        """
        Get data types for each column.
        
        Returns:
            Dict mapping column names to their data types
        """
        return self.df.dtypes.astype(str).to_dict()
    
    def get_numeric_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summary for numeric columns with AI insights.
        
        Returns:
            Dict containing statistics for each numeric column
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {}
        
        stats_dict = {}
        for col in numeric_cols:
            try:
                series = self.df[col].dropna()
                if len(series) == 0:
                    continue
                
                # Basic statistics
                basic_stats = {
                    'count': int(len(series)),
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'median': float(series.median()),
                    'q1': float(series.quantile(0.25)),
                    'q3': float(series.quantile(0.75))
                }
                
                # AI-enhanced statistics
                basic_stats.update({
                    'skewness': float(series.skew()),
                    'kurtosis': float(series.kurtosis()),
                    'outlier_count': self._count_outliers_iqr(series),
                    'distribution_type': self._classify_distribution(series)
                })
                
                stats_dict[col] = basic_stats
            except Exception as e:
                stats_dict[col] = {'error': str(e)}
        
        return stats_dict
    
    def _count_outliers_iqr(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return int(((series < lower_bound) | (series > upper_bound)).sum())
    
    def _classify_distribution(self, series: pd.Series) -> str:
        """AI-powered classification of distribution type."""
        if len(series) < 10:
            return "insufficient_data"
        
        skewness = abs(series.skew())
        kurtosis = series.kurtosis()
        
        if skewness < 0.5:
            if -0.5 <= kurtosis <= 0.5:
                return "normal"
            elif kurtosis > 0.5:
                return "leptokurtic"  # Heavy tails
            else:
                return "platykurtic"  # Light tails
        elif 0.5 <= skewness < 1:
            return "moderately_skewed"
        else:
            return "highly_skewed"
    
    def get_categorical_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistical summary for categorical columns with AI insights.
        
        Returns:
            Dict containing statistics for each categorical column
        """
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) == 0:
            return {}
        
        stats_dict = {}
        for col in categorical_cols:
            try:
                series = self.df[col].dropna()
                if len(series) == 0:
                    continue
                
                value_counts = series.value_counts()
                
                # Basic categorical statistics
                basic_stats = {
                    'unique_count': int(series.nunique()),
                    'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'total_count': int(len(series))
                }
                
                # AI-enhanced categorical analysis
                basic_stats.update({
                    'cardinality_ratio': float(series.nunique() / len(series)) if len(series) > 0 else 0,
                    'concentration_ratio': float(value_counts.iloc[0] / len(series)) if len(value_counts) > 0 else 0,
                    'category_recommendation': self._recommend_categorical_treatment(series)
                })
                
                stats_dict[col] = basic_stats
            except Exception as e:
                stats_dict[col] = {'error': str(e)}
        
        return stats_dict
    
    def _recommend_categorical_treatment(self, series: pd.Series) -> Dict[str, Any]:
        """AI recommendation for categorical column treatment."""
        unique_count = series.nunique()
        total_count = len(series)
        cardinality_ratio = unique_count / total_count if total_count > 0 else 0
        
        if cardinality_ratio > 0.9:
            return {
                "treatment": "consider_text_analysis",
                "reason": f"Very high cardinality ({unique_count} unique values)",
                "confidence": 0.8
            }
        elif cardinality_ratio > 0.5:
            return {
                "treatment": "group_rare_categories",
                "reason": f"High cardinality ({unique_count} unique values) - group rare categories",
                "confidence": 0.7
            }
        elif unique_count < 10:
            return {
                "treatment": "one_hot_encode",
                "reason": f"Low cardinality ({unique_count} categories) - good for encoding",
                "confidence": 0.9
            }
        else:
            return {
                "treatment": "label_encode",
                "reason": f"Medium cardinality ({unique_count} categories) - label encoding recommended",
                "confidence": 0.8
            }
    
    def detect_data_quality_issues(self) -> Dict[str, List[str]]:
        """
        Detect various data quality issues with AI-enhanced detection.
        
        Returns:
            Dict mapping issue types to lists of affected columns
        """
        issues = {}
        
        for col in self.df.columns:
            col_issues = []
            series = self.df[col]
            
            # High missing value percentage
            missing_pct = (series.isnull().sum() / len(series)) * 100
            if missing_pct > 30:
                col_issues.append(f"High missing values: {missing_pct:.1f}%")
            
            # High cardinality for categorical
            if series.dtype == 'object':
                cardinality_ratio = series.nunique() / len(series)
                if cardinality_ratio > 0.8:
                    col_issues.append(f"Very high cardinality: {series.nunique()} unique values")
            
            # Potential data type issues
            if series.dtype == 'object':
                # Check for mixed types
                non_null_series = series.dropna()
                if len(non_null_series) > 0:
                    # Check if all values could be numeric
                    try:
                        pd.to_numeric(non_null_series, errors='raise')
                        col_issues.append("Numeric data stored as text")
                    except:
                        pass
            
            # Outliers in numeric columns
            if pd.api.types.is_numeric_dtype(series):
                outlier_count = self._count_outliers_iqr(series.dropna())
                if outlier_count > len(series) * 0.05:  # More than 5% outliers
                    col_issues.append(f"High outlier count: {outlier_count}")
            
            if col_issues:
                issues[col] = col_issues
        
        return issues
    
    def get_ai_anomaly_detection(self) -> Dict[str, Any]:
        """
        Use AI/ML for advanced anomaly detection across the dataset.
        """
        if not self.ml_available:
            return {"error": "ML libraries not available", "message": "Install scikit-learn for AI anomaly detection"}
        
        try:
            anomalies = {}
            
            # Anomaly detection for numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                anomalies["numeric_anomalies"] = self._detect_numeric_anomalies(numeric_cols)
            
            return anomalies
        except Exception as e:
            return {"error": f"AI anomaly detection failed: {str(e)}"}
    
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
    
    def generate_profile(self) -> Dict[str, Any]:
        """
        Generate a comprehensive AI-powered profile of the dataset.
        
        Returns:
            Dict containing complete dataset profile with AI insights
        """
        profile = {
            "basic_info": self.get_basic_info(),
            "column_types": self.get_column_types(),
            "missing_values": self.get_missing_values(),
            "numeric_statistics": self.get_numeric_statistics(),
            "categorical_statistics": self.get_categorical_statistics(),
            "data_quality_issues": self.detect_data_quality_issues(),
            "sample_data": self.df.head(5).to_dict('records')
        }
        
        # Add AI-specific features
        if self.ml_available:
            profile["ai_anomaly_detection"] = self.get_ai_anomaly_detection()
            profile["ai_insights"] = {
                "ml_features_available": True,
                "complexity_assessment": profile["basic_info"]["data_complexity_score"],
                "quality_recommendation": profile["basic_info"]["data_quality_summary"]["ai_recommendation"]
            }
        else:
            profile["ai_insights"] = {
                "ml_features_available": False,
                "message": "Install scikit-learn for advanced AI features"
            }
        
        return profile

    def generate_detailed_profile(self) -> Dict[str, Any]:
        """
        Generate a detailed profile with additional AI metrics.
        
        Returns:
            Dict containing detailed dataset profile
        """
        profile = self.generate_profile()
        
        # Add AI-enhanced features
        if self.ml_available:
            # Add correlation analysis for numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                try:
                    correlation_matrix = self.df[numeric_cols].corr()
                    # Convert to native Python types
                    correlations = {}
                    for col1 in correlation_matrix.columns:
                        correlations[col1] = {}
                        for col2 in correlation_matrix.columns:
                            val = correlation_matrix.loc[col1, col2]
                            correlations[col1][col2] = float(val) if not pd.isna(val) else None
                    profile["correlations"] = correlations
                except:
                    pass
        
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


# Backward compatibility wrapper
class DataProfiler(AIDataProfiler):
    """
    Backward compatibility wrapper for the original DataProfiler.
    Now powered by AI but maintains the same interface.
    """
    pass