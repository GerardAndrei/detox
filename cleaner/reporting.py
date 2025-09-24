"""
Reporting Module

Provides comprehensive reporting capabilities for data cleaning operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def generate_cleaning_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                           cleaning_steps: List[str], outliers_info: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive cleaning report comparing original and cleaned datasets.
    
    Args:
        original_df (pd.DataFrame): Original dataset
        cleaned_df (pd.DataFrame): Cleaned dataset
        cleaning_steps (List[str]): List of cleaning operations performed
        outliers_info (Dict, optional): Information about outliers detected
    
    Returns:
        Dict containing comprehensive cleaning report
    """
    
    report = {
        "metadata": {
            "report_generated": datetime.now().isoformat(),
            "cleaning_steps_applied": len(cleaning_steps),
            "outliers_analyzed": outliers_info is not None
        },
        "dataset_comparison": _compare_datasets(original_df, cleaned_df),
        "cleaning_operations": cleaning_steps,
        "data_quality_improvement": _calculate_quality_improvement(original_df, cleaned_df),
        "column_analysis": _analyze_column_changes(original_df, cleaned_df),
        "recommendations": _generate_recommendations(original_df, cleaned_df, cleaning_steps)
    }
    
    if outliers_info:
        report["outlier_analysis"] = _summarize_outliers(outliers_info)
    
    return report

def _compare_datasets(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare original and cleaned datasets.
    
    Args:
        original_df (pd.DataFrame): Original dataset
        cleaned_df (pd.DataFrame): Cleaned dataset
    
    Returns:
        Dict containing dataset comparison
    """
    original_missing = original_df.isnull().sum().sum()
    cleaned_missing = cleaned_df.isnull().sum().sum()
    original_duplicates = original_df.duplicated().sum()
    cleaned_duplicates = cleaned_df.duplicated().sum()
    
    return {
        "shape_change": {
            "original": list(original_df.shape),
            "cleaned": list(cleaned_df.shape),
            "rows_removed": int(original_df.shape[0] - cleaned_df.shape[0]),
            "columns_removed": int(original_df.shape[1] - cleaned_df.shape[1]),
            "data_retention_rate": float((cleaned_df.shape[0] / original_df.shape[0]) * 100 if original_df.shape[0] > 0 else 0)
        },
        "missing_values": {
            "original_count": int(original_missing),
            "cleaned_count": int(cleaned_missing),
            "reduction": int(original_missing - cleaned_missing),
            "reduction_percentage": float(((original_missing - cleaned_missing) / original_missing) * 100 if original_missing > 0 else 0)
        },
        "duplicates": {
            "original_count": int(original_duplicates),
            "cleaned_count": int(cleaned_duplicates),
            "removed": int(original_duplicates - cleaned_duplicates)
        },
        "memory_usage": {
            "original_mb": float(original_df.memory_usage(deep=True).sum() / (1024 * 1024)),
            "cleaned_mb": float(cleaned_df.memory_usage(deep=True).sum() / (1024 * 1024)),
            "reduction_mb": float((original_df.memory_usage(deep=True).sum() - cleaned_df.memory_usage(deep=True).sum()) / (1024 * 1024))
        }
    }

def _calculate_quality_improvement(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate data quality improvement metrics.
    
    Args:
        original_df (pd.DataFrame): Original dataset
        cleaned_df (pd.DataFrame): Cleaned dataset
    
    Returns:
        Dict containing quality improvement metrics
    """
    # Calculate completeness (percentage of non-null values)
    original_completeness = ((original_df.notna().sum().sum()) / (original_df.shape[0] * original_df.shape[1])) * 100 if original_df.shape[0] > 0 and original_df.shape[1] > 0 else 0
    cleaned_completeness = ((cleaned_df.notna().sum().sum()) / (cleaned_df.shape[0] * cleaned_df.shape[1])) * 100 if cleaned_df.shape[0] > 0 and cleaned_df.shape[1] > 0 else 0
    
    # Calculate uniqueness (percentage of unique rows)
    original_uniqueness = ((original_df.shape[0] - original_df.duplicated().sum()) / original_df.shape[0]) * 100 if original_df.shape[0] > 0 else 0
    cleaned_uniqueness = ((cleaned_df.shape[0] - cleaned_df.duplicated().sum()) / cleaned_df.shape[0]) * 100 if cleaned_df.shape[0] > 0 else 0
    
    # Calculate consistency score (simplified metric based on data types and patterns)
    original_consistency = _calculate_consistency_score(original_df)
    cleaned_consistency = _calculate_consistency_score(cleaned_df)
    
    # Ensure no NaN values in calculations
    original_completeness = 0.0 if pd.isna(original_completeness) else original_completeness
    cleaned_completeness = 0.0 if pd.isna(cleaned_completeness) else cleaned_completeness
    original_uniqueness = 0.0 if pd.isna(original_uniqueness) else original_uniqueness
    cleaned_uniqueness = 0.0 if pd.isna(cleaned_uniqueness) else cleaned_uniqueness
    original_consistency = 0.0 if pd.isna(original_consistency) else original_consistency
    cleaned_consistency = 0.0 if pd.isna(cleaned_consistency) else cleaned_consistency
    
    return {
        "completeness": {
            "original": float(round(original_completeness, 2)),
            "cleaned": float(round(cleaned_completeness, 2)),
            "improvement": float(round(cleaned_completeness - original_completeness, 2))
        },
        "uniqueness": {
            "original": float(round(original_uniqueness, 2)),
            "cleaned": float(round(cleaned_uniqueness, 2)),
            "improvement": float(round(cleaned_uniqueness - original_uniqueness, 2))
        },
        "consistency": {
            "original": float(round(original_consistency, 2)),
            "cleaned": float(round(cleaned_consistency, 2)),
            "improvement": float(round(cleaned_consistency - original_consistency, 2))
        },
        "overall_quality_score": {
            "original": float(round((original_completeness + original_uniqueness + original_consistency) / 3, 2)),
            "cleaned": float(round((cleaned_completeness + cleaned_uniqueness + cleaned_consistency) / 3, 2))
        }
    }

def _calculate_consistency_score(df: pd.DataFrame) -> float:
    """
    Calculate a consistency score based on data type uniformity and pattern consistency.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    
    Returns:
        float: Consistency score (0-100)
    """
    if df.empty:
        return 0.0
    
    consistency_scores = []
    
    for col in df.columns:
        # Check for mixed types in object columns
        if df[col].dtype == 'object':
            non_null_values = df[col].dropna()
            if len(non_null_values) == 0:
                consistency_scores.append(100.0)
                continue
            
            # Check if values can be consistently converted to numeric
            numeric_convertible = 0
            for val in non_null_values.sample(min(100, len(non_null_values))):
                try:
                    float(str(val))
                    numeric_convertible += 1
                except:
                    pass
            
            # If most values are numeric-convertible, consistency is high
            numeric_ratio = numeric_convertible / min(100, len(non_null_values))
            if numeric_ratio > 0.9 or numeric_ratio < 0.1:
                consistency_scores.append(100.0)
            else:
                consistency_scores.append(50.0)  # Mixed types
        else:
            # Non-object columns are considered consistent
            consistency_scores.append(100.0)
    
    return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0

def _analyze_column_changes(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze changes in individual columns.
    
    Args:
        original_df (pd.DataFrame): Original dataset
        cleaned_df (pd.DataFrame): Cleaned dataset
    
    Returns:
        Dict containing column-wise analysis
    """
    column_analysis = {}
    
    # Analyze common columns
    common_columns = set(original_df.columns) & set(cleaned_df.columns)
    
    for col in common_columns:
        original_series = original_df[col]
        cleaned_series = cleaned_df[col]
        
        analysis = {
            "original_stats": _get_column_stats(original_series),
            "cleaned_stats": _get_column_stats(cleaned_series),
            "changes": {
                "missing_values_removed": int(original_series.isnull().sum() - cleaned_series.isnull().sum()),
                "unique_values_change": int(cleaned_series.nunique() - original_series.nunique()),
                "data_type_changed": str(original_series.dtype) != str(cleaned_series.dtype)
            }
        }
        
        column_analysis[col] = analysis
    
    # Identify removed columns
    removed_columns = set(original_df.columns) - set(cleaned_df.columns)
    if removed_columns:
        column_analysis["removed_columns"] = {
            "count": len(removed_columns),
            "names": list(removed_columns),
            "reasons": ["High missing value percentage" for _ in removed_columns]  # Simplified
        }
    
    return column_analysis

def _get_column_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Get statistics for a pandas Series.
    
    Args:
        series (pd.Series): Series to analyze
    
    Returns:
        Dict containing series statistics
    """
    stats = {
        "count": int(len(series)),
        "non_null_count": int(series.notna().sum()),
        "null_count": int(series.isnull().sum()),
        "unique_count": int(series.nunique()),
        "data_type": str(series.dtype)
    }
    
    if series.dtype in ['int64', 'float64']:
        non_null_series = series.dropna()
        if len(non_null_series) > 0:
            stats.update({
                "mean": float(non_null_series.mean()),
                "std": float(non_null_series.std()),
                "min": float(non_null_series.min()),
                "max": float(non_null_series.max()),
                "median": float(non_null_series.median())
            })
    
    return stats

def _summarize_outliers(outliers_info: Dict) -> Dict[str, Any]:
    """
    Summarize outlier detection results.
    
    Args:
        outliers_info (Dict): Outlier detection results
    
    Returns:
        Dict containing outlier summary
    """
    if not outliers_info:
        return {"total_outliers": 0, "columns_with_outliers": 0}
    
    total_outliers = sum(info.get("outlier_count", 0) for info in outliers_info.values())
    columns_with_outliers = sum(1 for info in outliers_info.values() if info.get("outlier_count", 0) > 0)
    
    outlier_summary = {
        "total_outliers": int(total_outliers),
        "columns_analyzed": int(len(outliers_info)),
        "columns_with_outliers": int(columns_with_outliers),
        "outliers_by_column": {
            col: {
                "count": int(info.get("outlier_count", 0)),
                "percentage": float(info.get("outlier_percentage", 0)),
                "method": str(info.get("method", "unknown"))
            }
            for col, info in outliers_info.items()
        }
    }
    
    return outlier_summary

def _generate_recommendations(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                            cleaning_steps: List[str]) -> List[str]:
    """
    Generate recommendations for further data improvement.
    
    Args:
        original_df (pd.DataFrame): Original dataset
        cleaned_df (pd.DataFrame): Cleaned dataset
        cleaning_steps (List[str]): Cleaning steps performed
    
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # Check if there are still missing values
    remaining_missing = cleaned_df.isnull().sum().sum()
    if remaining_missing > 0:
        recommendations.append(f"Consider handling the remaining {remaining_missing} missing values")
    
    # Check for high cardinality categorical columns
    categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_ratio = cleaned_df[col].nunique() / len(cleaned_df)
        if unique_ratio > 0.9:
            recommendations.append(f"Column '{col}' has high cardinality ({unique_ratio:.1%}), consider if it should be treated as an identifier")
    
    # Check for potential data type optimizations
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            # Check if it could be converted to category
            unique_ratio = cleaned_df[col].nunique() / len(cleaned_df)
            if unique_ratio < 0.5:
                recommendations.append(f"Consider converting '{col}' to categorical data type for memory optimization")
    
    # Check data retention rate
    retention_rate = (cleaned_df.shape[0] / original_df.shape[0]) * 100 if original_df.shape[0] > 0 else 0
    if retention_rate < 70:
        recommendations.append(f"Data retention rate is {retention_rate:.1f}%, verify if the cleaning was too aggressive")
    
    # Check if normalization might be needed
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = cleaned_df[col].dropna()
        if len(series) > 0:
            cv = series.std() / series.mean() if series.mean() != 0 else 0
            if cv > 1:
                recommendations.append(f"Column '{col}' has high variability, consider normalization for analysis")
    
    # General recommendations based on cleaning steps
    if "Handled missing values" not in str(cleaning_steps):
        if original_df.isnull().sum().sum() > 0:
            recommendations.append("Consider implementing missing value handling strategies")
    
    if "Removed duplicates" not in str(cleaning_steps):
        if original_df.duplicated().sum() > 0:
            recommendations.append("Consider removing duplicate rows to improve data quality")
    
    return recommendations

def generate_summary_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                          cleaning_steps: List[str]) -> str:
    """
    Generate a human-readable summary report.
    
    Args:
        original_df (pd.DataFrame): Original dataset
        cleaned_df (pd.DataFrame): Cleaned dataset
        cleaning_steps (List[str]): Cleaning steps performed
    
    Returns:
        str: Human-readable summary report
    """
    report = generate_cleaning_report(original_df, cleaned_df, cleaning_steps)
    
    summary = f"""
DETOX Data Cleaning Summary Report
Generated: {report['metadata']['report_generated']}

DATASET OVERVIEW:
- Original size: {report['dataset_comparison']['shape_change']['original'][0]:,} rows × {report['dataset_comparison']['shape_change']['original'][1]} columns
- Cleaned size: {report['dataset_comparison']['shape_change']['cleaned'][0]:,} rows × {report['dataset_comparison']['shape_change']['cleaned'][1]} columns
- Data retention: {report['dataset_comparison']['shape_change']['data_retention_rate']:.1f}%

DATA QUALITY IMPROVEMENTS:
- Completeness: {report['data_quality_improvement']['completeness']['original']:.1f}% → {report['data_quality_improvement']['completeness']['cleaned']:.1f}%
- Uniqueness: {report['data_quality_improvement']['uniqueness']['original']:.1f}% → {report['data_quality_improvement']['uniqueness']['cleaned']:.1f}%
- Overall Quality Score: {report['data_quality_improvement']['overall_quality_score']['original']:.1f}% → {report['data_quality_improvement']['overall_quality_score']['cleaned']:.1f}%

CLEANING OPERATIONS PERFORMED:
"""
    
    for i, step in enumerate(cleaning_steps, 1):
        summary += f"{i}. {step}\n"
    
    if report.get('recommendations'):
        summary += "\nRECOMMENDATIONS:\n"
        for i, rec in enumerate(report['recommendations'], 1):
            summary += f"{i}. {rec}\n"
    
    return summary