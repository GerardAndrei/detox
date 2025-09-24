"""
DETOX Data Cleaning Library

A comprehensive data cleaning and analysis library for pandas DataFrames.
Includes profiling, cleaning, outlier detection, and reporting capabilities.
"""

__version__ = "1.0.0"
__author__ = "DETOX Team"

from .profiling import DataProfiler
from .cleaning import DataCleaner
from .outlier import OutlierDetector
from .reporting import generate_cleaning_report

__all__ = [
    "DataProfiler",
    "DataCleaner", 
    "OutlierDetector",
    "generate_cleaning_report"
]