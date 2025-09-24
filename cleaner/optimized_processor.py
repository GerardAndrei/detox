"""
Optimized Data Processing Module

Provides memory-efficient data processing with streaming and chunked operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Iterator, Generator
import warnings
import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')

class OptimizedDataProcessor:
    """
    Memory-efficient data processor with streaming capabilities.
    """
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize with configurable chunk size for memory efficiency.
        
        Args:
            chunk_size (int): Size of chunks for processing large datasets
        """
        self.chunk_size = chunk_size
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def stream_csv_analysis(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Stream CSV file analysis in chunks to handle large files efficiently.
        
        Args:
            file_path (str): Path to the CSV file
            
        Yields:
            Dict containing analysis results for each chunk
        """
        chunk_reader = pd.read_csv(file_path, chunksize=self.chunk_size)
        
        total_rows = 0
        column_info = {}
        
        for chunk_idx, chunk in enumerate(chunk_reader):
            chunk_analysis = {
                'chunk_id': chunk_idx,
                'chunk_size': len(chunk),
                'memory_usage': chunk.memory_usage(deep=True).sum(),
                'columns': list(chunk.columns),
                'dtypes': chunk.dtypes.to_dict()
            }
            
            # Update running statistics
            total_rows += len(chunk)
            
            # Memory cleanup
            gc.collect()
            
            yield chunk_analysis
    
    async def parallel_column_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze columns in parallel for faster processing.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            
        Returns:
            Dict containing column analysis results
        """
        loop = asyncio.get_event_loop()
        
        # Split columns into batches for parallel processing
        columns = list(df.columns)
        batch_size = max(1, len(columns) // 4)  # 4 parallel batches
        column_batches = [columns[i:i + batch_size] for i in range(0, len(columns), batch_size)]
        
        # Create analysis tasks
        tasks = []
        for batch in column_batches:
            task = loop.run_in_executor(
                self.executor,
                self._analyze_column_batch,
                df[batch]
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Merge results
        final_analysis = {}
        for result in results:
            final_analysis.update(result)
        
        return final_analysis
    
    def _analyze_column_batch(self, df_batch: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a batch of columns (used for parallel processing).
        
        Args:
            df_batch (pd.DataFrame): Batch of columns to analyze
            
        Returns:
            Dict containing analysis for the batch
        """
        analysis = {}
        
        for col in df_batch.columns:
            column_data = df_batch[col]
            
            analysis[col] = {
                'dtype': str(column_data.dtype),
                'null_count': column_data.isnull().sum(),
                'null_percentage': (column_data.isnull().sum() / len(column_data)) * 100,
                'unique_count': column_data.nunique(),
                'memory_usage': column_data.memory_usage(deep=True)
            }
            
            # Type-specific analysis
            if column_data.dtype in ['int64', 'float64']:
                analysis[col].update({
                    'min': column_data.min(),
                    'max': column_data.max(),
                    'mean': column_data.mean(),
                    'std': column_data.std(),
                    'median': column_data.median()
                })
            elif column_data.dtype == 'object':
                analysis[col].update({
                    'avg_length': column_data.astype(str).str.len().mean(),
                    'max_length': column_data.astype(str).str.len().max(),
                    'most_common': column_data.value_counts().head(5).to_dict()
                })
        
        return analysis
    
    def efficient_duplicate_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Memory-efficient duplicate detection using hash-based approach.
        
        Args:
            df (pd.DataFrame): DataFrame to check for duplicates
            
        Returns:
            Dict containing duplicate information
        """
        # Use hash-based duplicate detection for large datasets
        if len(df) > 50000:
            # Create hash for each row
            df_hash = pd.util.hash_pandas_object(df, index=False)
            duplicate_mask = df_hash.duplicated()
            duplicate_count = duplicate_mask.sum()
        else:
            # Standard duplicate detection for smaller datasets
            duplicate_mask = df.duplicated()
            duplicate_count = duplicate_mask.sum()
        
        return {
            'duplicate_count': int(duplicate_count),
            'duplicate_percentage': (duplicate_count / len(df)) * 100,
            'duplicate_indices': duplicate_mask[duplicate_mask].index.tolist()[:100]  # Limit to first 100
        }
    
    def optimized_outlier_detection(self, df: pd.DataFrame, method: str = 'iqr', 
                                  sample_size: int = 10000) -> Dict[str, Any]:
        """
        Memory-efficient outlier detection with sampling for large datasets.
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            method (str): Detection method ('iqr' or 'zscore')
            sample_size (int): Sample size for large datasets
            
        Returns:
            Dict containing outlier information
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_columns:
            column_data = df[col].dropna()
            
            # Use sampling for large datasets
            if len(column_data) > sample_size:
                column_data = column_data.sample(n=sample_size, random_state=42)
            
            if method == 'iqr':
                Q1 = column_data.quantile(0.25)
                Q3 = column_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
            else:  # zscore
                z_scores = np.abs((column_data - column_data.mean()) / column_data.std())
                outliers = column_data[z_scores > 3]
            
            outlier_info[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(column_data)) * 100,
                'lower_bound': lower_bound if method == 'iqr' else None,
                'upper_bound': upper_bound if method == 'iqr' else None
            }
        
        return outlier_info
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)