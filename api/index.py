from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import numpy as np
import io
import json
import uuid
import os
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import your existing modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Inline cleaner classes for Vercel compatibility
class DataCleaner:
    def remove_duplicates(self, df):
        return df.drop_duplicates()
    
    def handle_missing_values(self, df, strategy='drop'):
        if strategy == 'drop':
            return df.dropna()
        elif strategy == 'fill_mean':
            return df.fillna(df.select_dtypes(include=[np.number]).mean())
        elif strategy == 'fill_median':
            return df.fillna(df.select_dtypes(include=[np.number]).median())
        elif strategy == 'fill_mode':
            return df.fillna(df.mode().iloc[0])
        return df
    
    def standardize_data_formats(self, df):
        # Basic standardization
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        return df

class OutlierDetector:
    def detect_iqr_outliers(self, series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def detect_zscore_outliers(self, series, threshold=3):
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    def detect_isolation_forest_outliers(self, df, contamination=0.1):
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(df)
        return outliers == -1
    
    def remove_outliers_iqr(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outliers = self.detect_iqr_outliers(df[col])
            df = df[~outliers]
        return df

class DataProfiler:
    def generate_basic_profile(self, df):
        profile = {
            'shape': df.shape,
            'columns': {},
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        for col in df.columns:
            col_info = {
                'type': 'numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical',
                'unique_count': df[col].nunique(),
                'missing_count': df[col].isnull().sum()
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None,
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None
                })
            
            profile['columns'][col] = col_info
        
        return profile

class ReportGenerator:
    def generate_summary_report(self, df, profile):
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        return {
            'summary': {
                'total_rows': df.shape[0],
                'total_columns': df.shape[1],
                'total_cells': total_cells,
                'missing_cells': int(missing_cells),
                'missing_percentage': float(missing_cells / total_cells * 100) if total_cells > 0 else 0
            },
            'data_types': profile['data_types'],
            'column_summary': profile['columns']
        }

app = FastAPI(title="Detox API", version="1.0.0")

# Configure CORS for Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for data storage (in production, use a database)
uploaded_files = {}
processing_results = {}

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

@app.get("/")
async def root():
    return {"message": "Detox API is running on Vercel!"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "detox-api"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read file content
        content = await file.read()
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Store in memory (in production, use cloud storage)
        uploaded_files[file_id] = {
            'filename': file.filename,
            'data': df,
            'upload_time': pd.Timestamp.now().isoformat()
        }
        
        # Generate basic preview
        preview_data = {
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'head': df.head(10).to_dict('records'),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return {
            'file_id': file_id,
            'filename': file.filename,
            'preview': preview_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/api/process")
async def process_data(request: Dict[str, Any]):
    try:
        file_id = request.get('file_id')
        options = request.get('options', {})
        
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        df = uploaded_files[file_id]['data'].copy()
        
        # Initialize components
        cleaner = DataCleaner()
        outlier_detector = OutlierDetector()
        profiler = DataProfiler()
        reporter = ReportGenerator()
        
        results = {}
        
        # Data Cleaning
        if options.get('remove_duplicates', False):
            df_cleaned = cleaner.remove_duplicates(df)
            results['duplicates_removed'] = len(df) - len(df_cleaned)
            df = df_cleaned
        
        if options.get('handle_missing', False):
            missing_strategy = options.get('missing_strategy', 'drop')
            df = cleaner.handle_missing_values(df, strategy=missing_strategy)
            results['missing_handled'] = True
        
        if options.get('standardize_formats', False):
            df = cleaner.standardize_data_formats(df)
            results['formats_standardized'] = True
        
        # Outlier Detection
        outlier_methods = options.get('outlier_methods', [])
        outliers_detected = {}
        
        if 'iqr' in outlier_methods:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                outliers = outlier_detector.detect_iqr_outliers(df[col])
                outliers_detected[f'{col}_iqr'] = outliers.sum()
        
        if 'isolation_forest' in outlier_methods:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                outliers = outlier_detector.detect_isolation_forest_outliers(numeric_df)
                outliers_detected['isolation_forest'] = outliers.sum()
        
        if 'zscore' in outlier_methods:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                outliers = outlier_detector.detect_zscore_outliers(df[col])
                outliers_detected[f'{col}_zscore'] = outliers.sum()
        
        # Remove outliers if requested
        if options.get('remove_outliers', False) and outlier_methods:
            if 'iqr' in outlier_methods:
                df = outlier_detector.remove_outliers_iqr(df)
        
        # Generate profile
        profile = profiler.generate_basic_profile(df)
        
        # Generate report
        report = reporter.generate_summary_report(df, profile)
        
        # Store results
        processing_results[file_id] = {
            'processed_data': df,
            'results': results,
            'outliers': outliers_detected,
            'profile': profile,
            'report': report,
            'processing_time': pd.Timestamp.now().isoformat()
        }
        
        return {
            'file_id': file_id,
            'results': results,
            'outliers_detected': outliers_detected,
            'final_shape': df.shape,
            'profile_summary': {
                'total_columns': len(profile.get('columns', {})),
                'numeric_columns': len([col for col, info in profile.get('columns', {}).items() if info.get('type') == 'numeric']),
                'categorical_columns': len([col for col, info in profile.get('columns', {}).items() if info.get('type') == 'categorical'])
            },
            'report': report
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.get("/api/download/{file_id}")
async def download_processed_data(file_id: str):
    try:
        if file_id not in processing_results:
            raise HTTPException(status_code=404, detail="Processed data not found")
        
        df = processing_results[file_id]['processed_data']
        
        # Convert to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        # Create streaming response
        def generate():
            yield csv_content
        
        return StreamingResponse(
            io.BytesIO(csv_content.encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=cleaned_data_{file_id}.csv"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading data: {str(e)}")

@app.get("/api/profile/{file_id}")
async def get_data_profile(file_id: str):
    try:
        if file_id not in processing_results:
            raise HTTPException(status_code=404, detail="Processed data not found")
        
        return processing_results[file_id]['profile']
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving profile: {str(e)}")

# Vercel serverless function handler
from mangum import Mangum
handler = Mangum(app)