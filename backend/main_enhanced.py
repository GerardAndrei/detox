"""
Enhanced main.py with performance optimizations and background processing
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import json
import uuid
import os
from typing import Dict, Any, Optional
from pathlib import Path
import tempfile
import uvicorn
from pydantic import BaseModel
import chardet
import asyncio
from contextlib import asynccontextmanager
import aiofiles
import io

# Import our optimized modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cleaner.profiling import DataProfiler
from cleaner.cleaning import DataCleaner
from cleaner.outlier import OutlierDetector
from cleaner.reporting import generate_cleaning_report
from cleaner.optimized_processor import OptimizedDataProcessor
from cleaner.caching import cache_manager, cached

def clean_for_json(obj):
    """
    Recursively clean an object to make it JSON-serializable by handling NaN values.
    """
    if isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif pd.isna(obj) or (isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj))):
        return None
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif hasattr(obj, 'item'):  # numpy scalar
        val = obj.item()
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return val
    else:
        return obj

# Background task storage
background_tasks_storage: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cache_manager.clear_expired()  # Clean up old cache on startup
    
    # Background task to clean cache periodically
    async def cache_cleanup():
        while True:
            await asyncio.sleep(3600)  # Every hour
            cache_manager.clear_expired()
    
    cleanup_task = asyncio.create_task(cache_cleanup())
    
    yield
    
    # Shutdown
    cleanup_task.cancel()

app = FastAPI(
    title="DETOX API - Enhanced", 
    description="High-Performance Data Cleaning and Analysis API", 
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for uploaded files and their data
uploaded_files: Dict[str, Dict[str, Any]] = {}
cleaned_files: Dict[str, pd.DataFrame] = {}

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize optimized processor
processor = OptimizedDataProcessor(chunk_size=10000)

class CleaningOptions(BaseModel):
    handle_missing: bool = True
    remove_duplicates: bool = True
    outlier_method: str = "iqr"  # "iqr" or "zscore"
    outlier_threshold: float = 1.5

class ProcessingStatus(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "DETOX Enhanced API is running!", 
        "status": "healthy",
        "version": "2.0.0",
        "features": ["streaming", "caching", "background_processing", "parallel_analysis"]
    }

@cached(ttl=1800)  # Cache for 30 minutes
def get_file_profile_cached(file_path: str, file_size: int) -> Dict[str, Any]:
    """Cached version of file profiling"""
    df = pd.read_csv(file_path)
    profiler = DataProfiler(df)
    return profiler.generate_profile()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Enhanced upload with caching and streaming analysis"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Stream file content to disk
        temp_path = UPLOAD_DIR / f"{file_id}.csv"
        
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        file_size = len(content)
        
        # Check cache first
        cache_key = cache_manager.generate_cache_key(f"{file.filename}:{file_size}")
        cached_profile = cache_manager.get(cache_key)
        
        if cached_profile:
            # Load data into memory
            df = pd.read_csv(temp_path)
            uploaded_files[file_id] = {
                "filename": file.filename,
                "path": str(temp_path),
                "dataframe": df,
                "profile": cached_profile,
                "upload_time": pd.Timestamp.now().isoformat(),
                "cached": True
            }
            
            return clean_for_json({
                "file_id": file_id,
                "filename": file.filename,
                "profile": cached_profile,
                "message": "File uploaded successfully (cached profile)",
                "cached": True
            })
        
        # Detect encoding efficiently
        with open(temp_path, 'rb') as f:
            raw_sample = f.read(10000)  # Read only first 10KB for detection
            encoding_result = chardet.detect(raw_sample)
            detected_encoding = encoding_result.get('encoding', 'utf-8')
        
        # Load and profile data
        try:
            df = pd.read_csv(temp_path, encoding=detected_encoding)
        except (UnicodeDecodeError, pd.errors.ParserError):
            # Fallback encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(temp_path, encoding=encoding)
                    break
                except:
                    continue
            else:
                raise HTTPException(status_code=400, detail="Unable to decode file")
        
        # Generate profile with parallel processing
        profiler = DataProfiler(df)
        if hasattr(processor, 'parallel_column_analysis'):
            profile = await processor.parallel_column_analysis(df)
            # Convert to expected format
            profile = profiler.generate_profile()  # Fallback to regular profiling
        else:
            profile = profiler.generate_profile()
        
        # Cache the profile
        cache_manager.set(cache_key, profile, ttl=1800)
        
        # Store file information
        uploaded_files[file_id] = {
            "filename": file.filename,
            "path": str(temp_path),
            "dataframe": df,
            "profile": profile,
            "upload_time": pd.Timestamp.now().isoformat(),
            "cached": False
        }
        
        return clean_for_json({
            "file_id": file_id,
            "filename": file.filename,
            "profile": profile,
            "message": "File uploaded and analyzed successfully",
            "cached": False
        })
        
    except HTTPException:
        raise
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def background_clean_dataset(file_id: str, options: CleaningOptions, task_id: str):
    """Background task for dataset cleaning"""
    try:
        # Update status
        background_tasks_storage[task_id] = {
            "status": "processing",
            "progress": 0.0,
            "message": "Starting data cleaning..."
        }
        
        if file_id not in uploaded_files:
            background_tasks_storage[task_id] = {
                "status": "failed",
                "progress": 0.0,
                "message": "File not found"
            }
            return
        
        df = uploaded_files[file_id]["dataframe"].copy()
        original_shape = df.shape
        
        # Initialize cleaner
        cleaner = DataCleaner(df)
        cleaning_steps = []
        
        # Progress updates
        background_tasks_storage[task_id]["progress"] = 25.0
        background_tasks_storage[task_id]["message"] = "Handling missing values..."
        
        if options.handle_missing:
            df = cleaner.handle_missing_values()
            cleaning_steps.append("Handled missing values")
        
        background_tasks_storage[task_id]["progress"] = 50.0
        background_tasks_storage[task_id]["message"] = "Removing duplicates..."
        
        if options.remove_duplicates:
            df = cleaner.remove_duplicates()
            cleaning_steps.append("Removed duplicates")
        
        background_tasks_storage[task_id]["progress"] = 75.0
        background_tasks_storage[task_id]["message"] = "Detecting outliers..."
        
        # Enhanced outlier detection
        outlier_detector = OutlierDetector(df)
        if options.outlier_method == "iqr":
            outliers_info = outlier_detector.detect_outliers_iqr(threshold=options.outlier_threshold)
        else:
            outliers_info = outlier_detector.detect_outliers_zscore(threshold=options.outlier_threshold)
        
        # Store cleaned data
        cleaned_files[file_id] = df
        
        background_tasks_storage[task_id]["progress"] = 90.0
        background_tasks_storage[task_id]["message"] = "Generating report..."
        
        # Generate cleaning report
        report = generate_cleaning_report(
            original_df=uploaded_files[file_id]["dataframe"],
            cleaned_df=df,
            cleaning_steps=cleaning_steps,
            outliers_info=outliers_info
        )
        
        # Complete
        background_tasks_storage[task_id] = {
            "status": "completed",
            "progress": 100.0,
            "message": "Dataset cleaned successfully",
            "result": clean_for_json({
                "file_id": file_id,
                "original_shape": original_shape,
                "cleaned_shape": df.shape,
                "cleaning_steps": cleaning_steps,
                "outliers_detected": len(outliers_info) if outliers_info else 0,
                "report": report,
                "preview": df.head(10).to_dict('records')
            })
        }
        
    except Exception as e:
        background_tasks_storage[task_id] = {
            "status": "failed",
            "progress": 0.0,
            "message": f"Error cleaning dataset: {str(e)}"
        }

@app.post("/clean/{file_id}/async")
async def clean_dataset_async(file_id: str, options: CleaningOptions, background_tasks: BackgroundTasks):
    """Start dataset cleaning as background task"""
    task_id = str(uuid.uuid4())
    
    background_tasks_storage[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Task queued for processing..."
    }
    
    background_tasks.add_task(background_clean_dataset, file_id, options, task_id)
    
    return {
        "task_id": task_id,
        "message": "Cleaning task started",
        "status_url": f"/task/{task_id}/status"
    }

@app.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    """Get status of background task"""
    if task_id not in background_tasks_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return background_tasks_storage[task_id]

@app.post("/clean/{file_id}")
async def clean_dataset(file_id: str, options: CleaningOptions):
    """Synchronous dataset cleaning (original endpoint)"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        df = uploaded_files[file_id]["dataframe"].copy()
        original_shape = df.shape
        
        # Initialize cleaner
        cleaner = DataCleaner(df)
        cleaning_steps = []
        
        if options.handle_missing:
            df = cleaner.handle_missing_values()
            cleaning_steps.append("Handled missing values")
        
        if options.remove_duplicates:
            df = cleaner.remove_duplicates()
            cleaning_steps.append("Removed duplicates")
        
        # Detect and handle outliers
        outlier_detector = OutlierDetector(df)
        if options.outlier_method == "iqr":
            outliers_info = outlier_detector.detect_outliers_iqr(threshold=options.outlier_threshold)
        else:
            outliers_info = outlier_detector.detect_outliers_zscore(threshold=options.outlier_threshold)
        
        # Store cleaned data
        cleaned_files[file_id] = df
        
        # Generate cleaning report
        report = generate_cleaning_report(
            original_df=uploaded_files[file_id]["dataframe"],
            cleaned_df=df,
            cleaning_steps=cleaning_steps,
            outliers_info=outliers_info
        )
        
        return clean_for_json({
            "file_id": file_id,
            "original_shape": original_shape,
            "cleaned_shape": df.shape,
            "cleaning_steps": cleaning_steps,
            "outliers_detected": len(outliers_info) if outliers_info else 0,
            "report": report,
            "preview": df.head(10).to_dict('records'),
            "message": "Dataset cleaned successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning dataset: {str(e)}")

@app.get("/download/{file_id}")
async def download_cleaned_file(file_id: str):
    """Enhanced download with streaming"""
    try:
        if file_id not in cleaned_files:
            raise HTTPException(status_code=404, detail="Cleaned file not found")
        
        df = cleaned_files[file_id]
        filename = uploaded_files[file_id]["filename"]
        
        # Stream CSV data
        def generate_csv():
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            for line in output:
                yield line
        
        clean_filename = f"cleaned_{filename}"
        
        return StreamingResponse(
            generate_csv(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={clean_filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

# Keep existing endpoints...
@app.get("/profile/{file_id}")
async def get_detailed_profile(file_id: str):
    """Get detailed dataset profile"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        df = uploaded_files[file_id]["dataframe"]
        profiler = DataProfiler(df)
        detailed_profile = profiler.generate_detailed_profile()
        
        return clean_for_json({
            "file_id": file_id,
            "profile": detailed_profile
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating profile: {str(e)}")

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete uploaded file and its data"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Remove file from disk
        file_path = Path(uploaded_files[file_id]["path"])
        if file_path.exists():
            os.remove(file_path)
        
        # Remove from memory
        del uploaded_files[file_id]
        if file_id in cleaned_files:
            del cleaned_files[file_id]
        
        return {"message": "File deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main_enhanced:app", host="0.0.0.0", port=8000, reload=True)