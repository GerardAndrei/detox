from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

# Import our custom cleaning modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cleaner.profiling import DataProfiler
from cleaner.cleaning import DataCleaner
from cleaner.outlier import OutlierDetector
from cleaner.reporting import generate_cleaning_report

app = FastAPI(title="DETOX API", description="Data Cleaning and Analysis API", version="1.0.0")

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

class CleaningOptions(BaseModel):
    handle_missing: bool = True
    remove_duplicates: bool = True
    outlier_method: str = "iqr"  # "iqr" or "zscore"
    outlier_threshold: float = 1.5

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "DETOX API is running!", "status": "healthy"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and analyze CSV file"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Read the file content
        content = await file.read()
        
        # Create a temporary file to save the upload
        temp_path = UPLOAD_DIR / f"{file_id}.csv"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Load data into pandas DataFrame with automatic encoding detection
        try:
            # Detect encoding first
            with open(temp_path, 'rb') as f:
                raw_data = f.read()
                encoding_result = chardet.detect(raw_data)
                detected_encoding = encoding_result['encoding']
                confidence = encoding_result['confidence']
            
            # Try detected encoding first if confidence is high
            if detected_encoding and confidence > 0.7:
                try:
                    df = pd.read_csv(temp_path, encoding=detected_encoding)
                except (UnicodeDecodeError, pd.errors.ParserError):
                    # Fall back to manual encoding attempts
                    df = None
            else:
                df = None
            
            # If detection failed or confidence was low, try common encodings
            if df is None:
                encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(temp_path, encoding=encoding)
                        break
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
                
                if df is None:
                    os.remove(temp_path)
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unable to read CSV file with any supported encoding. Detected: {detected_encoding} (confidence: {confidence:.2f})"
                    )
                    
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        # Profile the data
        profiler = DataProfiler(df)
        profile = profiler.generate_profile()
        
        # Store file information
        uploaded_files[file_id] = {
            "filename": file.filename,
            "path": str(temp_path),
            "dataframe": df,
            "profile": profile,
            "upload_time": pd.Timestamp.now().isoformat()
        }
        
        return clean_for_json({
            "file_id": file_id,
            "filename": file.filename,
            "profile": profile,
            "message": "File uploaded successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/clean/{file_id}")
async def clean_dataset(file_id: str, options: CleaningOptions):
    """Clean dataset with specified options"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        df = uploaded_files[file_id]["dataframe"].copy()
        original_shape = df.shape
        
        # Initialize cleaner
        cleaner = DataCleaner(df)
        
        # Apply cleaning operations
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
    """Download cleaned dataset as CSV"""
    try:
        if file_id not in cleaned_files:
            raise HTTPException(status_code=404, detail="Cleaned file not found")
        
        df = cleaned_files[file_id]
        filename = uploaded_files[file_id]["filename"]
        
        # Create a temporary file for download
        temp_path = UPLOAD_DIR / f"{file_id}_cleaned.csv"
        df.to_csv(temp_path, index=False)
        
        # Return file response
        clean_filename = f"cleaned_{filename}"
        return FileResponse(
            path=temp_path,
            filename=clean_filename,
            media_type="text/csv"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

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
        
        # Remove cleaned file if exists
        cleaned_path = UPLOAD_DIR / f"{file_id}_cleaned.csv"
        if cleaned_path.exists():
            os.remove(cleaned_path)
        
        # Remove from memory
        del uploaded_files[file_id]
        if file_id in cleaned_files:
            del cleaned_files[file_id]
        
        return {"message": "File deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@app.get("/ai-insights/{file_id}")
async def get_ai_insights(file_id: str):
    """Get AI-powered insights and recommendations for the dataset"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        df = uploaded_files[file_id]["dataframe"]
        profiler = DataProfiler(df)
        
        # Generate enhanced AI profile
        if hasattr(profiler, 'generate_enhanced_profile'):
            enhanced_profile = profiler.generate_enhanced_profile()
        else:
            enhanced_profile = profiler.generate_profile()
            enhanced_profile["ai_insights"] = {
                "message": "AI insights available with enhanced profiling",
                "ml_available": profiler.ml_available
            }
        
        return clean_for_json({
            "file_id": file_id,
            "ai_insights": enhanced_profile.get("ai_insights", {}),
            "ml_recommendations": enhanced_profile.get("ml_recommendations", {}),
            "data_quality_score": enhanced_profile.get("ai_insights", {}).get("data_quality_score", None),
            "message": "AI insights generated successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI insights: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)