"""
Database Integration Module for DETOX

Provides efficient database operations for large dataset handling.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Generator
from pathlib import Path
import asyncio
import aiosqlite
import json

class DatabaseManager:
    """
    Manages database operations for efficient large dataset handling.
    """
    
    def __init__(self, db_path: str = "detox.db"):
        """
        Initialize database manager.
        
        Args:
            db_path (str): Path to SQLite database
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Files table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    file_size INTEGER,
                    row_count INTEGER,
                    column_count INTEGER,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    profile_data TEXT,
                    status TEXT DEFAULT 'uploaded'
                )
            ''')
            
            # Processing tasks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_tasks (
                    task_id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    progress REAL DEFAULT 0.0,
                    result_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES files (id)
                )
            ''')
            
            # Data chunks table for large files
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES files (id)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_status ON files(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_status ON processing_tasks(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_file ON data_chunks(file_id, chunk_index)')
            
            conn.commit()
    
    def store_file_metadata(self, file_id: str, filename: str, file_path: str, 
                          file_size: int, df_info: Dict[str, Any]) -> bool:
        """
        Store file metadata in database.
        
        Args:
            file_id: Unique file identifier
            filename: Original filename
            file_path: Path to stored file
            file_size: File size in bytes
            df_info: DataFrame information (shape, dtypes, etc.)
            
        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO files (id, filename, original_path, file_size, 
                                     row_count, column_count, profile_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    file_id,
                    filename,
                    file_path,
                    file_size,
                    df_info.get('row_count', 0),
                    df_info.get('column_count', 0),
                    json.dumps(df_info)
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error storing file metadata: {e}")
            return False
    
    def store_data_in_chunks(self, file_id: str, df: pd.DataFrame, chunk_size: int = 1000) -> bool:
        """
        Store large DataFrame in chunks for efficient retrieval.
        
        Args:
            file_id: File identifier
            df: DataFrame to store
            chunk_size: Number of rows per chunk
            
        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for chunk_index, start_idx in enumerate(range(0, len(df), chunk_size)):
                    end_idx = min(start_idx + chunk_size, len(df))
                    chunk_df = df.iloc[start_idx:end_idx]
                    
                    chunk_id = f"{file_id}_chunk_{chunk_index}"
                    chunk_json = chunk_df.to_json(orient='records')
                    
                    cursor.execute('''
                        INSERT INTO data_chunks (chunk_id, file_id, chunk_index, data_json)
                        VALUES (?, ?, ?, ?)
                    ''', (chunk_id, file_id, chunk_index, chunk_json))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error storing data chunks: {e}")
            return False
    
    def retrieve_data_chunks(self, file_id: str, limit: Optional[int] = None) -> Generator[pd.DataFrame, None, None]:
        """
        Retrieve data chunks for a file.
        
        Args:
            file_id: File identifier
            limit: Maximum number of chunks to retrieve
            
        Yields:
            pd.DataFrame: Data chunks
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT chunk_index, data_json 
                    FROM data_chunks 
                    WHERE file_id = ? 
                    ORDER BY chunk_index
                '''
                
                if limit:
                    query += f' LIMIT {limit}'
                
                cursor = conn.cursor()
                cursor.execute(query, (file_id,))
                
                for chunk_index, data_json in cursor.fetchall():
                    chunk_df = pd.read_json(data_json, orient='records')
                    yield chunk_df
                    
        except Exception as e:
            print(f"Error retrieving data chunks: {e}")
    
    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get file information from database.
        
        Args:
            file_id: File identifier
            
        Returns:
            Dict containing file information or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM files WHERE id = ?
                ''', (file_id,))
                
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    file_info = dict(zip(columns, row))
                    
                    # Parse profile data
                    if file_info['profile_data']:
                        file_info['profile_data'] = json.loads(file_info['profile_data'])
                    
                    return file_info
                    
        except Exception as e:
            print(f"Error getting file info: {e}")
            
        return None
    
    async def create_processing_task(self, task_id: str, file_id: str, task_type: str) -> bool:
        """
        Create a new processing task.
        
        Args:
            task_id: Unique task identifier
            file_id: Associated file identifier
            task_type: Type of processing task
            
        Returns:
            bool: Success status
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('''
                    INSERT INTO processing_tasks (task_id, file_id, task_type)
                    VALUES (?, ?, ?)
                ''', (task_id, file_id, task_type))
                await conn.commit()
                return True
        except Exception as e:
            print(f"Error creating processing task: {e}")
            return False
    
    async def update_task_progress(self, task_id: str, progress: float, 
                                 status: Optional[str] = None, 
                                 result_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update task progress and status.
        
        Args:
            task_id: Task identifier
            progress: Progress percentage (0-100)
            status: Optional status update
            result_data: Optional result data
            
        Returns:
            bool: Success status
        """
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                update_fields = ['progress = ?', 'updated_at = CURRENT_TIMESTAMP']
                update_values = [progress]
                
                if status:
                    update_fields.append('status = ?')
                    update_values.append(status)
                
                if result_data:
                    update_fields.append('result_data = ?')
                    update_values.append(json.dumps(result_data))
                
                update_values.append(task_id)
                
                query = f'''
                    UPDATE processing_tasks 
                    SET {', '.join(update_fields)}
                    WHERE task_id = ?
                '''
                
                await conn.execute(query, update_values)
                await conn.commit()
                return True
                
        except Exception as e:
            print(f"Error updating task progress: {e}")
            return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status and progress.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dict containing task information or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM processing_tasks WHERE task_id = ?
                ''', (task_id,))
                
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    task_info = dict(zip(columns, row))
                    
                    # Parse result data
                    if task_info['result_data']:
                        task_info['result_data'] = json.loads(task_info['result_data'])
                    
                    return task_info
                    
        except Exception as e:
            print(f"Error getting task status: {e}")
            
        return None
    
    def cleanup_old_data(self, days: int = 7) -> bool:
        """
        Clean up old files and tasks.
        
        Args:
            days: Number of days to keep data
            
        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old files and their associated data
                cursor.execute('''
                    DELETE FROM data_chunks WHERE file_id IN (
                        SELECT id FROM files 
                        WHERE upload_time < datetime('now', '-{} days')
                    )
                '''.format(days))
                
                cursor.execute('''
                    DELETE FROM processing_tasks WHERE file_id IN (
                        SELECT id FROM files 
                        WHERE upload_time < datetime('now', '-{} days')
                    )
                '''.format(days))
                
                cursor.execute('''
                    DELETE FROM files 
                    WHERE upload_time < datetime('now', '-{} days')
                '''.format(days))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error cleaning up old data: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager()