"""
Caching Module for DETOX API

Provides intelligent caching for file profiles and processing results.
"""

import hashlib
import json
import os
import pickle
import time
from typing import Any, Dict, Optional
from pathlib import Path
import redis
from functools import wraps

class CacheManager:
    """
    Manages caching for file profiles and processing results.
    """
    
    def __init__(self, cache_type: str = 'file', redis_url: Optional[str] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_type (str): Type of cache ('file', 'redis', or 'memory')
            redis_url (str, optional): Redis URL for Redis caching
        """
        self.cache_type = cache_type
        self.cache_dir = Path('.cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        
        # Redis setup (optional)
        self.redis_client = None
        if cache_type == 'redis' and redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()  # Test connection
            except Exception as e:
                print(f"Redis connection failed, falling back to file cache: {e}")
                self.cache_type = 'file'
    
    def generate_cache_key(self, data: Any) -> str:
        """
        Generate a cache key from data.
        
        Args:
            data: Data to generate key from
            
        Returns:
            str: Cache key
        """
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            if self.cache_type == 'memory':
                item = self.memory_cache.get(key)
                if item and item['expires'] > time.time():
                    return item['data']
                elif item:
                    del self.memory_cache[key]  # Expired
                    
            elif self.cache_type == 'redis' and self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    return pickle.loads(data)
                    
            elif self.cache_type == 'file':
                cache_file = self.cache_dir / f"{key}.cache"
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        item = pickle.load(f)
                        if item['expires'] > time.time():
                            return item['data']
                        else:
                            cache_file.unlink()  # Remove expired file
                            
        except Exception as e:
            print(f"Cache get error: {e}")
            
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set value in cache.
        
        Args:
            key (str): Cache key
            value: Value to cache
            ttl (int): Time to live in seconds (default: 1 hour)
            
        Returns:
            bool: Success status
        """
        try:
            expires_at = time.time() + ttl
            
            if self.cache_type == 'memory':
                self.memory_cache[key] = {
                    'data': value,
                    'expires': expires_at
                }
                
            elif self.cache_type == 'redis' and self.redis_client:
                self.redis_client.setex(key, ttl, pickle.dumps(value))
                
            elif self.cache_type == 'file':
                cache_file = self.cache_dir / f"{key}.cache"
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'data': value,
                        'expires': expires_at
                    }, f)
                    
            return True
            
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: Success status
        """
        try:
            if self.cache_type == 'memory':
                self.memory_cache.pop(key, None)
                
            elif self.cache_type == 'redis' and self.redis_client:
                self.redis_client.delete(key)
                
            elif self.cache_type == 'file':
                cache_file = self.cache_dir / f"{key}.cache"
                if cache_file.exists():
                    cache_file.unlink()
                    
            return True
            
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    def clear_expired(self):
        """Clear expired cache entries."""
        try:
            if self.cache_type == 'memory':
                current_time = time.time()
                expired_keys = [
                    key for key, item in self.memory_cache.items()
                    if item['expires'] <= current_time
                ]
                for key in expired_keys:
                    del self.memory_cache[key]
                    
            elif self.cache_type == 'file':
                current_time = time.time()
                for cache_file in self.cache_dir.glob('*.cache'):
                    try:
                        with open(cache_file, 'rb') as f:
                            item = pickle.load(f)
                            if item['expires'] <= current_time:
                                cache_file.unlink()
                    except Exception:
                        # Remove corrupted cache files
                        cache_file.unlink()
                        
        except Exception as e:
            print(f"Cache cleanup error: {e}")

# Global cache manager instance
cache_manager = CacheManager()

def cached(ttl: int = 3600, key_func: Optional[callable] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl (int): Time to live in seconds
        key_func (callable, optional): Function to generate cache key
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager.generate_cache_key(
                    f"{func.__name__}:{args}:{kwargs}"
                )
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

@cached(ttl=1800)  # Cache for 30 minutes
def cache_file_profile(file_path: str, file_size: int) -> Dict[str, Any]:
    """
    Cache file profile based on path and size.
    
    Args:
        file_path (str): Path to file
        file_size (int): File size in bytes
        
    Returns:
        Dict: File profile (this is just a placeholder function)
    """
    # This would be implemented in the main profiling code
    return {"cached": True, "file_path": file_path, "size": file_size}