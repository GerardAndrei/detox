import axios from 'axios';
import toast from 'react-hot-toast';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Enhanced axios instance with optimizations
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // Increased to 60 seconds for large file processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request queue for managing concurrent requests
const requestQueue = new Map();
const maxConcurrentRequests = 3;
let activeRequests = 0;

// Simple in-memory cache for GET requests
const cache = new Map();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

// Retry configuration
const retryConfig = {
  maxRetries: 3,
  retryDelay: 1000,
  retryCondition: (error) => {
    return error.code === 'ECONNABORTED' || 
           error.response?.status >= 500 ||
           error.code === 'NETWORK_ERROR';
  }
};

// Request interceptor with caching and queuing
api.interceptors.request.use(
  async (config) => {
    // Add cache key for GET requests
    if (config.method === 'get') {
      const cacheKey = `${config.url}${JSON.stringify(config.params || {})}`;
      const cachedResponse = cache.get(cacheKey);
      
      if (cachedResponse && Date.now() - cachedResponse.timestamp < CACHE_TTL) {
        console.log(`Cache hit for ${config.url}`);
        return Promise.reject({ 
          cached: true, 
          data: cachedResponse.data,
          status: 200 
        });
      }
      
      config.cacheKey = cacheKey;
    }

    // Queue management for concurrent requests
    if (activeRequests >= maxConcurrentRequests) {
      await new Promise(resolve => {
        const checkQueue = () => {
          if (activeRequests < maxConcurrentRequests) {
            activeRequests++;
            resolve();
          } else {
            setTimeout(checkQueue, 100);
          }
        };
        checkQueue();
      });
    } else {
      activeRequests++;
    }

    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor with caching and retry logic
api.interceptors.response.use(
  (response) => {
    activeRequests--;
    
    // Cache GET responses
    if (response.config.method === 'get' && response.config.cacheKey) {
      cache.set(response.config.cacheKey, {
        data: response.data,
        timestamp: Date.now()
      });
    }
    
    return response;
  },
  async (error) => {
    activeRequests--;
    
    // Handle cached responses
    if (error.cached) {
      return Promise.resolve({ 
        data: error.data, 
        status: error.status,
        cached: true 
      });
    }
    
    console.error('Response error:', error);
    
    // Retry logic
    const config = error.config;
    if (config && retryConfig.retryCondition(error)) {
      config.retryCount = config.retryCount || 0;
      
      if (config.retryCount < retryConfig.maxRetries) {
        config.retryCount++;
        console.log(`Retrying request ${config.retryCount}/${retryConfig.maxRetries}`);
        
        await new Promise(resolve => 
          setTimeout(resolve, retryConfig.retryDelay * config.retryCount)
        );
        
        return api(config);
      }
    }
    
    // Error handling with user-friendly messages
    if (error.code === 'ECONNABORTED') {
      toast.error('Request timeout - please try again');
    } else if (error.response?.status === 500) {
      toast.error('Server error - please try again later');
    } else if (error.response?.status === 404) {
      toast.error('Resource not found');
    } else if (error.response?.data?.detail) {
      toast.error(error.response.data.detail);
    } else if (error.message && !error.cached) {
      toast.error(error.message);
    } else if (!error.cached) {
      toast.error('An unexpected error occurred');
    }
    
    return Promise.reject(error);
  }
);

export const apiService = {
  // Health check
  async healthCheck() {
    try {
      const response = await api.get('/');
      return response.data;
    } catch (error) {
      throw new Error('Unable to connect to the server');
    }
  },

  // Upload file
  async uploadFile(file) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await api.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          console.log(`Upload progress: ${percentCompleted}%`);
        },
      });
      
      return response.data;
    } catch (error) {
      console.error('Upload error:', error);
      throw error;
    }
  },

  // Clean dataset with async option
  async cleanDatasetAsync(fileId, options) {
    try {
      const response = await api.post(`/clean/${fileId}/async`, options);
      return response.data;
    } catch (error) {
      console.error('Async cleaning error:', error);
      throw error;
    }
  },

  // Get task status
  async getTaskStatus(taskId) {
    try {
      const response = await api.get(`/task/${taskId}/status`);
      return response.data;
    } catch (error) {
      console.error('Task status error:', error);
      throw error;
    }
  },

  // Clean dataset (synchronous, original)
  async cleanDataset(fileId, options) {
    try {
      const response = await api.post(`/clean/${fileId}`, options);
      return response.data;
    } catch (error) {
      console.error('Cleaning error:', error);
      throw error;
    }
  },

  // Download cleaned file
  async downloadCleanedFile(fileId, filename = 'cleaned_data.csv') {
    try {
      const response = await api.get(`/download/${fileId}`, {
        responseType: 'blob',
      });
      
      // Create blob link to download
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      return true;
    } catch (error) {
      console.error('Download error:', error);
      throw error;
    }
  },

  // Get detailed profile
  async getDetailedProfile(fileId) {
    try {
      const response = await api.get(`/profile/${fileId}`);
      return response.data;
    } catch (error) {
      console.error('Profile error:', error);
      throw error;
    }
  },

  // Delete file
  async deleteFile(fileId) {
    try {
      const response = await api.delete(`/files/${fileId}`);
      return response.data;
    } catch (error) {
      console.error('Delete error:', error);
      throw error;
    }
  },
};

// Helper functions for data formatting
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const formatNumber = (num) => {
  if (num === null || num === undefined || isNaN(num)) return '0';
  const numValue = Number(num);
  if (numValue >= 1000000) {
    return (numValue / 1000000).toFixed(1) + 'M';
  }
  if (numValue >= 1000) {
    return (numValue / 1000).toFixed(1) + 'K';
  }
  return numValue.toString();
};

export const formatPercentage = (num, decimals = 1) => {
  if (num === null || num === undefined || isNaN(num)) return '0.0%';
  return Number(num).toFixed(decimals) + '%';
};

export const validateFileType = (file) => {
  const allowedTypes = ['text/csv', 'application/vnd.ms-excel'];
  const allowedExtensions = ['.csv'];
  
  const fileExtension = file.name.toLowerCase().substr(file.name.lastIndexOf('.'));
  
  return allowedTypes.includes(file.type) || allowedExtensions.includes(fileExtension);
};

export const validateFileSize = (file, maxSizeMB = 50) => {
  const maxSizeBytes = maxSizeMB * 1024 * 1024;
  return file.size <= maxSizeBytes;
};

export default apiService;