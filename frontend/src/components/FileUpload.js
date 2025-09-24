import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { 
  Upload, 
  FileText, 
  AlertTriangle, 
  CheckCircle, 
  RefreshCw,
  Server,
  Skull,
  Target
} from 'lucide-react';
import toast from 'react-hot-toast';
import { validateFileType, validateFileSize, formatFileSize } from '../services/api';

const FileUpload = ({ onFileUpload, isLoading, serverStatus, onRetryConnection }) => {
  const [dragActive, setDragActive] = useState(false);

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    setDragActive(false);
    
    // Handle rejected files
    if (rejectedFiles.length > 0) {
      rejectedFiles.forEach((file) => {
        file.errors.forEach((error) => {
          if (error.code === 'file-too-large') {
            toast.error(`File ${file.file.name} is too large. Maximum size is 50MB.`);
          } else if (error.code === 'file-invalid-type') {
            toast.error(`File ${file.file.name} is not a valid CSV file.`);
          } else {
            toast.error(`Error with file ${file.file.name}: ${error.message}`);
          }
        });
      });
      return;
    }

    // Handle accepted files
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      
      // Additional validation
      if (!validateFileType(file)) {
        toast.error('Please upload a valid CSV file.');
        return;
      }
      
      if (!validateFileSize(file, 50)) {
        toast.error('File size must be less than 50MB.');
        return;
      }
      
      // File is valid, proceed with upload
      toast.success(`File "${file.name}" ready for upload!`);
      onFileUpload(file);
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.csv']
    },
    maxSize: 50 * 1024 * 1024, // 50MB
    multiple: false
  });

  const ServerStatusIndicator = () => {
    const statusConfig = {
      checking: {
        icon: RefreshCw,
        color: 'text-white',
        bgColor: 'bg-white',
        text: 'SCANNING SYSTEMS...',
        spin: true
      },
      connected: {
        icon: Target,
        color: 'text-white',
        bgColor: 'bg-white',
        text: 'SYSTEMS ONLINE',
        spin: false
      },
      disconnected: {
        icon: Skull,
        color: 'text-white',
        bgColor: 'bg-white',
        text: 'SYSTEMS DOWN',
        spin: false
      }
    };

    const config = statusConfig[serverStatus];
    const Icon = config.icon;

    return (
      <div className="flex items-center justify-center space-x-3 md:space-x-4 mb-6 md:mb-8">
        <div className={`p-2 md:p-3 bg-black border-2 md:border-3 lg:border-4 border-white ${config.spin ? 'animate-spin' : 'animate-pulse-scale'}`}>
          <Icon className="w-4 h-4 md:w-5 md:h-5 lg:w-6 lg:h-6 text-white" />
        </div>
        <span className="font-quirky font-bold text-lg md:text-xl text-white">
          {config.text}
        </span>
        {serverStatus === 'disconnected' && (
          <button
            onClick={onRetryConnection}
            className="btn-outline ml-3 md:ml-4 text-sm md:text-base px-4 md:px-6 py-2 md:py-3"
          >
            RETRY
          </button>
        )}
      </div>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 30, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.8, type: "spring" }}
      className="h-full flex flex-col max-w-4xl mx-auto"
    >
      {/* Main Content - Optimized sizing */}
      <div className="flex-1 flex flex-col justify-center">
        <div className="text-center mb-6 md:mb-8">
          <h2 className="font-display text-3xl md:text-4xl lg:text-5xl text-white mb-3 md:mb-4 transform-chaos animate-gritty-hover shadow-brutal font-bold">
            UPLOAD TARGET
          </h2>
          <p className="font-quirky text-lg md:text-xl lg:text-2xl text-white transform-tilt-2 animate-shake">
            FEED THE MACHINE YOUR DIRTY DATA
          </p>
        </div>

        <ServerStatusIndicator />

        {/* Upload Zone - Optimized sizing */}
        <div className="max-w-3xl mx-auto w-full mb-6 md:mb-8 relative">
          {/* Chaotic Background Elements - smaller and subtler */}
          <div className="absolute -top-2 -left-2 w-4 h-4 bg-white transform-brutal animate-rotate-slow opacity-15"></div>
          <div className="absolute -bottom-2 -right-2 w-3 h-3 bg-white transform-chaos animate-bounce opacity-20"></div>
          <div className="absolute top-1/2 -left-3 w-2 h-2 bg-white transform-tilt-3 animate-pulse opacity-18"></div>
          
          <div
            {...getRootProps()}
            className={`upload-zone-bold min-h-[200px] md:min-h-[280px] lg:min-h-[320px] flex items-center justify-center border-brutal shadow-chaotic transform-aggressive animate-chaotic ${isDragActive ? 'drag-active animate-brutal' : ''} ${
              serverStatus !== 'connected' ? 'opacity-30 pointer-events-none' : ''
            }`}
          >
            <input {...getInputProps()} />
            
            <div className="text-center">
              <motion.div
                animate={{
                  scale: isDragActive ? 1.1 : 1,
                  rotate: isDragActive ? [0, 3, -3, 0] : 0,
                }}
                transition={{ duration: 0.3 }}
                className="mb-4 md:mb-6"
              >
                {isLoading ? (
                  <div className="flex items-center justify-center">
                    <div className="spinner-bold mr-3 animate-glitch"></div>
                    <Upload className="w-12 h-12 md:w-16 md:h-16 lg:w-20 lg:h-20 text-current transform-tilt-2" />
                  </div>
                ) : (
                  <div className="transform-tilt-2 animate-gritty-hover shadow-brutal bg-white text-black p-3 md:p-4 inline-block">
                    <Upload className="w-12 h-12 md:w-16 md:h-16 lg:w-20 lg:h-20 text-current mx-auto" />
                  </div>
                )}
              </motion.div>
              
              <div className="space-y-2 md:space-y-4">
                {isLoading ? (
                  <p className="font-quirky text-lg md:text-xl lg:text-2xl font-bold transform-skew-2 animate-shake">
                    PROCESSING YOUR DATA...
                  </p>
                ) : (
                  <>
                    <p className="font-quirky text-lg md:text-xl lg:text-2xl font-bold transform-chaos animate-gritty-hover">
                      {isDragActive
                        ? "RELEASE THE CHAOS!"
                        : "DRAG & DROP YOUR CSV"}
                    </p>
                    <p className="font-mono text-sm md:text-base lg:text-lg transform-tilt-1">
                      OR CLICK TO SELECT FROM DEVICE
                    </p>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Info Sections - Optimized sizing */}
        <div className="max-w-4xl mx-auto w-full grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8">
          {/* File requirements */}
          <div className="bg-black text-white p-4 md:p-6 border-gritty shadow-chaotic transform-tilt-1 animate-brutal">
            <h3 className="font-quirky text-lg md:text-xl lg:text-2xl font-bold mb-4 md:mb-6 text-center text-white transform-skew-1">
              MISSION PARAMETERS
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center space-x-3 animate-gritty-hover">
                <div className="p-2 bg-white text-black flex-shrink-0 transform-tilt-2 shadow-brutal">
                  <FileText className="w-5 h-5" />
                </div>
                <span className="font-mono font-bold text-sm md:text-base text-white">
                  CSV FORMAT ONLY
                </span>
              </div>
              
              <div className="flex items-center space-x-3 animate-gritty-hover">
                <div className="p-2 bg-white text-black flex-shrink-0 transform-tilt-2 shadow-brutal">
                  <Server className="w-5 h-5" />
                </div>
                <span className="font-mono font-bold text-sm md:text-base text-white">
                  MAX 50MB SIZE
                </span>
              </div>
              
              <div className="flex items-center space-x-3 animate-gritty-hover">
                <div className="p-2 bg-white text-black flex-shrink-0 transform-tilt-2 shadow-brutal">
                  <CheckCircle className="w-5 h-5" />
                </div>
                <span className="font-mono font-bold text-sm md:text-base text-white">
                  HEADERS ADVISED
                </span>
              </div>
              
              <div className="flex items-center space-x-3 animate-gritty-hover">
                <div className="p-2 bg-white text-black flex-shrink-0 transform-tilt-2 shadow-brutal">
                  <AlertTriangle className="w-5 h-5" />
                </div>
                <span className="font-mono font-bold text-sm md:text-base text-white">
                  NO SENSITIVE DATA
                </span>
              </div>
            </div>
          </div>

          {/* Sample data note - Enhanced */}
          <div className="bg-white text-black p-6 md:p-8 border-chaotic shadow-gritty transform-tilt-minus-1 animate-chaotic">
            <div className="flex items-start space-x-4">
              <div className="p-2 bg-black text-white flex-shrink-0 transform-chaos shadow-brutal">
                <FileText className="w-6 h-6" />
              </div>
              <div>
                <h4 className="font-quirky font-bold text-lg md:text-xl mb-2 text-black transform-skew-1 animate-gritty-hover">
                  NEED TEST DATA?
                </h4>
                <p className="font-mono text-xs md:text-sm leading-relaxed text-black transform-tilt-1">
                  SAMPLE DATASET WITH INTENTIONAL CHAOS INCLUDED.<br/>
                  LOCATION: data/sample.csv<br/>
                  PURPOSE: TESTING DESTRUCTION CAPABILITIES
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default FileUpload;