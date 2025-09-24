import React, { useState, useMemo, useCallback, lazy, Suspense } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import { apiService } from './services/api';

// Lazy load components for code splitting
const Header = lazy(() => import('./components/Header'));
const FileUpload = lazy(() => import('./components/FileUpload'));
const DataPreview = lazy(() => import('./components/DataPreview'));
const CleaningOptions = lazy(() => import('./components/CleaningOptions'));
const Results = lazy(() => import('./components/Results'));

// Loading component for lazy-loaded components
const LoadingSpinner = ({ message = "Loading..." }) => (
  <motion.div
    className="flex items-center justify-center min-h-screen bg-black text-white"
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
  >
    <div className="text-center">
      <motion.div
        className="w-16 h-16 border-4 border-white border-t-transparent rounded-full mx-auto mb-4"
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
      />
      <p className="font-bold text-xl">{message}</p>
    </div>
  </motion.div>
);

const App = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [cleaningOptions, setCleaningOptions] = useState({
    handle_missing: true,
    remove_duplicates: true,
    outlier_method: 'iqr',
    outlier_threshold: 1.5
  });
  const [cleaningResults, setCleaningResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState('checking');
  const [asyncTaskId, setAsyncTaskId] = useState(null);
  const [taskProgress, setTaskProgress] = useState(0);

  // Memoized server status checker
  const checkServerStatus = useCallback(async () => {
    try {
      await apiService.healthCheck();
      setServerStatus('connected');
    } catch (error) {
      setServerStatus('disconnected');
    }
  }, []);

  // Memoized file upload handler
  const handleFileUpload = useCallback(async (file) => {
    setIsLoading(true);
    try {
      const response = await apiService.uploadFile(file);
      setUploadedFile(file);
      setFileData(response);
      setCurrentStep(2);
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Enhanced cleaning with async support
  const handleCleanData = useCallback(async () => {
    if (!fileData?.file_id) return;
    
    setIsLoading(true);
    try {
      // Use async cleaning for better UX
      const response = await apiService.cleanDatasetAsync(fileData.file_id, cleaningOptions);
      setAsyncTaskId(response.task_id);
      
      // Poll for task completion
      const pollTask = async () => {
        while (true) {
          const status = await apiService.getTaskStatus(response.task_id);
          setTaskProgress(status.progress);
          
          if (status.status === 'completed') {
            setCleaningResults(status.result);
            setCurrentStep(4);
            setIsLoading(false);
            break;
          } else if (status.status === 'failed') {
            console.error('Cleaning failed:', status.message);
            setIsLoading(false);
            break;
          }
          
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      };
      
      pollTask();
      
    } catch (error) {
      console.error('Cleaning failed:', error);
      setIsLoading(false);
    }
  }, [fileData, cleaningOptions]);

  // Memoized download handler
  const handleDownload = useCallback(async () => {
    if (!fileData?.file_id || !uploadedFile?.name) return;
    
    try {
      const filename = `cleaned_${uploadedFile.name}`;
      await apiService.downloadCleanedFile(fileData.file_id, filename);
    } catch (error) {
      console.error('Download failed:', error);
    }
  }, [fileData, uploadedFile]);

  // Memoized reset function
  const resetApp = useCallback(() => {
    setCurrentStep(1);
    setUploadedFile(null);
    setFileData(null);
    setCleaningResults(null);
    setAsyncTaskId(null);
    setTaskProgress(0);
    setCleaningOptions({
      handle_missing: true,
      remove_duplicates: true,
      outlier_method: 'iqr',
      outlier_threshold: 1.5
    });
  }, []);

  // Memoized step variants
  const stepVariants = useMemo(() => ({
    hidden: { opacity: 0, y: 30, scale: 0.95 },
    visible: { opacity: 1, y: 0, scale: 1 },
    exit: { opacity: 0, y: -30, scale: 0.95 }
  }), []);

  // Memoized current step renderer
  const renderCurrentStep = useMemo(() => {
    const stepComponents = {
      1: (
        <FileUpload
          onFileUpload={handleFileUpload}
          isLoading={isLoading}
          serverStatus={serverStatus}
          onRetryConnection={checkServerStatus}
        />
      ),
      2: (
        <DataPreview
          fileData={fileData}
          onNext={() => setCurrentStep(3)}
          onBack={() => setCurrentStep(1)}
        />
      ),
      3: (
        <CleaningOptions
          options={cleaningOptions}
          onOptionsChange={setCleaningOptions}
          onClean={handleCleanData}
          onBack={() => setCurrentStep(2)}
          isLoading={isLoading}
          taskProgress={taskProgress}
        />
      ),
      4: (
        <Results
          results={cleaningResults}
          onDownload={handleDownload}
          onStartOver={resetApp}
          onBack={() => setCurrentStep(3)}
        />
      )
    };

    return stepComponents[currentStep] || null;
  }, [
    currentStep,
    handleFileUpload,
    isLoading,
    serverStatus,
    checkServerStatus,
    fileData,
    cleaningOptions,
    handleCleanData,
    taskProgress,
    cleaningResults,
    handleDownload,
    resetApp
  ]);

  return (
    <div className="min-h-screen bg-black overflow-hidden">
      {/* Optimized Background Pattern */}
      <div className="fixed inset-0 pointer-events-none opacity-5">
        <div className="bg-stripes absolute inset-0 animate-pulse-scale"></div>
        <div className="bg-dots absolute inset-0 transform rotate-45"></div>
      </div>

      {/* Optimized Floating Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none opacity-20">
        <motion.div 
          className="absolute top-10 right-10 w-32 h-32 border-4 border-white"
          animate={{ rotate: 360 }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
        />
        <motion.div 
          className="absolute bottom-20 left-20 w-24 h-24 bg-white"
          animate={{ y: [-10, 10, -10] }}
          transition={{ duration: 4, repeat: Infinity }}
        />
      </div>

      <div className="relative z-10">
        {/* Enhanced Hero Header */}
        <div className="text-center py-16 border-b-4 border-white">
          <motion.h1 
            className="font-display text-mega text-white mb-4 transform-skew"
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.8, type: "spring" }}
          >
            DETOX
          </motion.h1>
          <motion.p 
            className="font-quirky text-big text-white transform-skew-reverse"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.6 }}
          >
            BRUTAL DATA CLEANING 2.0
          </motion.p>
          
          {/* Performance indicator */}
          <motion.div 
            className="flex justify-center items-center space-x-4 mt-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
          >
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-white text-sm">ENHANCED PERFORMANCE</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
              <span className="text-white text-sm">ASYNC PROCESSING</span>
            </div>
          </motion.div>
          
          <motion.div 
            className="w-32 h-2 bg-white mx-auto mt-8"
            initial={{ width: 0 }}
            animate={{ width: 128 }}
            transition={{ delay: 0.8, duration: 0.8 }}
          />
        </div>

        <Suspense fallback={<LoadingSpinner message="Loading interface..." />}>
          <Header currentStep={currentStep} />
        </Suspense>
        
        <main className="container mx-auto px-4 py-8 min-h-screen">
          {/* Progress indicator for async tasks */}
          {isLoading && asyncTaskId && (
            <motion.div 
              className="fixed top-4 right-4 bg-black border-2 border-white p-4 z-50"
              initial={{ x: 100, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
            >
              <div className="text-white font-bold mb-2">PROCESSING...</div>
              <div className="w-48 h-2 bg-white/20 overflow-hidden">
                <motion.div
                  className="h-full bg-white"
                  style={{ width: `${taskProgress}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
              <div className="text-white text-sm mt-1">{taskProgress.toFixed(0)}% Complete</div>
            </motion.div>
          )}
          
          {/* Main Content - Optimized with Suspense */}
          <AnimatePresence mode="wait">
            <motion.div
              key={currentStep}
              variants={stepVariants}
              initial="hidden"
              animate="visible"
              exit="exit"
              transition={{ duration: 0.5, type: "spring", damping: 25 }}
              className="max-w-full mx-auto h-full"
            >
              <Suspense fallback={<LoadingSpinner message="Loading component..." />}>
                {renderCurrentStep}
              </Suspense>
            </motion.div>
          </AnimatePresence>
        </main>

        {/* Enhanced Footer */}
        <footer className="border-t-4 border-white mt-16 py-8">
          <div className="container mx-auto px-4">
            <div className="text-center">
              <motion.p 
                className="font-mono text-white font-bold text-lg mb-2"
                animate={{ opacity: [1, 0.5, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                ⚡ POWERED BY ENHANCED BLACK & WHITE ENERGY ⚡
              </motion.p>
              <div className="text-white/60 text-sm">
                v2.0 | Async Processing | Smart Caching | Optimized Performance
              </div>
            </div>
          </div>
        </footer>
      </div>

      {/* Enhanced Toast notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#000000',
            color: '#ffffff',
            border: '4px solid #ffffff',
            borderRadius: '0px',
            fontSize: '16px',
            fontWeight: 'bold',
            fontFamily: 'var(--font-quirky)',
          },
          success: {
            iconTheme: {
              primary: '#ffffff',
              secondary: '#000000',
            },
          },
          error: {
            iconTheme: {
              primary: '#ffffff',
              secondary: '#000000',
            },
          },
        }}
      />
    </div>
  );
};

export default App;