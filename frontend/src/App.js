import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import DataPreview from './components/DataPreview';
import CleaningOptions from './components/CleaningOptions';
import Results from './components/Results';
import { apiService } from './services/api';

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

  // Check server status on component mount
  useEffect(() => {
    checkServerStatus();
  }, []);

  const checkServerStatus = async () => {
    try {
      await apiService.healthCheck();
      setServerStatus('connected');
    } catch (error) {
      setServerStatus('disconnected');
    }
  };

  const handleFileUpload = async (file) => {
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
  };

  const handleCleanData = async () => {
    if (!fileData?.file_id) return;
    
    setIsLoading(true);
    try {
      const response = await apiService.cleanDataset(fileData.file_id, cleaningOptions);
      setCleaningResults(response);
      setCurrentStep(4);
    } catch (error) {
      console.error('Cleaning failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!fileData?.file_id || !uploadedFile?.name) return;
    
    try {
      const filename = `cleaned_${uploadedFile.name}`;
      await apiService.downloadCleanedFile(fileData.file_id, filename);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const resetApp = () => {
    setCurrentStep(1);
    setUploadedFile(null);
    setFileData(null);
    setCleaningResults(null);
    setCleaningOptions({
      handle_missing: true,
      remove_duplicates: true,
      outlier_method: 'iqr',
      outlier_threshold: 1.5
    });
  };

  const stepVariants = {
    hidden: { opacity: 0, y: 30, scale: 0.95 },
    visible: { opacity: 1, y: 0, scale: 1 },
    exit: { opacity: 0, y: -30, scale: 0.95 }
  };

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <FileUpload
            onFileUpload={handleFileUpload}
            isLoading={isLoading}
            serverStatus={serverStatus}
            onRetryConnection={checkServerStatus}
          />
        );
      case 2:
        return (
          <DataPreview
            fileData={fileData}
            onNext={() => setCurrentStep(3)}
            onBack={() => setCurrentStep(1)}
          />
        );
      case 3:
        return (
          <CleaningOptions
            options={cleaningOptions}
            onOptionsChange={setCleaningOptions}
            onClean={handleCleanData}
            onBack={() => setCurrentStep(2)}
            isLoading={isLoading}
          />
        );
      case 4:
        return (
          <Results
            results={cleaningResults}
            onDownload={handleDownload}
            onStartOver={resetApp}
            onBack={() => setCurrentStep(3)}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-black overflow-hidden">
      {/* Aggressive Background Pattern */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="bg-stripes opacity-5 absolute inset-0 animate-pulse-scale"></div>
        <div className="bg-dots opacity-10 absolute inset-0 transform rotate-45"></div>
      </div>

      {/* Floating Elements - Optimized and subtle */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-16 right-8 w-16 md:w-24 h-16 md:h-24 border-2 border-white transform-rotate-1 animate-rotate-slow opacity-10"></div>
        <div className="absolute bottom-16 left-8 w-12 md:w-16 h-12 md:h-16 bg-white transform-skew animate-bounce-subtle opacity-15"></div>
        <div className="absolute top-1/2 left-4 w-8 h-24 md:h-32 bg-white transform-skew-reverse opacity-8"></div>
        <div className="absolute bottom-8 right-1/4 w-24 md:w-32 h-4 md:h-6 bg-white transform-rotate-minus-1 opacity-12"></div>
      </div>

      <div className="relative z-10">
        {/* Hero Header - Optimized sizing */}
        <div className="text-center py-8 md:py-12 border-b-4 border-white">
          <motion.h1 
            className="font-display text-6xl md:text-7xl lg:text-8xl text-white mb-3 transform-skew font-black"
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.8, type: "spring" }}
          >
            DETOX
          </motion.h1>
          <motion.p 
            className="font-quirky text-xl md:text-2xl lg:text-3xl text-white transform-skew-reverse font-bold"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.6 }}
          >
            BRUTAL DATA CLEANING
          </motion.p>
          <motion.div 
            className="w-24 md:w-32 h-1 md:h-2 bg-white mx-auto mt-4 md:mt-6"
            initial={{ width: 0 }}
            animate={{ width: '6rem' }}
            transition={{ delay: 0.6, duration: 0.8 }}
          ></motion.div>
        </div>

        <Header currentStep={currentStep} />
        
        <main className="container mx-auto px-4 md:px-6 lg:px-8 py-6 md:py-8">
          {/* Main Content - Full Screen Optimized */}
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
              {renderCurrentStep()}
            </motion.div>
          </AnimatePresence>
        </main>

        {/* Footer - Optimized sizing */}
        <footer className="border-t-4 border-white mt-8 md:mt-12 py-6">
          <div className="container mx-auto px-4">
            <div className="text-center">
              <motion.p 
                className="font-mono text-white font-bold text-sm md:text-base lg:text-lg"
                animate={{ opacity: [1, 0.5, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                ⚡ POWERED BY BLACK & WHITE ENERGY ⚡
              </motion.p>
            </div>
          </div>
        </footer>
      </div>

      {/* Toast notifications with brutal styling */}
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