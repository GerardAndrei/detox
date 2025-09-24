import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Download, 
  RotateCcw, 
  ArrowLeft,
  FileText,
  AlertTriangle,
  Eye,
  Activity,
  Skull,
  Target,
  Zap,
  Trophy
} from 'lucide-react';
import { formatNumber, formatPercentage } from '../services/api';

const Results = ({ results, onDownload, onStartOver, onBack }) => {
  const [activeTab, setActiveTab] = useState('summary');

  if (!results) {
    return (
      <div className="bg-black text-white p-8 border-4 border-white text-center min-h-[400px] flex items-center justify-center transform-skew animate-pulse-scale">
        <div className="font-quirky text-xl">PROCESSING DESTRUCTION RESULTS...</div>
      </div>
    );
  }

  const {
    original_shape = [],
    cleaned_shape = [],
    cleaning_steps = [],
    outliers_detected = 0,
    report = {},
    preview = []
  } = results;

  const tabs = [
    { id: 'summary', name: 'DAMAGE REPORT', icon: Skull },
    { id: 'report', name: 'FULL AUTOPSY', icon: FileText },
    { id: 'preview', name: 'SURVIVORS', icon: Eye }
  ];

  const GrittyMetricCard = ({ icon: Icon, title, value, change, isVictory = false, chaos = false }) => (
    <motion.div
      whileHover={{ 
        scale: 1.1, 
        rotate: chaos ? [-2, 2, -1, 1, 0] : [0, 1, -1, 0],
        transition: { duration: 0.3 }
      }}
      className={`p-4 md:p-6 border-4 transition-all duration-300 transform ${
        chaos ? 'transform-skew-reverse' : 'transform-skew'
      } ${
        isVictory 
          ? 'bg-white text-black border-black shadow-brutal-invert animate-bounce-subtle' 
          : 'bg-black text-white border-white shadow-brutal'
      }`}
      style={{
        transform: `rotate(${chaos ? Math.random() * 4 - 2 : Math.random() * 2 - 1}deg)`
      }}
    >
      <div className={`w-12 h-12 border-4 flex items-center justify-center mb-4 ${
        isVictory ? 'bg-black text-white border-black' : 'bg-white text-black border-white'
      } transform rotate-45`}>
        <Icon className="w-6 h-6 transform -rotate-45" />
      </div>
      <div className="font-quirky text-lg md:text-xl font-bold mb-2 uppercase tracking-wide">{title}</div>
      <div className="font-mono text-2xl md:text-3xl font-black mb-2">{value}</div>
      {change && (
        <div className={`font-mono text-sm font-bold ${
          isVictory ? 'text-gray-700' : 'text-gray-300'
        }`}>
          {change}
        </div>
      )}
    </motion.div>
  );

  const renderSummary = () => {
    const dataRetentionRate = ((cleaned_shape[0] || 0) / (original_shape[0] || 1)) * 100;
    const rowsReduced = (original_shape[0] || 0) - (cleaned_shape[0] || 0);

    return (
      <motion.div
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="space-y-6"
      >
        {/* DESTRUCTION SUMMARY */}
        <div className="bg-black border-4 border-white p-6 transform-skew shadow-brutal">
          <h3 className="font-quirky text-2xl md:text-3xl font-black mb-6 uppercase tracking-wider text-center">
            DESTRUCTION SUMMARY
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex justify-between items-center py-3 border-b-2 border-white border-dotted">
                <span className="font-mono font-bold uppercase">ORIGINAL DATASET:</span>
                <span className="font-mono text-xl font-black">{formatNumber(original_shape[0] || 0)} × {original_shape[1] || 0}</span>
              </div>
              <div className="flex justify-between items-center py-3 border-b-2 border-white border-dotted">
                <span className="font-mono font-bold uppercase">SURVIVORS:</span>
                <span className="font-mono text-xl font-black text-green-400">{formatNumber(cleaned_shape[0] || 0)} × {cleaned_shape[1] || 0}</span>
              </div>
              <div className="flex justify-between items-center py-3 border-b-2 border-white border-dotted">
                <span className="font-mono font-bold uppercase">ELIMINATION COUNT:</span>
                <span className="font-mono text-xl font-black text-red-400">{formatNumber(rowsReduced)}</span>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="flex justify-between items-center py-3 border-b-2 border-white border-dotted">
                <span className="font-mono font-bold uppercase">RETENTION RATE:</span>
                <span className="font-mono text-xl font-black">{formatPercentage(dataRetentionRate / 100)}</span>
              </div>
              <div className="flex justify-between items-center py-3 border-b-2 border-white border-dotted">
                <span className="font-mono font-bold uppercase">OUTLIERS DESTROYED:</span>
                <span className="font-mono text-xl font-black text-orange-400">{formatNumber(outliers_detected)}</span>
              </div>
              <div className="flex justify-between items-center py-3 border-b-2 border-white border-dotted">
                <span className="font-mono font-bold uppercase">CLEANING OPERATIONS:</span>
                <span className="font-mono text-xl font-black">{cleaning_steps.length}</span>
              </div>
            </div>
          </div>
        </div>

        {/* CLEANING OPERATIONS */}
        {cleaning_steps.length > 0 && (
          <div className="bg-black border-4 border-white p-6 transform-skew-reverse shadow-brutal">
            <h3 className="font-quirky text-2xl font-black mb-4 uppercase tracking-wider">
              DESTRUCTION METHODS
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {cleaning_steps.map((step, index) => (
                <motion.div
                  key={index}
                  whileHover={{ scale: 1.05, rotate: [0, 1, -1, 0] }}
                  className="bg-white text-black p-4 border-2 border-black transform-skew font-mono font-bold uppercase tracking-wide text-center"
                  style={{ transform: `rotate(${Math.random() * 4 - 2}deg)` }}
                >
                  {step}
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </motion.div>
    );
  };

  const renderReport = () => {
    if (!report || Object.keys(report).length === 0) {
      return (
        <div className="bg-black border-4 border-white p-8 text-center transform-skew">
          <div className="font-quirky text-xl uppercase tracking-wider">
            NO DETAILED REPORT AVAILABLE
          </div>
        </div>
      );
    }

    return (
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        <div className="bg-black border-4 border-white p-6 transform-skew-reverse shadow-brutal">
          <h3 className="font-quirky text-2xl font-black mb-6 uppercase tracking-wider text-center">
            AUTOPSY REPORT
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(report).map(([key, value], index) => (
              <motion.div
                key={key}
                whileHover={{ scale: 1.02, rotate: [0, 0.5, -0.5, 0] }}
                className="p-4 bg-white text-black border-2 border-black transform-skew"
                style={{ transform: `rotate(${Math.random() * 2 - 1}deg)` }}
              >
                <div className="font-mono font-bold uppercase tracking-wide mb-2">
                  {key.replace(/_/g, ' ')}
                </div>
                <div className="font-mono text-xl font-black">
                  {typeof value === 'number' ? formatNumber(value) : String(value)}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>
    );
  };

  const renderPreview = () => {
    if (!preview || preview.length === 0) {
      return (
        <div className="bg-black border-4 border-white p-8 text-center transform-skew">
          <div className="font-quirky text-xl uppercase tracking-wider">
            NO SURVIVORS TO PREVIEW
          </div>
        </div>
      );
    }

    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-black border-4 border-white p-6 transform-skew shadow-brutal overflow-hidden"
      >
        <h3 className="font-quirky text-2xl font-black mb-4 uppercase tracking-wider text-center">
          SURVIVING DATA
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full font-mono text-sm">
            <thead>
              <tr className="border-b-2 border-white">
                {preview[0] && Object.keys(preview[0]).map((key, index) => (
                  <th key={key} className="p-3 text-left font-bold uppercase tracking-wide border-r-2 border-white last:border-r-0">
                    {key}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.slice(0, 10).map((row, index) => (
                <tr key={index} className="border-b border-gray-600 hover:bg-white hover:text-black transition-all duration-200">
                  {Object.values(row).map((value, cellIndex) => (
                    <td key={cellIndex} className="p-3 border-r border-gray-600 last:border-r-0">
                      {value !== null && value !== undefined ? String(value) : 'NULL'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {preview.length > 10 && (
          <div className="mt-4 text-center font-mono text-gray-300">
            SHOWING 10 OF {preview.length} ROWS
          </div>
        )}
      </motion.div>
    );
  };

  return (
    <div className="bg-black text-white min-h-screen relative overflow-hidden max-w-6xl mx-auto">
      {/* VICTORY HEADER - Optimized */}
      <motion.div
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative p-4 md:p-6 lg:p-8 border-b-4 border-white bg-black"
      >
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white via-50% to-transparent opacity-8 animate-pulse-scale"></div>
        <div className="relative z-10 flex flex-col md:flex-row items-center justify-between">
          <div className="flex items-center gap-3 md:gap-4 mb-3 md:mb-0">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-10 h-10 md:w-12 md:h-12 lg:w-16 lg:h-16 bg-white text-black border-2 md:border-3 lg:border-4 border-black flex items-center justify-center transform rotate-45"
            >
              <Trophy className="w-5 h-5 md:w-6 md:h-6 lg:w-8 lg:h-8 transform -rotate-45" />
            </motion.div>
            <div>
              <h1 className="font-quirky text-2xl md:text-3xl lg:text-4xl font-black transform-skew tracking-wider">
                DETOX COMPLETE
              </h1>
              <p className="font-mono text-xs md:text-sm lg:text-base text-gray-300 transform-skew-reverse">
                DATA ANNIHILATION SUCCESSFUL
              </p>
            </div>
          </div>
          
          <motion.button
            whileHover={{ scale: 1.05, rotate: [0, -3, 3, 0] }}
            whileTap={{ scale: 0.95 }}
            onClick={onBack}
            className="px-3 md:px-4 py-2 bg-white text-black border-2 md:border-3 lg:border-4 border-black font-mono font-bold text-sm md:text-base uppercase tracking-wide hover:bg-black hover:text-white transition-all duration-300 transform-skew"
          >
            <ArrowLeft className="w-3 h-3 md:w-4 md:h-4 inline mr-1 md:mr-2" />
            ESCAPE
          </motion.button>
        </div>
      </motion.div>

      {/* DAMAGE METRICS - Optimized */}
      <div className="p-4 md:p-6 lg:p-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6 mb-8"
        >
          <GrittyMetricCard
            icon={Activity}
            title="ORIGINAL ROWS"
            value={formatNumber(original_shape[0] || 0)}
            change="BEFORE DESTRUCTION"
            chaos={true}
          />
          <GrittyMetricCard
            icon={Target}
            title="SURVIVORS"
            value={formatNumber(cleaned_shape[0] || 0)}
            change="AFTER PURGE"
            isVictory={true}
          />
          <GrittyMetricCard
            icon={Zap}
            title="ELIMINATION RATE"
            value={formatPercentage(1 - (cleaned_shape[0] || 0) / (original_shape[0] || 1))}
            change="DESTRUCTION EFFICIENCY"
            chaos={true}
          />
          <GrittyMetricCard
            icon={AlertTriangle}
            title="OUTLIERS KILLED"
            value={formatNumber(outliers_detected)}
            change="ANOMALIES DESTROYED"
            isVictory={true}
          />
        </motion.div>

        {/* TAB NAVIGATION */}
        <div className="flex flex-wrap gap-2 mb-6 border-4 border-white bg-black p-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <motion.button
                key={tab.id}
                whileHover={{ scale: 1.05, rotate: [0, 1, -1, 0] }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 font-mono font-bold uppercase tracking-wide border-2 transition-all duration-300 transform-skew ${
                  activeTab === tab.id
                    ? 'bg-white text-black border-black shadow-brutal-invert'
                    : 'bg-black text-white border-white hover:bg-white hover:text-black'
                }`}
              >
                <Icon className="w-4 h-4 inline mr-2" />
                {tab.name}
              </motion.button>
            );
          })}
        </div>

        {/* TAB CONTENT */}
        <div className="mb-8">
          {activeTab === 'summary' && renderSummary()}
          {activeTab === 'report' && renderReport()}
          {activeTab === 'preview' && renderPreview()}
        </div>

        {/* ACTION BUTTONS */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col md:flex-row gap-4 justify-center items-center"
        >
          <motion.button
            whileHover={{ scale: 1.1, rotate: [0, -2, 2, 0] }}
            whileTap={{ scale: 0.95 }}
            onClick={onDownload}
            className="px-8 py-4 bg-white text-black border-4 border-black font-quirky text-xl font-black uppercase tracking-wider hover:bg-black hover:text-white transition-all duration-300 transform-skew shadow-brutal-invert"
          >
            <Download className="w-6 h-6 inline mr-3" />
            EXTRACT SURVIVORS
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.1, rotate: [0, 2, -2, 0] }}
            whileTap={{ scale: 0.95 }}
            onClick={onStartOver}
            className="px-8 py-4 bg-black text-white border-4 border-white font-mono text-lg font-bold uppercase tracking-wide hover:bg-white hover:text-black transition-all duration-300 transform-skew-reverse shadow-brutal"
          >
            <RotateCcw className="w-5 h-5 inline mr-3" />
            RESTART DESTRUCTION
          </motion.button>
        </motion.div>
      </div>
    </div>
  );
};

export default Results;