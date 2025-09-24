import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Skull, 
  Eye, 
  AlertTriangle, 
  Target,
  Database,
  Hash,
  ArrowLeft,
  ArrowRight,
  Bomb
} from 'lucide-react';
import { formatNumber, formatPercentage } from '../services/api';

const DataPreview = ({ fileData, onNext, onBack }) => {
  const [activeTab, setActiveTab] = useState('overview');

  // Debug logging
  console.log('DataPreview received fileData:', fileData);

  if (!fileData) {
    return (
      <motion.div 
        className="bg-black text-white p-6 md:p-8 lg:p-12 border-4 md:border-6 lg:border-8 border-white text-center min-h-screen flex items-center justify-center relative overflow-hidden max-w-4xl mx-auto"
        initial={{ scale: 0, rotate: 180 }}
        animate={{ scale: 1, rotate: 0 }}
        transition={{ type: "spring", stiffness: 200, damping: 15 }}
      >
        <div className="absolute inset-0 opacity-8">
          <div className="absolute top-1/4 left-1/4 text-4xl md:text-6xl lg:text-7xl transform rotate-45">💀</div>
          <div className="absolute top-1/3 right-1/4 text-4xl md:text-6xl lg:text-7xl transform -rotate-45">⚡</div>
          <div className="absolute bottom-1/4 left-1/3 text-4xl md:text-6xl lg:text-7xl transform rotate-12">🎯</div>
        </div>
        <div className="relative z-10">
          <div className="font-black text-3xl md:text-4xl lg:text-5xl font-space tracking-widest mb-3 md:mb-4 transform -skew-x-12">
            NO DATA
          </div>
          <div className="font-black text-2xl md:text-3xl lg:text-4xl font-space tracking-widest transform skew-x-6">
            UPLOAD FIRST
          </div>
          <motion.div 
            className="mt-6 md:mt-8 text-4xl md:text-5xl lg:text-6xl"
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          >
            🚨
          </motion.div>
        </div>
      </motion.div>
    );
  }

  if (!fileData.profile) {
    return (
      <motion.div 
        className="bg-black text-white p-6 md:p-8 lg:p-12 border-4 md:border-6 lg:border-8 border-white text-center min-h-screen flex items-center justify-center relative overflow-hidden max-w-4xl mx-auto"
        initial={{ scale: 0, rotate: 180 }}
        animate={{ scale: 1, rotate: 0 }}
        transition={{ type: "spring", stiffness: 200, damping: 15 }}
      >
        <div className="absolute inset-0 opacity-8">
          <div className="absolute top-1/4 left-1/4 text-4xl md:text-6xl lg:text-7xl transform rotate-45">💀</div>
          <div className="absolute top-1/3 right-1/4 text-4xl md:text-6xl lg:text-7xl transform -rotate-45">⚡</div>
          <div className="absolute bottom-1/4 left-1/3 text-4xl md:text-6xl lg:text-7xl transform rotate-12">🎯</div>
        </div>
        <div className="relative z-10">
          <div className="font-black text-3xl md:text-4xl lg:text-5xl font-space tracking-widest mb-3 md:mb-4 transform -skew-x-12">
            ANALYZING
          </div>
          <div className="font-black text-2xl md:text-3xl lg:text-4xl font-space tracking-widest transform skew-x-6">
            DATA STRUCTURE...
          </div>
          <motion.div 
            className="mt-8 text-8xl"
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          >
            💣
          </motion.div>
          <div className="mt-4 font-jetbrains text-lg">
            File ID: {fileData.file_id || 'Unknown'}
          </div>
          <div className="font-jetbrains text-lg">
            Filename: {fileData.filename || 'Unknown'}
          </div>
        </div>
      </motion.div>
    );
  }

  const { profile, filename } = fileData;
  const { 
    basic_info = {}, 
    missing_values = {}, 
    data_quality_issues = {}, 
    sample_data = [] 
  } = profile;

  const tabs = [
    { id: 'overview', name: 'INTEL', icon: Eye },
    { id: 'columns', name: 'ARSENAL', icon: Database },
    { id: 'quality', name: 'THREATS', icon: AlertTriangle },
    { id: 'sample', name: 'EVIDENCE', icon: Skull }
  ];

  const ChaosStatCard = ({ icon: Icon, title, value, subtitle, isHighlight = false, destructionLevel = 'NORMAL' }) => {
    const levelConfig = {
      CRITICAL: { 
        bg: 'bg-black', 
        text: 'text-white', 
        border: 'border-white',
        shadow: 'shadow-brutal',
        rotation: 'transform rotate-3',
        warning: '🚨 CRITICAL' 
      },
      WARNING: { 
        bg: 'bg-white', 
        text: 'text-black', 
        border: 'border-black',
        shadow: 'shadow-brutal-invert',
        rotation: 'transform -rotate-2',
        warning: '⚠️ WARNING' 
      },
      NORMAL: { 
        bg: 'bg-black', 
        text: 'text-white', 
        border: 'border-white',
        shadow: 'shadow-brutal',
        rotation: 'transform rotate-1',
        warning: '✓ STABLE' 
      }
    };

    const config = levelConfig[destructionLevel];

    return (
      <motion.div
        whileHover={{ scale: 1.02, rotate: isHighlight ? [0, 1, -1, 0] : 0.5 }}
        className={`${config.bg} ${config.text} p-4 md:p-5 lg:p-6 border-2 md:border-3 lg:border-4 ${config.border} ${config.shadow} 
                   ${config.rotation} transition-all duration-300 cursor-pointer relative overflow-hidden
                   ${isHighlight ? 'animate-pulse-border' : ''}`}
      >
        {/* Background chaos - smaller */}
        <div className="absolute inset-0 opacity-3 pointer-events-none">
          <div className="absolute top-1 right-1 text-sm">▲</div>
          <div className="absolute bottom-1 left-1 text-sm">■</div>
          <div className="absolute top-1 left-1 text-sm">●</div>
        </div>

        <div className="relative z-10">
          <div className="flex items-center justify-between mb-2 md:mb-3">
            <motion.div 
              className={`p-1 md:p-2 border-1 md:border-2 ${config.border} ${config.bg === 'bg-black' ? 'bg-white' : 'bg-black'} 
                         transform rotate-12`}
              whileHover={{ rotate: -12, scale: 1.05 }}
            >
              <Icon className={`h-3 w-3 md:h-4 md:w-4 lg:h-5 lg:w-5 ${config.bg === 'bg-black' ? 'text-black' : 'text-white'}`} />
            </motion.div>
            <div className="text-xs font-black opacity-60 tracking-widest">
              {config.warning}
            </div>
          </div>

          <h3 className="font-black text-base md:text-lg lg:text-xl font-space tracking-wider transform -skew-x-3 mb-1 md:mb-2">
            {title.toUpperCase()}
          </h3>
          
          <div className="font-black text-2xl md:text-3xl font-jetbrains transform skew-x-2 mb-1">
            {value}
          </div>
          
          {subtitle && (
            <div className="text-sm opacity-80 font-jetbrains transform -skew-x-1">
              {subtitle}
            </div>
          )}
        </div>
      </motion.div>
    );
  };

  const BrutalTab = ({ tab, isActive, onClick }) => (
    <motion.button
      onClick={onClick}
      className={`px-6 py-4 border-4 font-black text-lg tracking-widest transition-all duration-300 relative
        ${isActive 
          ? 'bg-white text-black border-black shadow-brutal transform rotate-2 z-10' 
          : 'bg-black text-white border-white shadow-brutal-invert transform -rotate-1 hover:rotate-1'}`}
      whileHover={{ scale: 1.05, y: -2 }}
      whileTap={{ scale: 0.95 }}
    >
      <div className="flex items-center space-x-2">
        <tab.icon className="h-5 w-5" />
        <span className="font-space">{tab.name}</span>
      </div>
    </motion.button>
  );

  const renderOverview = () => (
    <div className="space-y-8">
      {/* File Info */}
      <motion.div 
        className="bg-black text-white p-6 border-4 border-white shadow-brutal transform rotate-1"
        initial={{ x: -100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ delay: 0.1 }}
      >
        <h3 className="font-black text-2xl font-space tracking-wider mb-4 transform -skew-x-6">
          TARGET ACQUIRED
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-jetbrains">
          <div className="transform skew-x-2">
            <span className="opacity-80">FILENAME:</span> {filename}
          </div>
          <div className="transform -skew-x-2">
            <span className="opacity-80">ENCODING:</span> {basic_info.encoding || 'UNKNOWN'}
          </div>
        </div>
      </motion.div>

      {/* Core Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <ChaosStatCard
          icon={Database}
          title="Total Rows"
          value={formatNumber(basic_info.total_rows || 0)}
          subtitle="DATA POINTS"
          destructionLevel="NORMAL"
        />
        <ChaosStatCard
          icon={Hash}
          title="Total Columns"
          value={basic_info.total_columns || 0}
          subtitle="PARAMETERS"
          destructionLevel="NORMAL"
        />
        <ChaosStatCard
          icon={Target}
          title="Missing Values"
          value={formatPercentage(missing_values.total_missing_percentage || 0)}
          subtitle="VOID CONTAMINATION"
          destructionLevel={(missing_values.total_missing_percentage || 0) > 10 ? "WARNING" : "NORMAL"}
          isHighlight={(missing_values.total_missing_percentage || 0) > 10}
        />
        <ChaosStatCard
          icon={AlertTriangle}
          title="Quality Issues"
          value={Object.keys(data_quality_issues || {}).length}
          subtitle="ANOMALIES DETECTED"
          destructionLevel={Object.keys(data_quality_issues || {}).length > 3 ? "CRITICAL" : "NORMAL"}
          isHighlight={Object.keys(data_quality_issues || {}).length > 3}
        />
      </div>

      {/* Data Types */}
      <motion.div 
        className="bg-white text-black p-6 border-4 border-black shadow-brutal-invert transform -rotate-1"
        initial={{ x: 100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        <h3 className="font-black text-2xl font-space tracking-wider mb-6 transform skew-x-6">
          DATA TYPE ARSENAL
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center transform rotate-1">
            <div className="font-black text-3xl font-jetbrains">
              {basic_info.numeric_columns || 0}
            </div>
            <div className="font-space tracking-wider">NUMERIC</div>
          </div>
          <div className="text-center transform -rotate-1">
            <div className="font-black text-3xl font-jetbrains">
              {basic_info.categorical_columns || 0}
            </div>
            <div className="font-space tracking-wider">CATEGORICAL</div>
          </div>
          <div className="text-center transform rotate-2">
            <div className="font-black text-3xl font-jetbrains">
              {basic_info.datetime_columns || 0}
            </div>
            <div className="font-space tracking-wider">DATETIME</div>
          </div>
          <div className="text-center transform -rotate-2">
            <div className="font-black text-3xl font-jetbrains">
              {basic_info.boolean_columns || 0}
            </div>
            <div className="font-space tracking-wider">BOOLEAN</div>
          </div>
        </div>
      </motion.div>
    </div>
  );

  const renderColumns = () => (
    <div className="space-y-6">
      <motion.div 
        className="bg-black text-white p-6 border-4 border-white shadow-brutal transform rotate-1"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 200 }}
      >
        <h3 className="font-black text-2xl font-space tracking-wider mb-6 transform -skew-x-6">
          COLUMN ARSENAL REPORT
        </h3>
        
        <div className="grid gap-4">
          {Object.entries(missing_values.column_missing || {}).map(([column, missing], index) => (
            <motion.div
              key={column}
              className={`p-4 border-2 border-white transform transition-all duration-300 relative
                ${missing > 50 ? 'bg-white text-black' : 'bg-transparent'}`}
              style={{ transform: `rotate(${(index % 3 - 1) * 1}deg)` }}
              whileHover={{ scale: 1.02, rotate: (index % 3 - 1) * 2 }}
            >
              <div className="flex justify-between items-center">
                <div className="font-jetbrains font-bold transform -skew-x-2">
                  {column}
                </div>
                <div className="flex items-center space-x-4">
                  <div className="font-black font-space">
                    {formatPercentage(missing)} VOID
                  </div>
                  {missing > 50 && (
                    <motion.div 
                      className="text-xl"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity }}
                    >
                      💀
                    </motion.div>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  );

  const renderQuality = () => (
    <div className="space-y-6">
      <motion.div 
        className="bg-white text-black p-6 border-4 border-black shadow-brutal-invert transform -rotate-1"
        initial={{ y: 100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
      >
        <h3 className="font-black text-2xl font-space tracking-wider mb-6 transform skew-x-6">
          THREAT ASSESSMENT
        </h3>
        
        {Object.keys(data_quality_issues).length === 0 ? (
          <div className="text-center py-12">
            <div className="text-8xl mb-4">✅</div>
            <div className="font-black text-3xl font-space tracking-widest">
              NO THREATS DETECTED
            </div>
            <div className="font-jetbrains text-xl opacity-80 mt-2">
              Data quality is pristine
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {Object.entries(data_quality_issues).map(([issue, details], index) => (
              <motion.div
                key={issue}
                className="bg-black text-white p-4 border-4 border-white transform rotate-1 relative"
                whileHover={{ rotate: -1, scale: 1.02 }}
                initial={{ x: index % 2 === 0 ? -100 : 100, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="flex items-center space-x-4">
                  <motion.div 
                    className="text-2xl"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    🚨
                  </motion.div>
                  <div className="flex-1">
                    <h4 className="font-black text-xl font-space tracking-wider transform -skew-x-3">
                      {issue.toUpperCase().replace(/_/g, ' ')}
                    </h4>
                    <p className="font-jetbrains opacity-80 transform skew-x-1">
                      {typeof details === 'object' ? JSON.stringify(details) : details}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>
    </div>
  );

  const renderSample = () => (
    <div className="space-y-6">
      <motion.div 
        className="bg-black text-white border-4 border-white shadow-brutal transform rotate-1 overflow-hidden"
        initial={{ rotateY: -90, opacity: 0 }}
        animate={{ rotateY: 0, opacity: 1 }}
        transition={{ type: "spring", stiffness: 200 }}
      >
        <div className="p-6 border-b-4 border-white">
          <h3 className="font-black text-2xl font-space tracking-wider transform -skew-x-6">
            EVIDENCE SAMPLE
          </h3>
          <div className="font-jetbrains opacity-80 mt-2 transform skew-x-2">
            First {sample_data.length} rows of captured data
          </div>
        </div>
        
        <div className="overflow-x-auto">
          {sample_data && sample_data.length > 0 ? (
            <table className="w-full">
              <thead className="bg-white text-black">
                <tr>
                  {sample_data[0] && Object.keys(sample_data[0]).map((column, index) => (
                    <th key={column} 
                        className={`px-4 py-3 font-black font-space tracking-wider text-left border-r-4 border-black
                          transform ${index % 2 === 0 ? 'skew-x-3' : '-skew-x-3'}`}>
                      {column.toUpperCase()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sample_data.map((row, rowIndex) => (
                  <motion.tr 
                    key={rowIndex}
                    className={`border-b-2 border-white ${rowIndex % 2 === 0 ? 'bg-black' : 'bg-gray-900'}`}
                    whileHover={{ backgroundColor: '#333', scale: 1.01 }}
                    initial={{ x: rowIndex % 2 === 0 ? -50 : 50, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ delay: rowIndex * 0.05 }}
                  >
                    {Object.values(row).map((value, colIndex) => (
                      <td key={colIndex} 
                          className={`px-4 py-3 font-jetbrains border-r-2 border-white/20
                            transform ${colIndex % 2 === 0 ? 'skew-x-1' : '-skew-x-1'}`}>
                        {String(value || '').length > 50 ? `${String(value || '').substring(0, 47)}...` : String(value || '')}
                      </td>
                    ))}
                  </motion.tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="text-center py-12">
              <div className="text-8xl mb-4">📊</div>
              <div className="font-black text-3xl font-space tracking-widest">
                NO SAMPLE DATA
              </div>
              <div className="font-jetbrains text-xl opacity-80 mt-2">
                Sample data not available for this file
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );

  return (
    <motion.div
      initial={{ opacity: 0, rotateX: -90 }}
      animate={{ opacity: 1, rotateX: 0 }}
      transition={{ type: "spring", stiffness: 200, damping: 20 }}
      className="min-h-screen bg-white text-black p-4 md:p-6 lg:p-8"
    >
      {/* Optimized Header */}
      <motion.div 
        className="text-center mb-8 md:mb-10 lg:mb-12 relative max-w-4xl mx-auto"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 15, delay: 0.2 }}
      >
        <div className="absolute inset-0 bg-black transform rotate-2 -z-10"></div>
        <div className="bg-white border-4 md:border-6 lg:border-8 border-black p-4 md:p-6 lg:p-8 shadow-brutal transform -rotate-1">
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-black font-playfair transform -skew-x-12 tracking-wider mb-2 md:mb-4">
            DATA
          </h1>
          <h2 className="text-xl md:text-2xl lg:text-3xl font-space tracking-widest text-black/80 transform skew-x-6">
            RECONNAISSANCE REPORT
          </h2>
          <div className="text-sm md:text-lg lg:text-xl font-jetbrains mt-2 md:mt-4 transform rotate-1 opacity-60">
            Intelligence gathered from hostile data territory
          </div>
        </div>
      </motion.div>

      {/* Optimized Tab Navigation */}
      <div className="flex flex-wrap justify-center gap-3 md:gap-4 mb-8 md:mb-10 lg:mb-12">
        {tabs.map((tab) => (
          <BrutalTab
            key={tab.id}
            tab={tab}
            isActive={activeTab === tab.id}
            onClick={() => setActiveTab(tab.id)}
          />
        ))}
      </div>

      {/* Content - Optimized max width */}
      <div className="max-w-5xl lg:max-w-6xl mx-auto">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'columns' && renderColumns()}
        {activeTab === 'quality' && renderQuality()}
        {activeTab === 'sample' && renderSample()}
      </div>

      {/* Navigation - Optimized spacing */}
      <motion.div 
        className="flex flex-col md:flex-row gap-4 md:gap-6 justify-center items-center mt-8 md:mt-12 lg:mt-16"
        initial={{ y: 100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <motion.button
          onClick={onBack}
          className="bg-black text-white px-8 py-4 border-4 border-white font-black text-xl 
                    tracking-wider transform -rotate-2 shadow-brutal hover:rotate-2 
                    transition-all duration-300"
          whileHover={{ scale: 1.1, rotate: 2 }}
          whileTap={{ scale: 0.9 }}
        >
          <div className="flex items-center space-x-3">
            <ArrowLeft className="h-6 w-6" />
            <span className="font-space">RETREAT</span>
          </div>
        </motion.button>

        <motion.button
          onClick={onNext}
          className="bg-white text-black px-12 py-6 border-4 border-black font-black text-2xl 
                    tracking-widest transform rotate-2 shadow-brutal-invert hover:-rotate-2 
                    hover:scale-110 transition-all duration-300 relative overflow-hidden"
          whileHover={{ scale: 1.1, rotate: -2 }}
          whileTap={{ scale: 0.95 }}
        >
          <div className="flex items-center space-x-4">
            <Bomb className="h-8 w-8" />
            <span className="font-space">PROCEED TO DESTRUCTION</span>
            <ArrowRight className="h-8 w-8" />
          </div>
        </motion.button>
      </motion.div>
    </motion.div>
  );
};

export default DataPreview;