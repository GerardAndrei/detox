import React from 'react';
import { motion } from 'framer-motion';
import { 
  Skull, 
  Zap, 
  Bomb, 
  Target, 
  ArrowLeft, 
  Play,
  CheckCircle
} from 'lucide-react';

const CleaningOptions = ({ options, onOptionsChange, onClean, onBack, isLoading }) => {
  const handleOptionChange = (key, value) => {
    onOptionsChange({
      ...options,
      [key]: value
    });
  };

  const DestructionCard = ({ 
    id, 
    title, 
    description, 
    icon: Icon, 
    enabled, 
    onToggle, 
    children,
    destructionLevel = 'MODERATE'
  }) => {
    const levelConfig = {
      MILD: { 
        bg: 'bg-white', 
        text: 'text-black', 
        border: 'border-black',
        shadow: 'shadow-brutal-invert',
        warning: 'SURGICAL PRECISION' 
      },
      MODERATE: { 
        bg: 'bg-black', 
        text: 'text-white', 
        border: 'border-white',
        shadow: 'shadow-brutal',
        warning: 'CONTROLLED CHAOS' 
      },
      EXTREME: { 
        bg: 'bg-white', 
        text: 'text-black', 
        border: 'border-black',
        shadow: 'shadow-brutal-invert transform-rotate-1',
        warning: 'TOTAL ANNIHILATION' 
      }
    };

    const config = levelConfig[destructionLevel];

    return (
      <motion.div
        layout
        whileHover={{ 
          scale: 1.01, 
          rotate: enabled ? [0, 0.5, -0.5, 0] : 0,
          y: -3 
        }}
        className={`${config.bg} ${config.text} p-4 md:p-5 lg:p-6 border-3 md:border-4 ${config.border} ${config.shadow} 
                   transform transition-all duration-300 cursor-pointer overflow-hidden relative
                   ${enabled ? 'animate-pulse-border' : ''}`}
        onClick={() => onToggle(!enabled)}
      >
        {/* Chaos Background Pattern */}
        <div className="absolute inset-0 opacity-5 pointer-events-none">
          {[...Array(3)].map((_, i) => (
            <div key={i} className={`absolute ${config.text} transform rotate-45 -translate-x-1/2 -translate-y-1/2`}
                 style={{ top: `${20 + i * 30}%`, left: `${10 + i * 25}%` }}>
              ■ ▲ ●
            </div>
          ))}
        </div>

        {/* Header */}
        <div className="relative z-10 flex items-center justify-between">
          <div className="flex items-center space-x-3 md:space-x-4">
            <motion.div 
              className={`p-2 md:p-3 border-2 ${config.border} ${config.bg === 'bg-black' ? 'bg-white' : 'bg-black'} 
                         transform rotate-12 shadow-lg`}
              whileHover={{ rotate: -12, scale: 1.05 }}
            >
              <Icon className={`h-4 w-4 md:h-5 md:w-5 lg:h-6 lg:w-6 ${config.bg === 'bg-black' ? 'text-black' : 'text-white'}`} />
            </motion.div>
            <div>
              <h3 className="font-black text-lg md:text-xl lg:text-2xl font-space tracking-wider transform -skew-x-6">
                {title.toUpperCase()}
              </h3>
              <p className="text-sm md:text-base lg:text-lg font-jetbrains transform skew-x-3 opacity-80">
                {description}
              </p>
              <div className="text-xs font-black opacity-60 tracking-widest mt-1">
                {config.warning}
              </div>
            </div>
          </div>
          
          <motion.div 
            className="flex items-center space-x-3 md:space-x-4"
            whileHover={{ x: 3 }}
          >
            <motion.button
              className={`w-8 h-8 md:w-10 md:h-10 lg:w-12 lg:h-12 border-2 md:border-3 lg:border-4 ${config.border} ${enabled ? config.bg : 'bg-transparent'} 
                         transform rotate-45 shadow-brutal transition-all duration-300`}
              whileHover={{ rotate: 90, scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={(e) => {
                e.stopPropagation();
                onToggle(!enabled);
              }}
            >
              {enabled && (
                <CheckCircle className={`h-4 w-4 md:h-5 md:w-5 lg:h-6 lg:w-6 ${config.text} transform -rotate-45 mx-auto`} />
              )}
            </motion.button>
          </motion.div>
        </div>

        {/* Expanded Content */}
        {enabled && children && (
          <motion.div
            initial={{ opacity: 0, height: 0, rotateX: -90 }}
            animate={{ opacity: 1, height: 'auto', rotateX: 0 }}
            exit={{ opacity: 0, height: 0, rotateX: -90 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            className="mt-6 pt-6 border-t-4 border-dashed border-current relative"
          >
            <div className="transform skew-x-1">
              {children}
            </div>
          </motion.div>
        )}
      </motion.div>
    );
  };

  const ChaosDetails = ({ title, children }) => (
    <motion.div 
      className="bg-black text-white p-3 md:p-4 border-2 md:border-3 lg:border-4 border-white shadow-brutal transform rotate-1"
      whileHover={{ rotate: -0.5, scale: 1.01 }}
    >
      <h4 className="font-black text-lg md:text-xl font-space tracking-wider mb-3 md:mb-4 transform -skew-x-3">
        {title.toUpperCase()}
      </h4>
      <div className="font-jetbrains text-xs md:text-sm space-y-1 md:space-y-2 transform skew-x-1">
        {children}
      </div>
    </motion.div>
  );

  const estimatedTime = () => {
    let time = 1; // Base time
    if (options.handle_missing) time += 1;
    if (options.remove_duplicates) time += 1;
    if (options.outlier_method) time += 2;
    return `~${time} SECONDS OF CHAOS`;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 50, rotateX: -90 }}
      animate={{ opacity: 1, y: 0, rotateX: 0 }}
      transition={{ type: "spring", stiffness: 200, damping: 20 }}
      className="min-h-screen bg-white text-black p-4 md:p-6 lg:p-8"
    >
      {/* Optimized Header */}
      <motion.div 
        className="text-center mb-8 md:mb-10 relative"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 15, delay: 0.2 }}
      >
        <div className="absolute inset-0 bg-black transform rotate-2 -z-10"></div>
        <div className="bg-white border-4 md:border-6 lg:border-8 border-black p-4 md:p-6 lg:p-8 shadow-brutal transform -rotate-1">
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-black font-playfair transform -skew-x-12 tracking-wider mb-2 md:mb-4">
            CONFIGURE
          </h1>
          <h2 className="text-xl md:text-2xl lg:text-3xl font-space tracking-widest text-black/80 transform skew-x-6">
            DESTRUCTION PARAMETERS
          </h2>
          <div className="text-sm md:text-lg font-jetbrains mt-2 md:mt-4 transform rotate-1 opacity-60">
            Choose your weapons of mass data liberation
          </div>
        </div>
      </motion.div>

      <div className="max-w-4xl lg:max-w-5xl mx-auto space-y-6 md:space-y-8">
        {/* Missing Values - MILD */}
        <DestructionCard
          id="missing"
          title="Missing Value Obliterator"
          description="Surgical removal of data voids and null territories"
          icon={Target}
          enabled={options.handle_missing}
          onToggle={(value) => handleOptionChange('handle_missing', value)}
          destructionLevel="MILD"
        >
          <ChaosDetails title="OPERATION PROTOCOL">
            <div className="transform -skew-x-2">
              • COLUMNS WITH &gt;50% VOID: TERMINATED
            </div>
            <div className="transform skew-x-2">
              • NUMERIC GAPS: FILLED WITH MEDIAN PRECISION
            </div>
            <div className="transform -skew-x-1">
              • TEXT HOLES: PATCHED WITH MODE DOMINANCE
            </div>
            <div className="transform skew-x-3">
              • INTEGRITY PRESERVATION: MAXIMUM
            </div>
          </ChaosDetails>
        </DestructionCard>

        {/* Remove Duplicates - MODERATE */}
        <DestructionCard
          id="duplicates"
          title="Duplicate Terminator"
          description="Clone elimination protocol for data purification"
          icon={Zap}
          enabled={options.remove_duplicates}
          onToggle={(value) => handleOptionChange('remove_duplicates', value)}
          destructionLevel="MODERATE"
        >
          <ChaosDetails title="EXTERMINATION SEQUENCE">
            <div className="transform rotate-1">
              • IDENTICAL ROW DETECTION: ENGAGED
            </div>
            <div className="transform -rotate-1">
              • FIRST OCCURRENCE: PRESERVED
            </div>
            <div className="transform rotate-2">
              • REDUNDANCY ELIMINATION: BRUTAL
            </div>
            <div className="transform -rotate-2">
              • SIZE REDUCTION: SIGNIFICANT
            </div>
          </ChaosDetails>
        </DestructionCard>

        {/* Outliers - EXTREME */}
        <DestructionCard
          id="outliers"
          title="Outlier Annihilator"
          description="Statistical anomaly destruction with extreme prejudice"
          icon={Bomb}
          enabled={!!options.outlier_method}
          onToggle={(value) => handleOptionChange('outlier_method', value ? 'iqr' : null)}
          destructionLevel="EXTREME"
        >
          {options.outlier_method && (
            <div className="space-y-6">
              <ChaosDetails title="ANOMALY DETECTION METHOD">
                <motion.div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {['iqr', 'zscore'].map((method) => (
                    <motion.button
                      key={method}
                      className={`p-4 border-4 font-black tracking-wider transform transition-all duration-300
                        ${options.outlier_method === method 
                          ? 'bg-black text-white border-white shadow-brutal rotate-2' 
                          : 'bg-white text-black border-black shadow-brutal-invert -rotate-1 hover:rotate-1'}`}
                      whileHover={{ scale: 1.05, rotate: method === options.outlier_method ? -2 : 2 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleOptionChange('outlier_method', method);
                      }}
                    >
                      {method === 'iqr' ? 'IQR PROTOCOL' : 'Z-SCORE MASSACRE'}
                    </motion.button>
                  ))}
                </motion.div>
              </ChaosDetails>
              
              <ChaosDetails title="DESTRUCTION PARAMETERS">
                <div className="transform skew-x-2">
                  • STATISTICAL THRESHOLD: MERCILESS
                </div>
                <div className="transform -skew-x-2">
                  • OUTLIER IDENTIFICATION: PRECISE
                </div>
                <div className="transform skew-x-1">
                  • DATA POINT ELIMINATION: SWIFT
                </div>
                <div className="transform -skew-x-3">
                  • DISTRIBUTION NORMALIZATION: BRUTAL
                </div>
              </ChaosDetails>
            </div>
          )}
        </DestructionCard>

        {/* Execution Panel - Optimized */}
        <motion.div 
          className="bg-black text-white p-4 md:p-6 lg:p-8 border-4 md:border-6 lg:border-8 border-white shadow-brutal transform rotate-1 relative overflow-hidden"
          whileHover={{ rotate: -0.5, scale: 1.01 }}
        >
          {/* Background chaos - reduced size */}
          <div className="absolute inset-0 opacity-8 pointer-events-none">
            <div className="absolute top-2 left-2 text-3xl md:text-4xl lg:text-5xl transform rotate-45">💀</div>
            <div className="absolute top-2 right-2 text-3xl md:text-4xl lg:text-5xl transform -rotate-45">⚡</div>
            <div className="absolute bottom-2 left-2 text-3xl md:text-4xl lg:text-5xl transform rotate-12">💣</div>
            <div className="absolute bottom-2 right-2 text-3xl md:text-4xl lg:text-5xl transform -rotate-12">🎯</div>
          </div>

          <div className="relative z-10 text-center">
            <h3 className="text-2xl md:text-3xl lg:text-4xl font-black font-space tracking-widest mb-3 md:mb-4 transform -skew-x-6">
              INITIATE DESTRUCTION SEQUENCE
            </h3>
            
            <div className="text-base md:text-lg lg:text-xl font-jetbrains mb-6 md:mb-8 transform skew-x-3">
              ESTIMATED CHAOS DURATION: {estimatedTime()}
            </div>

            <div className="flex flex-col md:flex-row gap-4 md:gap-6 justify-center items-center">
              <motion.button
                onClick={onBack}
                className="bg-white text-black px-6 md:px-8 py-3 md:py-4 border-3 md:border-4 border-black font-black text-lg md:text-xl 
                          tracking-wider transform -rotate-2 shadow-brutal-invert hover:rotate-2 
                          transition-all duration-300"
                whileHover={{ scale: 1.05, rotate: 2 }}
                whileTap={{ scale: 0.95 }}
              >
                <div className="flex items-center space-x-2 md:space-x-3">
                  <ArrowLeft className="h-4 w-4 md:h-5 md:w-5 lg:h-6 lg:w-6" />
                  <span className="font-space">RETREAT</span>
                </div>
              </motion.button>

              <motion.button
                onClick={onClean}
                disabled={isLoading}
                className={`px-8 md:px-10 lg:px-12 py-4 md:py-5 lg:py-6 border-3 md:border-4 border-white font-black text-lg md:text-xl lg:text-2xl tracking-widest 
                          transform rotate-2 shadow-brutal transition-all duration-300 relative overflow-hidden
                          ${isLoading 
                            ? 'bg-gray-800 text-gray-400 animate-pulse' 
                            : 'bg-white text-black hover:-rotate-2 hover:scale-105'}`}
                whileHover={!isLoading ? { scale: 1.05, rotate: -2 } : {}}
                whileTap={!isLoading ? { scale: 0.95 } : {}}
              >
                <div className="flex items-center space-x-2 md:space-x-3 lg:space-x-4">
                  <Play className="h-5 w-5 md:h-6 md:w-6 lg:h-7 lg:w-7" />
                  <span className="font-space">
                    {isLoading ? 'DESTROYING...' : 'EXECUTE PROTOCOL'}
                  </span>
                  <Skull className="h-5 w-5 md:h-6 md:w-6 lg:h-7 lg:w-7" />
                </div>
                
                {isLoading && (
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30"
                    animate={{ x: [-100, 300] }}
                    transition={{ repeat: Infinity, duration: 1.5 }}
                  />
                )}
              </motion.button>
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
};

export default CleaningOptions;