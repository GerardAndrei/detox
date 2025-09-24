import React from 'react';
import { motion } from 'framer-motion';
import { Skull, Zap, Target, Trophy } from 'lucide-react';

const Header = ({ currentStep }) => {
  const steps = [
    { id: 1, name: 'UPLOAD', icon: Target },
    { id: 2, name: 'INSPECT', icon: Skull },
    { id: 3, name: 'DESTROY', icon: Zap },
    { id: 4, name: 'VICTORY', icon: Trophy },
  ];

  return (
    <header className="relative z-20 py-4 md:py-6 border-b-4 border-white">
      <div className="container mx-auto px-4">
        {/* Progress steps - Optimized sizing */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="flex justify-center"
        >
          <div className="bg-black border-2 md:border-3 lg:border-4 border-white px-4 md:px-6 lg:px-8 py-3 md:py-4 shadow-brutal">
            <div className="flex items-center justify-center space-x-4 md:space-x-6 lg:space-x-8">
              {steps.map((step, index) => {
                const Icon = step.icon;
                const isActive = step.id <= currentStep;
                const isCurrent = step.id === currentStep;
                
                return (
                  <div key={step.id} className="flex items-center space-x-2 md:space-x-3 lg:space-x-4">
                    <motion.div
                      className={`flex items-center space-x-1 md:space-x-2 lg:space-x-3 transition-all duration-300`}
                      animate={{
                        scale: isCurrent ? 1.05 : 1,
                        rotate: isCurrent ? [0, 0.5, -0.5, 0] : 0,
                      }}
                      transition={{
                        rotate: { duration: 2, repeat: Infinity }
                      }}
                    >
                      <div
                        className={`p-1 md:p-2 lg:p-3 border-2 md:border-3 lg:border-4 transition-all duration-300 transform ${
                          isCurrent
                            ? 'bg-white text-black border-white animate-pulse-scale'
                            : isActive
                            ? 'bg-black text-white border-white'
                            : 'bg-transparent text-gray-400 border-gray-400'
                        }`}
                      >
                        <Icon className="w-3 h-3 md:w-4 md:h-4 lg:w-5 lg:h-5" />
                      </div>
                      <span className={`font-quirky font-bold text-xs md:text-sm lg:text-base ${
                        isCurrent
                          ? 'text-white'
                          : isActive
                          ? 'text-white'
                          : 'text-gray-400'
                      }`}>
                        {step.name}
                      </span>
                    </motion.div>
                    
                    {index < steps.length - 1 && (
                      <div
                        className={`w-4 md:w-6 lg:w-8 h-0.5 md:h-0.5 lg:h-1 transition-all duration-300 ${
                          step.id < currentStep
                            ? 'bg-white'
                            : 'bg-gray-500'
                        }`}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </motion.div>
      </div>
    </header>
  );
};

export default Header;