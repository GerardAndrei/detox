import { useCallback } from 'react';

/**
 * Custom hook for memoized callbacks to prevent unnecessary re-renders
 */
export const useOptimizedCallback = (callback, dependencies) => {
  return useCallback(callback, dependencies);
};

/**
 * Custom hook for lazy loading components
 */
export const useLazyComponent = (importFunc) => {
  return useCallback(importFunc, []);
};

export { useOptimizedCallback, useLazyComponent };