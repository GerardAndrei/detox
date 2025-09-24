#!/bin/bash

# Vercel build script for Detox app
echo "Building Detox frontend..."

# Navigate to frontend directory and build
cd frontend
echo "Installing frontend dependencies..."
npm ci --only=production
echo "Building React application..."
npm run build

echo "Build completed successfully!"