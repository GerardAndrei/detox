#!/bin/bash

echo "🧹 DETOX Setup Script"
echo "Setting up your data cleaning environment..."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+ from https://python.org"
    exit 1
else
    echo "✅ Found Python: $(python --version)"
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 16+ from https://nodejs.org"
    exit 1
else
    echo "✅ Found Node.js: $(node --version)"
fi

echo ""
echo "📦 Installing Python dependencies..."
cd backend
if pip install -r requirements.txt; then
    echo "✅ Python dependencies installed"
else
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

echo ""
echo "📦 Installing Node.js dependencies..."
cd ../frontend
if npm install; then
    echo "✅ Node.js dependencies installed"
else
    echo "❌ Failed to install Node.js dependencies"
    exit 1
fi

cd ..

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "1. Start the backend server:"
echo "   cd backend && python main.py"
echo ""
echo "2. In a new terminal, start the frontend:"
echo "   cd frontend && npm start"
echo ""
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo "Happy data cleaning! 🧹✨"