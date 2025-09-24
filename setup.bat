@echo off
echo 🧹 DETOX Setup Script
echo Setting up your data cleaning environment...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
) else (
    echo ✅ Python found
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found. Please install Node.js 16+ from https://nodejs.org
    pause
    exit /b 1
) else (
    echo ✅ Node.js found
)

echo.
echo 📦 Installing Python dependencies...
cd backend
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install Python dependencies
    pause
    exit /b 1
) else (
    echo ✅ Python dependencies installed
)

echo.
echo 📦 Installing Node.js dependencies...
cd ..\frontend
npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install Node.js dependencies
    pause
    exit /b 1
) else (
    echo ✅ Node.js dependencies installed
)

cd ..

echo.
echo 🎉 Setup complete!
echo.
echo To start the application:
echo 1. Start the backend server:
echo    cd backend ^&^& python main.py
echo.
echo 2. In a new terminal, start the frontend:
echo    cd frontend ^&^& npm start
echo.
echo 3. Open http://localhost:3000 in your browser
echo.
echo Happy data cleaning! 🧹✨
pause