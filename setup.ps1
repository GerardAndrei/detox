#!/usr/bin/env pwsh

Write-Host "🧹 DETOX Setup Script" -ForegroundColor Cyan
Write-Host "Setting up your data cleaning environment..." -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Check if Node.js is installed
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✅ Found Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js 16+ from https://nodejs.org" -ForegroundColor Red
    exit 1
}

Write-Host "`n📦 Installing Python dependencies..." -ForegroundColor Yellow
Set-Location backend
try {
    pip install -r requirements.txt
    Write-Host "✅ Python dependencies installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install Python dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "`n📦 Installing Node.js dependencies..." -ForegroundColor Yellow
Set-Location ../frontend
try {
    npm install
    Write-Host "✅ Node.js dependencies installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install Node.js dependencies" -ForegroundColor Red
    exit 1
}

Set-Location ..

Write-Host "`n🎉 Setup complete!" -ForegroundColor Green
Write-Host "`nTo start the application:" -ForegroundColor Cyan
Write-Host "1. Start the backend server:" -ForegroundColor White
Write-Host "   cd backend && python main.py" -ForegroundColor Gray
Write-Host "`n2. In a new terminal, start the frontend:" -ForegroundColor White
Write-Host "   cd frontend && npm start" -ForegroundColor Gray
Write-Host "`n3. Open http://localhost:3000 in your browser" -ForegroundColor White
Write-Host "`nHappy data cleaning! 🧹✨" -ForegroundColor Green