@echo off
echo 🚀 Starting DETOX Backend Server...
echo.
echo The server will be available at: http://localhost:8000
echo API documentation will be available at: http://localhost:8000/docs
echo.
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload