@echo off
echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║          PicturaAI — Neural Style Transfer        ║
echo  ║              Starting Server...                  ║
echo  ╚══════════════════════════════════════════════════╝
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Change to backend directory
cd backend

REM Start the FastAPI server
echo  ► Server starting at: http://localhost:8000
echo  ► Open your browser to: http://localhost:8000/app
echo  ► API docs at:         http://localhost:8000/docs
echo.
echo  Press Ctrl+C to stop the server.
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
