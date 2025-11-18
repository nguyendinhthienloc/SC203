@echo off
REM One-button launcher for IRAL Pipeline
REM Double-click this file to run the analysis

echo ================================================================================
echo IRAL Text Analysis Pipeline - One-Button Launcher
echo ================================================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo Error: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: venv\Scripts\pip install -e .[dev]
    pause
    exit /b 1
)

REM Run the analysis
venv\Scripts\python.exe run.py

echo.
echo ================================================================================
echo Press any key to exit...
pause >nul
