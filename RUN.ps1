# One-button launcher for IRAL Pipeline (PowerShell version)
# Right-click -> Run with PowerShell

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "IRAL Text Analysis Pipeline - One-Button Launcher" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "Error: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv venv"
    Write-Host "Then: venv\Scripts\pip install -e .[dev]"
    Read-Host "Press Enter to exit"
    exit 1
}

# Run the analysis
& "venv\Scripts\python.exe" run.py

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "Press any key to exit..." -ForegroundColor Green
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
