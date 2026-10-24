@echo off
title RAIM - Remote Assessment Integrity Monitoring
cd /d "%~dp0"

set PY=C:\Users\harsh\raim_venv\Scripts\python.exe

echo ============================================================
echo  RAIM - starting the local server
echo ============================================================
echo.

if not exist "%PY%" (
    echo ERROR: Python venv not found at:
    echo   %PY%
    echo.
    echo The virtual environment may have been moved or deleted.
    echo See README.md for setup instructions.
    echo.
    pause
    exit /b 1
)

echo Using Python: %PY%
echo Project dir : %cd%
echo.

echo [1/3] Applying database migrations...
"%PY%" manage.py migrate --no-input
if errorlevel 1 (
    echo.
    echo ERROR: migrate failed - see the output above.
    echo.
    pause
    exit /b 1
)

echo.
echo [2/3] Opening your browser in 3 seconds...
start "" /min cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8000/"

echo [3/3] Starting the server at http://127.0.0.1:8000/
echo.
echo   Keep THIS window open while you use RAIM.
echo   Press CTRL+C here to stop the server.
echo.
"%PY%" manage.py runserver 127.0.0.1:8000

echo.
echo Server stopped.
pause
