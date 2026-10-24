@echo off
title RAIM - train models
rem RAIM - train CNN, extract fusion features, train SVM, evaluate
cd /d "%~dp0"
set PY=C:\Users\harsh\raim_venv\Scripts\python.exe

if not exist "%PY%" (
    echo ERROR: Python venv not found at:
    echo   %PY%
    echo.
    pause
    exit /b 1
)

echo === 1/4 Training behavior CNN ===
"%PY%" ml_training\train_cnn.py || goto :err
echo === 2/4 Extracting fusion features ===
"%PY%" ml_training\extract_features.py || goto :err
echo === 3/4 Training fusion SVM ===
"%PY%" ml_training\train_svm.py || goto :err
echo === 4/4 Evaluating models ===
"%PY%" ml_training\evaluate_models.py
echo Done. Artifacts in ml_models\
pause
exit /b 0
:err
echo Training step failed - see output above.
pause
exit /b 1
