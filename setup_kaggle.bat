@echo off
echo ============================================
echo Kaggle API Setup for AIMO3
echo ============================================

:: 1. Set environment variable for current session
set KAGGLE_API_TOKEN=KGAT_b33acef3351632e2bd1dce705002c5a0

:: 2. Set it permanently (user level)
setx KAGGLE_API_TOKEN "KGAT_b33acef3351632e2bd1dce705002c5a0"

echo.
echo [1/3] Environment variable set!

:: 3. Check kaggle installation
pip show kaggle >nul 2>&1
if %errorlevel% neq 0 (
    echo [2/3] Installing kaggle...
    pip install kaggle
) else (
    echo [2/3] Kaggle already installed
)

echo.
echo [3/3] Testing connection...
kaggle competitions list -s aimo

echo.
echo ============================================
echo Setup complete! Now run:
echo   cd C:\Users\jrjin\Desktop\MathLLM\kaggle-test
echo   kaggle kernels push -p .
echo ============================================
pause
