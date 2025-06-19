@echo off
echo ================================================================
echo FINANCIAL ANALYSIS SYSTEM - START SERVER
echo ================================================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Cannot activate virtual environment
    echo Try running setup_and_run.bat to reset
    pause
    exit /b 1
)

echo OK: Environment ready!
echo.
echo ================================================================
echo SERVER INFORMATION
echo ================================================================
echo Application URL: http://127.0.0.1:8000/
echo Admin URL: http://127.0.0.1:8000/admin/
echo.
echo USAGE INSTRUCTIONS:
echo 1. Open browser and go to: http://127.0.0.1:8000/
echo 2. Choose one of two analysis options:
echo    - Fraud Detection: Upload phgl.xlsx (multi-user data)
echo    - Personal Finance: Upload canhan.xlsx (personal data)
echo 3. Upload the appropriate Excel file
echo 4. View analysis results and insights
echo.
echo Press Ctrl+C to stop server
echo ================================================================
echo.

REM Auto open browser is disabled - user can manually visit the URL
REM timeout /t 3 >nul
REM start http://127.0.0.1:8000/

echo Starting Django server...
python manage.py runserver

echo.
echo Server stopped. Thank you for using!
pause