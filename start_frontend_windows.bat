@echo off
setlocal EnableExtensions

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

if exist "%PROJECT_DIR%.venv\Scripts\python.exe" (
    "%PROJECT_DIR%.venv\Scripts\python.exe" "%PROJECT_DIR%frontend\server.py" --host 127.0.0.1 --port 8888 %*
    exit /b %ERRORLEVEL%
)

where py.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py.exe -3 "%PROJECT_DIR%frontend\server.py" --host 127.0.0.1 --port 8888 %*
    exit /b %ERRORLEVEL%
)

echo [ERREUR] Python 3 introuvable. Executez d'abord : install_windows.bat
exit /b 1
