@echo off
setlocal EnableExtensions

set "PROJECT_DIR=%~dp0"
set "TORCH_DLL_DIR=%PROJECT_DIR%.venv\Lib\site-packages\torch\lib"
set "PATH=%TORCH_DLL_DIR%;%PATH%"

cd /d "%PROJECT_DIR%"

set "SERVER_EXE=%PROJECT_DIR%.venv\Scripts\pulid-server.exe"
if not exist "%SERVER_EXE%" (
    echo [ERREUR] Serveur PuLID non installe.
    echo Executez d'abord : install_windows.bat
    exit /b 1
)

echo Serveur PuLID accessible sur le reseau local :
powershell.exe -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.AddressState -eq 'Preferred' } | ForEach-Object { '  http://' + $_.IPAddress + ':12693' }"
echo.
echo Arret du serveur : Ctrl+C
echo.

"%SERVER_EXE%" --host 0.0.0.0 --port 12693 --device cuda --dtype float16 --offload none --cors-origin "*" %*
exit /b %ERRORLEVEL%
