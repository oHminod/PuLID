@echo off
setlocal EnableExtensions

set "PROJECT_DIR=%~dp0"
set "PULID_MODELS_ROOT=%PROJECT_DIR%PuLID_models"
set "HF_HOME=%PULID_MODELS_ROOT%\huggingface"
set "HUGGINGFACE_HUB_CACHE=%PULID_MODELS_ROOT%\huggingface\hub"
set "TRANSFORMERS_CACHE=%PULID_MODELS_ROOT%\huggingface\transformers"
set "TORCH_HOME=%PULID_MODELS_ROOT%\torch"
set "XDG_CACHE_HOME=%PULID_MODELS_ROOT%\other"
set "MPLCONFIGDIR=%PULID_MODELS_ROOT%\other\matplotlib"
set "NO_ALBUMENTATIONS_UPDATE=1"

cd /d "%PROJECT_DIR%"

if not exist "%PULID_MODELS_ROOT%\" (
    echo [ERREUR] Dossier de modeles introuvable :
    echo   %PULID_MODELS_ROOT%
    exit /b 1
)

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

"%SERVER_EXE%" --host 0.0.0.0 --port 12693 --device cuda --dtype float16 --offload none --cors-origin "*"
exit /b %ERRORLEVEL%
