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
set "UV_CACHE_DIR=%PULID_MODELS_ROOT%\other\uv-windows"
set "UV_PYTHON_INSTALL_DIR=%PULID_MODELS_ROOT%\other\uv-python-windows"
set "NO_ALBUMENTATIONS_UPDATE=1"
set "LLAMA_CPP_VERSION=0.3.35"
set "LLAMA_CPP_CUDA_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu130"
set "LLAMA_CPP_CPU_WHEEL=https://github.com/abetlen/llama-cpp-python/releases/download/v%LLAMA_CPP_VERSION%/llama_cpp_python-%LLAMA_CPP_VERSION%-py3-none-win_amd64.whl"
set "LLAMA_CPP_CUDA_WHEEL=https://github.com/abetlen/llama-cpp-python/releases/download/v%LLAMA_CPP_VERSION%-cu130/llama_cpp_python-%LLAMA_CPP_VERSION%-py3-none-win_amd64.whl"
set "LLAMA_CPP_PORTABLE_CPU_DLL_SHA256=cd91f4ed375998da4da57fedaab1b0638fba8b2af88e74a2632bc046e7fa4850"

cd /d "%PROJECT_DIR%"

if not exist "%PULID_MODELS_ROOT%\" (
    echo [ERREUR] Dossier de modeles introuvable :
    echo   %PULID_MODELS_ROOT%
    echo Copiez le dossier PuLID_models a la racine du projet, puis relancez ce script.
    goto :error_exit
)

echo Nettoyage des metadonnees macOS incompatibles avec Windows...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$files = @(Get-ChildItem -LiteralPath $env:PULID_MODELS_ROOT -Recurse -Force -File -Filter '._*' -ErrorAction SilentlyContinue); if ($files.Count -gt 0) { Write-Host ('Suppression de ' + $files.Count + ' fichier(s) AppleDouble.'); $files | Remove-Item -Force -ErrorAction Stop }"
if errorlevel 1 (
    echo [ERREUR] Impossible de supprimer les fichiers AppleDouble sous :
    echo   %PULID_MODELS_ROOT%
    goto :error_exit
)

if not exist "%PULID_MODELS_ROOT%\checkpoints\realvisxlV50_v50LightningBakedvae.safetensors" (
    echo [ERREUR] Checkpoint SDXL par defaut introuvable :
    echo   %PULID_MODELS_ROOT%\checkpoints\realvisxlV50_v50LightningBakedvae.safetensors
    echo Verifiez que le dossier PuLID_models a ete copie integralement.
    goto :error_exit
)

if not exist "%PULID_MODELS_ROOT%\text_embedding\bge-m3-Q8_0.gguf" (
    echo [ERREUR] Modele d'embedding GGUF introuvable :
    echo   %PULID_MODELS_ROOT%\text_embedding\bge-m3-Q8_0.gguf
    echo Creez le dossier text_embedding sous PuLID_models et placez-y le GGUF.
    goto :error_exit
)

set "UV_EXE="
for /f "delims=" %%I in ('where uv.exe 2^>nul') do if not defined UV_EXE set "UV_EXE=%%I"

if not defined UV_EXE (
    set "UV_INSTALL_DIR=%LOCALAPPDATA%\PuLID\uv\bin"
    echo Installation de uv...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo [ERREUR] Impossible d'installer uv.
        echo Installez-le manuellement puis relancez ce script :
        echo   winget install --id=astral-sh.uv -e
        goto :error_exit
    )
    set "UV_EXE=%LOCALAPPDATA%\PuLID\uv\bin\uv.exe"
)

if not exist "%UV_EXE%" (
    echo [ERREUR] Executable uv introuvable : %UV_EXE%
    goto :error_exit
)

set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo Creation de l'environnement Python 3.11...
    "%UV_EXE%" venv --python 3.11 "%PROJECT_DIR%.venv"
    if errorlevel 1 goto :venv_error
)

set "TORCH_DLL_DIR=%PROJECT_DIR%.venv\Lib\site-packages\torch\lib"
set "LLAMA_CPP_LIB_DIR=%PROJECT_DIR%.venv\Lib\site-packages\llama_cpp\lib"
set "LLAMA_CPP_PORTABLE_DIR=%PROJECT_DIR%.venv\pulid-runtime\llama-cpp-%LLAMA_CPP_VERSION%"
set "LLAMA_CPP_PORTABLE_CPU_DLL=%LLAMA_CPP_PORTABLE_DIR%\ggml-cpu.dll"
set "PATH=%TORCH_DLL_DIR%;%PATH%"

echo Installation de PyTorch 2.13 avec CUDA 13.0...
"%UV_EXE%" pip install --python "%VENV_PYTHON%" "torch==2.13.0" "torchvision==0.28.0" --index-url "https://download.pytorch.org/whl/cu130"
if errorlevel 1 goto :dependency_error

echo Preparation du backend CPU portable de llama-cpp-python %LLAMA_CPP_VERSION%...
"%UV_EXE%" pip install --python "%VENV_PYTHON%" --reinstall-package llama-cpp-python "%LLAMA_CPP_CPU_WHEEL%"
if errorlevel 1 goto :dependency_error

if not exist "%LLAMA_CPP_PORTABLE_DIR%\" mkdir "%LLAMA_CPP_PORTABLE_DIR%"
if errorlevel 1 goto :llama_portable_error

if not exist "%LLAMA_CPP_LIB_DIR%\ggml-cpu.dll" goto :llama_portable_error
copy /Y "%LLAMA_CPP_LIB_DIR%\ggml-cpu.dll" "%LLAMA_CPP_PORTABLE_CPU_DLL%" >nul
if errorlevel 1 goto :llama_portable_error

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $env:LLAMA_CPP_PORTABLE_CPU_DLL).Hash.ToLowerInvariant(); if ($actual -ne $env:LLAMA_CPP_PORTABLE_CPU_DLL_SHA256) { Write-Error ('Empreinte ggml-cpu.dll inattendue : ' + $actual); exit 1 }"
if errorlevel 1 goto :llama_portable_error

echo Installation du runtime GGUF CUDA 13.0 pour les embeddings...
"%UV_EXE%" pip install --python "%VENV_PYTHON%" --reinstall-package llama-cpp-python "%LLAMA_CPP_CUDA_WHEEL%"
if errorlevel 1 goto :dependency_error

if not exist "%LLAMA_CPP_LIB_DIR%\ggml-cuda.dll" goto :llama_cuda_error
copy /Y "%LLAMA_CPP_PORTABLE_CPU_DLL%" "%LLAMA_CPP_LIB_DIR%\ggml-cpu.dll" >nul
if errorlevel 1 goto :llama_portable_error

echo Installation de PuLID et du serveur HTTP...
"%UV_EXE%" pip install --python "%VENV_PYTHON%" --extra-index-url "%LLAMA_CPP_CUDA_INDEX%" --only-binary llama-cpp-python -e ".[inference,pulid,server,embeddings,dev]"
if errorlevel 1 goto :dependency_error

echo Verification de CUDA...
"%VENV_PYTHON%" -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponible : mettez a jour le pilote NVIDIA'; print('CUDA OK :', torch.cuda.get_device_name(0), '- PyTorch', torch.__version__)"
if errorlevel 1 goto :cuda_error

echo Verification du runtime GGUF CUDA...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $env:LLAMA_CPP_LIB_DIR 'ggml-cpu.dll')).Hash.ToLowerInvariant(); if ($actual -ne $env:LLAMA_CPP_PORTABLE_CPU_DLL_SHA256) { Write-Error ('Le backend CPU portable a ete remplace : ' + $actual); exit 1 }"
if errorlevel 1 goto :llama_portable_error

"%VENV_PYTHON%" -c "from pulid_app.models.text_embedding import _cuda_dll_search_path; dll_context = _cuda_dll_search_path('cuda'); dll_context.__enter__(); import llama_cpp; info = llama_cpp.llama_print_system_info().decode(); assert 'CUDA' in info, 'Backend CUDA absent de llama-cpp-python'; assert 'AVX512 = 1' not in info, 'Backend CPU AVX-512 incompatible encore installe'; print(info); print('llama-cpp-python CUDA OK :', llama_cpp.__version__); dll_context.__exit__(None, None, None)"
if errorlevel 1 goto :llama_cuda_error

echo Verification du chargement et du calcul BGE-M3 sur CUDA...
"%VENV_PYTHON%" "%PROJECT_DIR%scripts\verify_text_embedding.py" --device cuda
if errorlevel 1 goto :llama_context_error

echo Verification de l'installation et des modeles...
"%PROJECT_DIR%.venv\Scripts\pulid-gen.exe" doctor
if errorlevel 1 goto :validation_error

"%VENV_PYTHON%" "%PROJECT_DIR%scripts\inspect_models.py" --show-cache-env --fail-on-internal-cache
if errorlevel 1 goto :validation_error

netsh advfirewall firewall show rule name="PuLID_API_12693" >nul 2>&1
if errorlevel 1 (
    echo Autorisation du port TCP 12693 sur les reseaux prives...
    echo Windows va demander une confirmation administrateur.
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$process = Start-Process -FilePath netsh.exe -ArgumentList @('advfirewall','firewall','add','rule','name=PuLID_API_12693','dir=in','action=allow','protocol=TCP','localport=12693','profile=private') -Verb RunAs -Wait -PassThru; exit $process.ExitCode"
    if errorlevel 1 (
        echo [AVERTISSEMENT] La regle de pare-feu n'a pas ete creee.
        echo Relancez install_windows.bat ou ajoutez manuellement le port TCP 12693 au profil prive.
    )
)

echo.
echo Installation terminee.
echo Lancez ensuite : start_windows.bat
exit /b 0

:venv_error
echo [ERREUR] Impossible de creer l'environnement Python 3.11.
goto :error_exit

:dependency_error
echo [ERREUR] Installation des dependances impossible.
echo Si l'erreur concerne InsightFace, installez Microsoft C++ Build Tools
echo avec la charge de travail "Desktop development with C++", puis relancez ce script.
echo Le runtime GGUF doit provenir de la wheel CUDA 13.0 indiquee par le script.
goto :error_exit

:llama_portable_error
echo [ERREUR] Impossible de preparer le backend CPU portable de llama-cpp-python.
echo Ce backend evite les instructions AVX-512 incompatibles avec certains Core i9,
echo sans desactiver CUDA pour les embeddings.
echo Relancez install_windows.bat afin de retablir les DLL de la version %LLAMA_CPP_VERSION%.
goto :error_exit

:cuda_error
echo [ERREUR] PyTorch ne detecte pas la carte NVIDIA.
echo Installez le dernier pilote NVIDIA compatible puis relancez ce script.
goto :error_exit

:llama_cuda_error
echo [ERREUR] llama-cpp-python ne parvient pas a charger ses DLL CUDA.
echo Le script a recherche les DLL CUDA dans PyTorch et dans le pilote NVIDIA.
echo Dossier PyTorch :
echo   %TORCH_DLL_DIR%
echo Mettez a jour le pilote NVIDIA afin d'installer nvcudart_hybrid64.dll,
echo puis relancez install_windows.bat.
goto :error_exit

:llama_context_error
echo [ERREUR] Le runtime CUDA est present mais BGE-M3 ne peut pas charger ou calculer.
echo Le test utilise la fenetre complete de 8192 tokens et le meme chemin que le serveur.
echo Consultez la trace affichee ci-dessus puis relancez install_windows.bat.
goto :error_exit

:validation_error
echo [ERREUR] L'installation ou le dossier PuLID_models est incomplet.
echo Consultez les erreurs affichees ci-dessus, corrigez-les puis relancez ce script.
goto :error_exit

:error_exit
echo.
echo Appuyez sur une touche pour fermer cette fenetre.
pause >nul
exit /b 1
