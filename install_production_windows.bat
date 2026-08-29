@echo off
setlocal EnableExtensions

call "%~dp0install_windows.bat" --production %*
exit /b %ERRORLEVEL%
