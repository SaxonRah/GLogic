@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo === Cleaning Rocq/Coq build artifacts in %CD% ===
echo.

del /s /q *.vo 2>nul
del /s /q *.vos 2>nul
del /s /q *.vok 2>nul
del /s /q *.glob 2>nul
del /s /q *.aux 2>nul
del /s /q *.coq-native 2>nul
del /s /q *.d 2>nul

echo Clean complete.
pause
