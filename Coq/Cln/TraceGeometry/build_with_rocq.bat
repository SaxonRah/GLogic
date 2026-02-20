@echo off
setlocal EnableExtensions

set "ROCQROOT=C:\Rocq-Platform~9.0~2025.08"
set "PATH=%ROCQROOT%\bin;%PATH%"

REM ---- Prefer ROCQLIB (Rocq 9), fall back to COQLIB ----
set "ROCQLIB="
if exist "%ROCQROOT%\lib\coq\theories\Init\Prelude.vo" set "ROCQLIB=%ROCQROOT%\lib\coq"
if not defined ROCQLIB if exist "%ROCQROOT%\lib\rocq\theories\Init\Prelude.vo" set "ROCQLIB=%ROCQROOT%\lib\rocq"
if not defined ROCQLIB if exist "%ROCQROOT%\share\coq\theories\Init\Prelude.vo" set "ROCQLIB=%ROCQROOT%\share\coq"
if not defined ROCQLIB if exist "%ROCQROOT%\share\rocq\theories\Init\Prelude.vo" set "ROCQLIB=%ROCQROOT%\share\rocq"

if not defined ROCQLIB goto :nolib

REM If any tools still look for COQLIB, set it too.
set "COQLIB=%ROCQLIB%"

cd /d "%~dp0"

echo === Rocq build in %CD% ===
where coqc
echo ROCQLIB=%ROCQLIB%
echo.

REM ---- Compile in dependency order ----
call :compile TraceGeometryCore.v || goto :fail
call :compile TraceGeometryToyNat.v || goto :fail
call :compile TraceGeometryMachine.v || goto :fail
call :compile TraceGeometryExplosion.v || goto :fail
call :compile TraceGeometryStep5Strong.v || goto :fail
call :compile TraceGeometryPipeline.v || goto :fail
call :compile TraceGeometryStep7Assemble.v || goto :fail

echo.
echo BUILD SUCCESS
pause
exit /b 0

:compile
echo [coqc] %~1
coqc -Q . TraceGeometry "%~1"
exit /b %errorlevel%

:nolib
echo ERROR: Could not locate Rocq prelude (Prelude.vo) under %ROCQROOT%.
echo Try:
echo   dir /s /b "%ROCQROOT%\Prelude.vo"
pause
exit /b 1

:fail
echo.
echo BUILD FAILED
pause
exit /b 1
