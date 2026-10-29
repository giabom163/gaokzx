@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem ============================================================
rem  Wenku pipeline launcher
rem  Always runs with the project venv interpreter, so the
rem  "[env] re-exec" notice never appears.
rem
rem  Usage:
rem    double-click          -> interactive menu
rem    run.cmd --step all    -> run steps from command line
rem ============================================================
title Wenku Pipeline
set "PY_EXE=C:\Users\Lenovo\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if not exist "%PY_EXE%" (
    if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
        set "PY_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
    ) else if exist "%~dp0.venv\Scripts\python.exe" (
        set "PY_EXE=%~dp0.venv\Scripts\python.exe"
    ) else (
        set "PY_EXE=python"
    )
)
"%PY_EXE%" "%~dp0pipeline.py" %*
set RC=%ERRORLEVEL%
echo ==================== EXIT=%RC% ====================
pause
