@echo off
REM ============================================================
REM  Topo-Downscale — SIH Demo Launcher (Windows)
REM  Double-click this file OR run it from the project root.
REM ============================================================
echo.
echo  ===================================================
echo   Topo-Downscale  ^|  SIH 2024  ^|  Problem 26074
echo  ===================================================
echo.

REM Change to the project root (the folder containing this .bat)
cd /d "%~dp0.."

REM Run the Python launch script
python scripts\serve.py %*

REM Keep window open if an error occurred
if errorlevel 1 (
    echo.
    echo [ERROR] Launch failed. See message above.
    pause
)
