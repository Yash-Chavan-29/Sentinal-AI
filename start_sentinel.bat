@echo off
title Sentinel-AI Live Surveillance System
echo ========================================================
echo       Starting Sentinel-AI Live Threat Detection
echo ========================================================
echo.
cd /d "%~dp0"
Sentinel-AI\.venv\Scripts\python.exe run.py %*
pause
