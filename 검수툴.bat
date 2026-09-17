@echo off
chcp 65001 >nul
cd /d "%~dp0"
"C:\Users\easyh\AppData\Local\Programs\Python\Python311\python.exe" scripts\review_crops.py
pause
