@echo off
set VENV=.venv\Scripts
%VENV%\python -m pip install "pyinstaller>=6.0"
if errorlevel 1 exit /b %errorlevel%

%VENV%\pyinstaller --clean piccolo.spec
if errorlevel 1 exit /b %errorlevel%

echo Build complete: dist\piccolo.exe
