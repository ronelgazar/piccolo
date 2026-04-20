@echo off
set VENV=.venv\Scripts
%VENV%\python -m pip install pyinstaller>=6.0
%VENV%\pyinstaller --clean piccolo.spec
echo Build complete: dist\piccolo.exe
