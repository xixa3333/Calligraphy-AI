@echo off
setlocal
pushd "%~dp0\.."
set "PYTHONPATH=src"
set "PYTHONIOENCODING=utf-8"
set "MPLBACKEND=Agg"
"..\.venv\Scripts\python.exe" -u scripts\train.py
popd
endlocal
