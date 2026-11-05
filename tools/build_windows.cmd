@echo off
python "%~dp0build_windows.py" %*
exit /b %errorlevel%
