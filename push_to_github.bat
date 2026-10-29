@echo off
setlocal
cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
  echo Git is not installed. Download it from https://git-scm.com/download/win and run this again.
  pause & exit /b 1
)

set "E="
for /f "delims=" %%i in ('git config --global user.email') do set "E=%%i"
if not defined E (
  set /p GNAME=Your name for Git commits: 
  set /p GMAIL=Your GitHub email: 
  git config --global user.name "%GNAME%"
  git config --global user.email "%GMAIL%"
)

if not exist .git git init
git branch -M main
git add .
git commit -m "Initial commit: ScanTwin v3 by NEURA MED AI"
git remote remove origin >nul 2>nul
git remote add origin https://github.com/NeuraMed-AI-Works/scantwin.git
git push -u origin main

echo.
echo Done. Open https://github.com/NeuraMed-AI-Works/scantwin to check.
pause
