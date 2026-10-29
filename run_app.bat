@echo off
echo Starting ScanTwin Digital Twin Dashboard...
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
python -m streamlit run app.py
pause
