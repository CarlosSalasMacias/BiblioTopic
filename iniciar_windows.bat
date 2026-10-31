@echo off
REM Primera vez: crea el entorno e instala dependencias (puede tardar 10-15 min)
cd /d %~dp0
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate
)
streamlit run app.py
