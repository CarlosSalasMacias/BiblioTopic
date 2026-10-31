#!/usr/bin/env bash
# Primera vez: crea el entorno e instala dependencias (puede tardar 10-15 min)
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi
streamlit run app.py
