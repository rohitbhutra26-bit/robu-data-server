#!/bin/bash
cd "$(dirname "$0")"
pip install -r requirements.txt --break-system-packages -q
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
