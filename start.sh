#!/bin/bash
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt -q
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
