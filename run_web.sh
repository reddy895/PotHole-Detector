#!/bin/bash
# Launch the Web UI (Live Video + Upload Video)
cd "$(dirname "$0")"
echo "Starting AI Pothole Detection Web App..."
echo "Open http://localhost:5000 in your browser"
.venv/bin/python app.py
