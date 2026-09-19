#!/bin/bash
# Launch the original terminal CLI
cd "$(dirname "$0")"
.venv/bin/python main.py "$@"
