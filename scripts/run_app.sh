#!/bin/bash
# 🔬 MicroSR — App Launcher

set -e

GREEN="\033[92m"
CYAN="\033[96m"
RESET="\033[0m"
BOLD="\033[1m"

echo -e "${BOLD}${CYAN}[App] Starting Gradio web server...${RESET}"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Default model path
CKPT="runs/exp1/best_model.pt"

# Override default model path if provided
if [ "$#" -gt 0 ]; then
    CKPT="$1"
fi

python frontend/app.py --ckpt "$CKPT"
