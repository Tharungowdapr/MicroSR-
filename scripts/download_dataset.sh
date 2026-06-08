#!/bin/bash
# 🔬 MicroSR — Dataset Downloader

set -e

GREEN="\033[92m"
YELLOW="\033[93m"
CYAN="\033[96m"
RESET="\033[0m"
BOLD="\033[1m"

echo -e "${BOLD}${CYAN}[BioSR] Downloading entire dataset (all structures)...${RESET}"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run download script with all structures
python scripts/download_data.py \
  --structures MTs ER F-actin CCPs \
  --out_dir data/BioSR

echo -e "\n${BOLD}${GREEN}✓ Dataset fully downloaded and unpacked in data/BioSR/${RESET}"
