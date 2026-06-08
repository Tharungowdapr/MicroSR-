#!/bin/bash
# 🔬 MicroSR — Setup Environment Script

set -e

GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
CYAN="\033[96m"
RESET="\033[0m"
BOLD="\033[1m"

echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║   🔬  MicroSR — Environment Setup Assistant              ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${RESET}"

# 1. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo -e "\n${CYAN}[1/4] Creating virtual environment (venv)...${RESET}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created in venv/${RESET}"
else
    echo -e "\n${GREEN}✓ Virtual environment already exists in venv/${RESET}"
fi

# Activate virtual environment
source venv/bin/activate

# 2. Upgrade pip & Install Dependencies
echo -e "\n${CYAN}[2/4] Installing dependencies from requirements.txt...${RESET}"
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest  # Ensure pytest is installed for testing

echo -e "${GREEN}✓ All packages installed successfully.${RESET}"

# 3. MLOps check
echo -e "\n${CYAN}[3/4] Verifying MLOps tools...${RESET}"
if python -c "import wandb" &> /dev/null; then
    echo -e "${GREEN}✓ Weights & Biases (wandb) is installed.${RESET}"
else
    echo -e "${YELLOW}⚠ wandb not found, installing...${RESET}"
    pip install wandb
fi

if python -c "import mlflow" &> /dev/null; then
    echo -e "${GREEN}✓ MLflow is installed.${RESET}"
else
    echo -e "${YELLOW}⚠ mlflow not found, installing...${RESET}"
    pip install mlflow
fi

# 4. Check GPU & CUDA Driver compatibility
echo -e "\n${CYAN}[4/4] Performing GPU and CUDA compatibility check...${RESET}"
RUNNING_KERNEL=$(uname -r)
echo -e "  Current running kernel: ${BOLD}${RUNNING_KERNEL}${RESET}"

# Test PyTorch CUDA status
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")

if [ "$CUDA_AVAILABLE" = "True" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo -e "${GREEN}✓ CUDA is AVAILABLE! Detected GPU: ${GPU_NAME}${RESET}"
else
    echo -e "${YELLOW}⚠ PyTorch reports CUDA is NOT available.${RESET}"
    
    # Diagnose Nvidia Driver / Kernel mismatch
    if lspci | grep -i nvidia &> /dev/null; then
        echo -e "${YELLOW}→ Nvidia GPU hardware detected via PCI bus.${RESET}"
        
        # Check if nvidia-smi runs
        if nvidia-smi &> /dev/null; then
            echo -e "${YELLOW}→ nvidia-smi is working. Your PyTorch version might be compiled without CUDA support.${RESET}"
            echo -e "  Re-installing CUDA-enabled PyTorch inside venv..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
        else
            echo -e "${RED}✗ nvidia-smi failed to communicate with the Nvidia driver.${RESET}"
            echo -e "${RED}  This is due to a kernel module mismatch. Your running kernel is ${RUNNING_KERNEL},${RESET}"
            echo -e "${RED}  but Nvidia modules are compiled for a different kernel version.${RESET}"
            echo -e "\n${BOLD}${YELLOW}🔧 TO RESOLVE THIS (Required to train on your RTX 5060 GPU):${RESET}"
            echo -e "  Please open a terminal and run the following command to install the correct kernel modules:"
            echo -e "  ${BOLD}sudo apt-get update && sudo apt-get install -y linux-modules-nvidia-580-open-${RUNNING_KERNEL}${RESET}"
            echo -e "\n  After installing, try running ${BOLD}sudo modprobe nvidia${RESET} or restart your machine."
        fi
    else
        echo -e "${YELLOW}→ No Nvidia GPU detected on PCI bus. Training will run on CPU (very slow).${RESET}"
    fi
fi

echo -e "\n${BOLD}${GREEN}🎉 Environment setup completed!${RESET}"
echo -e "Activate environment using: ${BOLD}source venv/bin/activate${RESET}"
echo -e "Download dataset using:     ${BOLD}./scripts/download_dataset.sh${RESET}"
echo -e "Launch training using:      ${BOLD}./scripts/run_training.sh${RESET}"
echo -e "Launch Gradio app using:    ${BOLD}./scripts/run_app.sh${RESET}"
echo -e "────────────────────────────────────────────────────────"
