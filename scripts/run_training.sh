#!/bin/bash
# 🔬 MicroSR — Training Launcher

set -e

GREEN="\033[92m"
YELLOW="\033[93m"
CYAN="\033[96m"
RESET="\033[0m"
BOLD="\033[1m"

echo -e "${BOLD}${CYAN}[Train] Initializing training run...${RESET}"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Detect CUDA status inside python
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")

# Default settings
EPOCHS=200
LAM=0.1
DATA_ROOT="data/BioSR"
RUN_DIR="runs/exp1"

# Allow command line overrides (e.g. --epochs 100)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift ;;
        --lam) LAM="$2"; shift ;;
        --data_root) DATA_ROOT="$2"; shift ;;
        --run_dir) RUN_DIR="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ "$CUDA_AVAILABLE" = "True" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo -e "${GREEN}✓ CUDA is active. Training on: ${GPU_NAME}${RESET}"
    
    # RTX 5060 has 8GB VRAM. Batch size 4 is perfect and safe with AMP.
    BATCH_SIZE=4
    NUM_WORKERS=2
    AMP_FLAG="--amp"
    DEVICE="cuda"
else
    echo -e "${YELLOW}⚠ CUDA is NOT active. Falling back to CPU training.${RESET}"
    echo -e "${YELLOW}  To prevent system lag/crashes, batch size is reduced to 2 and workers to 1.${RESET}"
    echo -e "${YELLOW}  Note: Diffusion model training on CPU will be extremely slow.${RESET}"
    
    BATCH_SIZE=2
    NUM_WORKERS=1
    AMP_FLAG=""
    DEVICE="cpu"
    
    # If running on CPU and not explicitly customized, restrict to 2 epochs for sanity check
    if [ "$EPOCHS" -eq 200 ]; then
        echo -e "${YELLOW}  Automatically capping to 2 epochs on CPU for quick pipeline testing.${RESET}"
        echo -e "${YELLOW}  If you really want 200 epochs on CPU, pass it explicitly: ./scripts/run_training.sh --epochs 200${RESET}"
        EPOCHS=2
    fi
fi

# Setup MLOps Offline modes by default if no login detected
WANDB_FLAG="--use_wandb"
MLFLOW_FLAG="--use_mlflow"

if python -c "import wandb; wandb.login()" 2>/dev/null; then
    echo -e "${GREEN}✓ logged into wandb. Logs will sync online.${RESET}"
else
    echo -e "${YELLOW}⚠ W&B not logged in. Saving runs offline (runs can be synced later with 'make wandb-sync').${RESET}"
    export WANDB_MODE=offline
fi

echo -e "\n${CYAN}Starting training with configuration:${RESET}"
echo -e "  - Device:      ${BOLD}${DEVICE}${RESET}"
echo -e "  - Epochs:      ${BOLD}${EPOCHS}${RESET}"
echo -e "  - Batch Size:  ${BOLD}${BATCH_SIZE}${RESET}"
echo -e "  - Lambda:      ${BOLD}${LAM}${RESET}"
echo -e "  - Data Root:   ${BOLD}${DATA_ROOT}${RESET}"
echo -e "  - Run Dir:     ${BOLD}${RUN_DIR}${RESET}"
echo -e "  - Workers:     ${BOLD}${NUM_WORKERS}${RESET}"
echo -e "  - AMP (Mixed): ${BOLD}${AMP_FLAG:-disabled}${RESET}"
echo -e "  - W&B Mode:    ${BOLD}${WANDB_MODE:-online}${RESET}"
echo -e "────────────────────────────────────────────────────────\n"

python train.py \
  --data_root "$DATA_ROOT" \
  --run_dir "$RUN_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lam "$LAM" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  $AMP_FLAG \
  $WANDB_FLAG \
  $MLFLOW_FLAG
