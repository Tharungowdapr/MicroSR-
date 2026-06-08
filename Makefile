# ═══════════════════════════════════════════════════════
#  MicroSR — Makefile
#  Plug-and-play commands for the entire project
#  Usage: make <command>
# ═══════════════════════════════════════════════════════

.PHONY: help setup download train test app deploy clean lint

# Default target
.DEFAULT_GOAL := help

# Dynamic python detection: use venv if present, otherwise system python
PYTHON    := $(shell [ -d venv ] && echo "venv/bin/python" || echo "python3")
MLFLOW    := $(shell [ -d venv ] && echo "venv/bin/mlflow" || echo "mlflow")

# ─── Config (override via CLI: make train EPOCHS=100) ───
DATA_DIR    := data/BioSR
RUN_DIR     := runs/exp1
CKPT        := $(RUN_DIR)/best_model.pt
EPOCHS      := 200
BATCH       := 4
LAM         := 0.1
PORT        := 7860
HF_REPO     := your-username/microsr   # ← change this

# ─────────────────────────────────────────────────────────
help:   ## Show all available commands
	@echo ""
	@echo "  🔬 MicroSR — Physics-Constrained Microscopy Super-Resolution"
	@echo "  ──────────────────────────────────────────────────────────────"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ \
	  { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# ─────────────────────────────────────────────────────────
setup:  ## Install all dependencies (creates venv and checks GPU)
	@chmod +x scripts/setup_env.sh
	@./scripts/setup_env.sh

# ─────────────────────────────────────────────────────────
download:  ## Download entire BioSR dataset (MTs, ER, F-actin, CCPs)
	@chmod +x scripts/download_dataset.sh
	@./scripts/download_dataset.sh

download-all: download ## Download all BioSR structures (alias for download)

# ─────────────────────────────────────────────────────────
train:  ## Train the model (automatically calibrates GPU/CPU batch sizes)
	@chmod +x scripts/run_training.sh
	@./scripts/run_training.sh --epochs $(EPOCHS) --lam $(LAM) --data_root $(DATA_DIR) --run_dir $(RUN_DIR)

train-fast:  ## Quick training (50 epochs, for testing pipeline)
	$(PYTHON) train.py \
	  --data_root $(DATA_DIR) \
	  --run_dir   runs/fast \
	  --epochs    50 \
	  --batch_size 4 \
	  --lam       0.1 \
	  --max_train 500

train-ablation:  ## Run all lambda ablation experiments
	@echo "\n[Ablation] Training 5 models with different lambda values..."
	for lam in 0.0 0.01 0.05 0.1 0.5; do \
	  echo "\n--- Lambda = $$lam ---"; \
	  $(PYTHON) train.py \
	    --data_root $(DATA_DIR) \
	    --run_dir   runs/ablation/lam_$$lam \
	    --run_name  microsr-lam-$$lam \
	    --epochs    $(EPOCHS) \
	    --lam       $$lam \
	    --amp; \
	done

# ─────────────────────────────────────────────────────────
test:  ## Evaluate trained model on all metrics
	@echo "\n[Test] Evaluating model..."
	$(PYTHON) test.py \
	  --ckpt      $(CKPT) \
	  --data_root $(DATA_DIR) \
	  --out_dir   test_results
	@echo "[Test] ✓ Results in test_results/"

# ─────────────────────────────────────────────────────────
app:  ## Launch the Gradio frontend locally
	@chmod +x scripts/run_app.sh
	@./scripts/run_app.sh $(CKPT)

app-share:  ## Launch with public HuggingFace tunnel URL
	$(PYTHON) frontend/app.py \
	  --ckpt  $(CKPT) \
	  --port  $(PORT) \
	  --share

# ─────────────────────────────────────────────────────────
deploy:  ## Deploy to HuggingFace Spaces
	@echo "\n[Deploy] Pushing to HuggingFace Spaces: $(HF_REPO)"
	@echo "Make sure you've run: huggingface-cli login"
	cp README_HF.md README.md
	git add .
	git commit -m "Deploy MicroSR"
	git push
	@echo "[Deploy] ✓ Live at: https://huggingface.co/spaces/$(HF_REPO)"

mlflow-ui:  ## Open MLflow experiment dashboard
	$(MLFLOW) ui --port 5000

wandb-sync:  ## Sync offline W&B runs
	@if [ -d venv ]; then source venv/bin/activate; fi; \
	wandb sync --sync-all

# ─────────────────────────────────────────────────────────
test-unit:  ## Run unit tests (no GPU or data needed)
	@echo "\n[Test] Running unit tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "[Test] ✓ Done"

lint:  ## Check code style
	@echo "\n[Lint] Checking code..."
	$(PYTHON) -m flake8 src/ train.py test.py --max-line-length 100 --ignore E501

clean:  ## Remove generated files (keeps data and checkpoints)
	@echo "\n[Clean] Removing cache and temp files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"     -delete 2>/dev/null || true
	rm -rf .pytest_cache/ 2>/dev/null || true
	@echo "[Clean] ✓ Done"

clean-all:  ## Remove everything including runs (CAREFUL)
	make clean
	rm -rf runs/ test_results/ wandb/ mlruns/ 2>/dev/null || true
	@echo "[Clean] ✓ All outputs removed"

