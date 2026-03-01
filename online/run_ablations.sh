#!/bin/bash

# ==========================================
# GIOROM ABLATION STUDY
# ==========================================
PROJECT_ROOT=$(pwd)
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/train_giorom.py"

LOG_DIR="$PROJECT_ROOT/results"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/ablation_study.log"

# Define the explicit path to the .pt file for GIOROM
DATASET="owl"
GIOROM_PT="/data/pt_dataset/$DATASET/rollout_full.pt"

echo "Starting GIOROM Ablation Study..." > "$LOG_FILE"
echo "Timestamp: $(date)" >> "$LOG_FILE"

# ==========================================
# ABLATION 1: GRID SIZE (Fixed Sparsity = 20)
# ==========================================
echo "=== Starting Grid Size Ablation ===" | tee -a "$LOG_FILE"
SPARSITY_FIXED=20
GRID_SIZES=(20 32 64 82 100)

for grid in "${GRID_SIZES[@]}"; do
    echo "Running: Grid=$grid, Sparsity=$SPARSITY_FIXED..." | tee -a "$LOG_FILE"
    python3 "$PYTHON_SCRIPT" \
        --data "$GIOROM_PT" \
        --grid "$grid" \
        --sparsity "$SPARSITY_FIXED" \
        --param "Grid"
done

# ==========================================
# ABLATION 2: SPARSITY (Fixed Grid Size = 64)
# ==========================================
echo "=== Starting Sparsity Ablation ===" | tee -a "$LOG_FILE"
GRID_FIXED=64
SPARSITIES=(12 14 18 20 24 36 48 60 72 100)

for sparse in "${SPARSITIES[@]}"; do
    echo "Running: Grid=$GRID_FIXED, Sparsity=$sparse..." | tee -a "$LOG_FILE"
    python3 "$PYTHON_SCRIPT" \
        --data "$GIOROM_PT" \
        --grid "$GRID_FIXED" \
        --sparsity "$sparse" \
        --param "sampling" 
done

echo "Ablations complete. Generating plots..." | tee -a "$LOG_FILE"
python3 "$PROJECT_ROOT/scripts/plot_ablations.py"
echo "Done." | tee -a "$LOG_FILE"