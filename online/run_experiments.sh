#!/bin/bash
set -e
set -u
set -o pipefail
trap 'echo "Error at line $LINENO processing $DATASET"; exit 1' ERR

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT=$(pwd)

# --- RENDERING CONFIG ---
DO_RENDER=false # Set to true to enable Blender rendering, takes a long time per model

# EXPLICIT DATA BASES
H5_DATA_BASE="/data/CROM_dataset/CROM_Ready_Data"
PT_DATA_BASE="/data/pt_dataset"
OFFLINE_BASE="/data/CROM_offline_training"

BLENDER_EXE="/data/blender-4.0.2-linux-x64/blender"
RENDER_POINTS_SCRIPT="$PROJECT_ROOT/src/visualizations/render_points.py"
RENDER_VISUALS_SCRIPT="$PROJECT_ROOT/src/visualizations/render_visuals.py"

# Blender Material Mapping
declare -A DATASET_MAT_MAP
DATASET_MAT_MAP["nclaw_Water"]="WATER"
DATASET_MAT_MAP["nclaw_Plasticine"]="PLASTICINE"
DATASET_MAT_MAP["nclaw_Sand"]="SAND"
DATASET_MAT_MAP["owl"]="ELASTIC"

# Datasets and Sparsities
declare -A DATASET_SPARSITY_MAP
DATASET_SPARSITY_MAP["nclaw_Water"]="9"
DATASET_SPARSITY_MAP["nclaw_Plasticine"]="4"
DATASET_SPARSITY_MAP["nclaw_Sand"]="10"
DATASET_SPARSITY_MAP["owl"]="30"

# Sweep Variables
LATENT_DIMS=("32")
DATASETS=("nclaw_Plasticine")
MODELS=("giorom" "gno" "coral" "colora" "dino" "pca" "crom" "licrom")

BASE_RADIUS=0.015
BASE_DIM=32

# ==========================================
# HELPER: Find Offline CROM Weights
# ==========================================
get_weights_path() {
    local TYPE=$1; local DS=$2; local SP=$3
    local SEARCH_DIR="$OFFLINE_BASE/outputs_${SP}_${DS}/weights"
    find "$SEARCH_DIR" -name "*_${TYPE}.pt" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || echo ""
}

# ==========================================
# HELPER: Benchmark + Render + Video
# ==========================================
run_bench_and_render() {
    local MODEL_TYPE=$1
    local EXTRA_ARGS=$2 
    
    local OBJ_DIR="$PROJECT_ROOT/visualizations/media/obj_output/${MODEL_TYPE}/${DATASET}"
    local VID_DIR="$PROJECT_ROOT/visualizations/media/videos"
    local RENDER_DIR="$OBJ_DIR/rendered"
    
    echo "========================================================"
    echo " PIPELINE: $MODEL_TYPE on $DATASET"
    echo "========================================================"

    if [[ "$MODEL_TYPE" == "giorom" ]]; then
        python3 "$PROJECT_ROOT/scripts/train_giorom.py" \
            --data "$GIOROM_PT" \
            --sparsity "$SPARSITY" \
            --grid 64 \
            --obj_out_dir "$OBJ_DIR" \
            $EXTRA_ARGS
    else
        python3 "$PROJECT_ROOT/scripts/benchmark_baselines.py" \
            -model_type "$MODEL_TYPE" \
            -data_root "$DATA_PATH" \
            -sparsity "$SPARSITY" \
            -enc "$ENC_PATH" \
            -dec "$DEC_PATH" \
            -latent_dim "$LATENT_DIM" \
            --save_obj \
            --obj_out_dir "$OBJ_DIR" \
            $EXTRA_ARGS
    fi

    if [ "$DO_RENDER" = false ]; then return; fi
    if [ ! -d "$OBJ_DIR" ] || [ -z "$(ls -A "$OBJ_DIR" 2>/dev/null)" ]; then return; fi

    mkdir -p "$RENDER_DIR" "$VID_DIR"
    MATERIAL=${DATASET_MAT_MAP[$DATASET]}

    if [[ "$MATERIAL" == "SAND" ]]; then
        cat <<EOF > "$PROJECT_ROOT/render_config_tmp.json"
{
    "object": { "location": {"x": 0,"y": 0,"z": 0}, "rotation": {"x": 180,"y": 270,"z": 0}, "scale": {"x": 1,"y": 1,"z": 1} },
    "box": { "location": {"x": 0,"y": 0,"z": 0}, "rotation": {"x": 0,"y": 0,"z": 0}, "scale": {"x": 1,"y": 1,"z": 1} },
    "camera": { "location": {"x": 2.2,"y": -1.4,"z": 1.5}, "rotation": {"x": 60,"y": 0,"z": 60} },
    "pointsToVolumeDensity": 0.5, "pointsToVolumeVoxelAmount": 128, "pointsToVolumeRadius": 0.02
}
EOF
        "$BLENDER_EXE" -b -P "$RENDER_VISUALS_SCRIPT" -- -b "$MATERIAL" "$PROJECT_ROOT/render_config_tmp.json" 1 0 0 "$OBJ_DIR" "$RENDER_DIR"
        ffmpeg -y -framerate 30 -i "$RENDER_DIR/pred_%04d.obj.png" -vf "transpose=1" -c:v libx264 -pix_fmt yuv420p "$VID_DIR/${DATASET}_${MODEL_TYPE}.mp4"
    else
        "$BLENDER_EXE" -b -P "$RENDER_POINTS_SCRIPT" -- "$MATERIAL" "$OBJ_DIR" "$RENDER_DIR"
        ffmpeg -y -framerate 30 -i "$RENDER_DIR/pred_%04d.png" -c:v libx264 -pix_fmt yuv420p "$VID_DIR/${DATASET}_${MODEL_TYPE}.mp4"
    fi
}

# ==========================================
# MAIN LOOP
# ==========================================
for LATENT_DIM in "${LATENT_DIMS[@]}"; do
    GNO_RADIUS=$(python3 -c "print(f'{ $BASE_RADIUS * ($LATENT_DIM / $BASE_DIM)**(1/3) :.5f}')")

    for DATASET in "${DATASETS[@]}"; do
        SPARSITY=${DATASET_SPARSITY_MAP[$DATASET]}
        DATA_PATH="$H5_DATA_BASE/$DATASET"
        GIOROM_PT="$PT_DATA_BASE/$DATASET/rollout_full.pt"
        
        echo "Processing $DATASET | Sparsity: $SPARSITY | Latent Dim: $LATENT_DIM"

        # ----------------------------------
        # 1. OFFLINE CROM (Retrain if needed)
        # ----------------------------------
        # cd "$OFFLINE_BASE"
        # python3 run.py -mode train -d "$DATA_PATH" -initial_lr 0.0001 -epo 200 -lr 1 -batch_size 16 \
        #     -lbl "$LATENT_DIM" -scale_mlp 4 -ks 9 -strides 4 \
        #     -siren_enc -enc_omega_0 30.0 -siren_dec -dec_omega_0 30.0 -sparsity "$SPARSITY"
        # cd "$PROJECT_ROOT"

        ENC_PATH=$(get_weights_path "enc" "$DATASET" "$SPARSITY")
        DEC_PATH=$(get_weights_path "dec" "$DATASET" "$SPARSITY")

        # ----------------------------------
        # 2. ONLINE TRAINING (Full Flags Restored)
        # ----------------------------------
        # DINo
        python3 "$PROJECT_ROOT/scripts/train_dino.py" -data "$DATA_PATH" -enc "$ENC_PATH" -dec "$DEC_PATH" \
            -out "$PROJECT_ROOT/checkpoints/dino" -epochs 100 -batch_size 4 -seq_len 20 -hidden_dim 128 -sparsity "$SPARSITY"

        # CoLoRA
        python3 "$PROJECT_ROOT/scripts/train_colora.py" --data_root "$DATA_PATH" --enc "$ENC_PATH" \
            --save_dir "$PROJECT_ROOT/checkpoints/colora" --epochs 100 --batch_size 4 --hidden_dim 128 --sparsity "$SPARSITY"
            
        # CORAL
        python3 "$PROJECT_ROOT/scripts/train_coral.py" --data_root "$DATA_PATH" --enc "$ENC_PATH" \
            --save_dir "$PROJECT_ROOT/checkpoints/coral_dynamics" --epochs 3 --batch_size 4 --seq_len 20 --hidden_dim 128 --sparsity "$SPARSITY"
            

        # GNO
        # THIS IS COMMENTED DUE TO CHANGES IN NEURAL OPERATOR LIBRARY CODEBASE SINCE LAST TRAINING. REQUIRES UPDATING TO NEW CODE STRUCTURE
        # python3 "$PROJECT_ROOT/scripts/train_gno.py" -data "$DATA_PATH" -out "$PROJECT_ROOT/checkpoints/gno" \
        #     -epochs 3 -lr 0.001 -radius "$GNO_RADIUS" -sparsity "$SPARSITY"

        # ----------------------------------
        # 3. BENCHMARKS
        # ----------------------------------
        DINO_CKPT=$(find "$PROJECT_ROOT/checkpoints/dino" -name "*${SPARSITY}_${DATASET}*.ckpt" | sort | tail -n 1 || echo "")
        COLORA_CKPT=$(find "$PROJECT_ROOT/checkpoints/colora" -name "*${SPARSITY}_${DATASET}*.pt" | sort -n | tail -1 || echo "")
        CORAL_CKPT=$(find "$PROJECT_ROOT/checkpoints/coral_dynamics" -name "*${SPARSITY}_${DATASET}*.pt" | sort -n | tail -1 || echo "")
        GNO_CKPT=$(find "$PROJECT_ROOT/checkpoints/gno" -name "*${SPARSITY}_${DATASET}*.pt" | sort | tail -n 1 || echo "")

        run_bench_and_render "giorom" ""
        #[ -n "$GNO_CKPT" ] && run_bench_and_render "gno" "-gno_ckpt $GNO_CKPT -radius $GNO_RADIUS" 
        # GNO rendering is currently disabled due to changes in the underlying neural operator codebase that require updates to the training script and checkpoint structure. Once those updates are made, this line can be uncommented to include GNO in the benchmarking and rendering pipeline.
        [ -n "$CORAL_CKPT" ] && run_bench_and_render "coral" "-coral_online_ckpt $CORAL_CKPT"
        [ -n "$COLORA_CKPT" ] && run_bench_and_render "colora" "-ckpt $COLORA_CKPT"
        [ -n "$DINO_CKPT" ] && run_bench_and_render "dino" "-ckpt $DINO_CKPT"
        run_bench_and_render "pca" ""
        run_bench_and_render "crom" ""
        run_bench_and_render "licrom" ""
    done
done