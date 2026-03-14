#!/bin/bash
# ================================================================
# MHR2SMPL Multi-View Pipeline
# ================================================================
# cd /home/jiawei/timingyang/CLEAN/_GITHUB/Fast_sam-3d-body && bash mhr2smpl/run_multiview_pipeline.sh 2>&1 | tee mhr2smpl/pipeline.log
# Pipeline:
#   Step 0:  SAM-3D-Body inference → sample_*.npz  (3DPW, EMDB)
#   Step 1a: 3DPW/EMDB fitted data collection
#   Step 1b: RICH/AIST multi-view data collection (GT SMPL)
#   Step 1c: Merge all 4 datasets → pairs_all_merged.npz
#   Step 2:  Multi-view network training
#   Step 3:  Evaluation
#   Smooth:  Train Pose Corrector (SmootherMLP on AMASS)
#   Demo:    Visualization (1V/2V/3V comparison)
#
# Usage:
#   bash mhr2smpl/run_multiview_pipeline.sh          # full pipeline
#   bash mhr2smpl/run_multiview_pipeline.sh 2>&1 | tee mhr2smpl/pipeline.log
#
# ================================================================

set -e

# ============================================================
# Step Control (1=run, 0=skip)
# ============================================================
RUN_STEP0=0                        # SAM-3D-Body inference
RUN_STEP1=0                        # Data collection + merge
RUN_STEP2=0                        # Multi-view training
RUN_STEP3=0                        # Evaluation
RUN_SMOOTH=1                       # SmootherMLP training
RUN_DEMO=1                         # Visualization demo

# ============================================================
# Training Config
# ============================================================
MAX_SAMPLES=30000                  # Max samples per dataset (-1=all)
EPOCHS=500                         # Multi-view network epochs
LR=1e-3                            # Learning rate
BATCH_SIZE=64                      # Training batch size
NUM_VIEWS_SELECT=2                 # Views to select during training
SV_LOSS_WEIGHT=0.5                 # Single-view loss weight

# ============================================================
# SmootherMLP Config
# ============================================================
SMOOTHER_WINDOW=5                  # Temporal window size
SMOOTHER_HIDDEN="512 256"          # Hidden layer dims
SMOOTHER_NOISE=0.02                # Noise std for training
SMOOTHER_EPOCHS=100                # Training epochs
SMOOTHER_FOOT_WEIGHT=0.1           # Foot contact loss weight

# ============================================================
# Data Collection Config
# ============================================================
NUM_VERTS=1500                     # Sampled vertices
MAX_VIEWS=4                        # Max views per sample
AIST_FRAME_STRIDE=18               # Frame stride for AIST

# ============================================================
# Demo Config
# ============================================================
AIST_MOTION=gBR_sBM_cAll_d04_mBR0_ch01
AIST_CAMS="01 05 03"
RICH_SCENE=Gym_010_lunge1
RICH_CAMS="01 04 05"
DEMO_MAX_FRAMES=450

# ============================================================
# SAM-3D-Body Inference Config (Step 0)
# ============================================================

# Core Performance
export GPU_HAND_PREP=1              # GPU hand preprocessing (faster)
export LAYER_DTYPE=fp32             # Layer dtype: fp32
# Multi-person scenarios require fp32; sam3dbody defaults to fp32 as well
export SKIP_KEYPOINT_PROMPT=1       # Skip keypoint prompt encoding
export IMG_SIZE=384                 # Input Backbone image size 448 (0=original 512)
# Backbone+decoder IMG_SIZE defaults to 512; important -- too small leads to inaccurate predictions!

# torch.compile Optimization
export USE_COMPILE=1                # Enable torch.compile
export USE_COMPILE_BACKBONE=1       # Compile backbone (DINOv3)
export DECODER_COMPILE=1            # Compile decoder
# export INTERM_COMPILE=1           # Compile intermediate layers (default=1)
export COMPILE_MODE=reduce-overhead  # Compile mode: default, reduce-overhead, max-autotune
export COMPILE_WARMUP_BATCH_SIZES=1  # Warmup batch sizes

# CUDA Graph
export MHR_USE_CUDA_GRAPH=0         # MHR CUDA Graph (0=off, 1=on)

# Intermediate Layer Prediction
export KEYPOINT_PROMPT_INTERM_INTERVAL=999  # Keypoint prompt interval (999=disable)
# export KEYPOINT_PROMPT_INTERM_LAYERS=0,1,2,3  # Specific layers for keypoint prompt
export BODY_INTERM_PRED_LAYERS=0,1,2        # Body decoder intermediate layers
# Fewer layers = faster decoder; reducing layers significantly improves speed. Optimal: 0,1,2
export HAND_INTERM_PRED_LAYERS=999          # Hand decoder intermediate layers
# Fewer layers = faster decoder; reducing layers significantly improves speed. Optimal: 0,1
# export INTERM_PRED_LAYERS=0,1,2,3         # Generic intermediate layers (overridden by BODY/HAND)
# export INTERM_PRED_INTERVAL=1             # Generic interval (overridden by BODY/HAND)

# MHR Head
export MHR_NO_CORRECTIVES=1         # Disable correctives (faster)
# export MOMENTUM_ENABLED=1         # Enable momentum (default=1)

# FOV Estimator (MoGe2)
export FOV_TRT=1                    # Use TensorRT encoder
export FOV_FAST=1                   # Fast mode (skip normal_head)
export FOV_MODEL=s                  # Model size: s(35M), b(104M), l(331M)
export FOV_LEVEL=0                  # Resolution level: 0-9 (0=1200 tokens, 9=3600 tokens)
# export FOV_SIZE=512               # Input size (auto=512 when FOV_TRT=1)
# TRT Path: checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine

# Backbone TensorRT (DINOv3) - requires engine file
# export USE_TRT_BACKBONE=1
# export TRT_BACKBONE_PATH=./checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine

# Decoder
# export PARALLEL_DECODERS=1        # Parallel body/hand decoders (default=1)

# Debug (set to 1 to enable)
export DEBUG_NAN=0
export DEBUG_HAND_PREP=0
export DEBUG_BACKBONE_INPUT=0
export INTERM_TIMING=0              # Intermediate layer timing

# ============================================================
# Environment
# ============================================================
source /home/jiawei/miniforge3/etc/profile.d/conda.sh
conda activate sam_3d_body

# ============================================================
# Paths (auto-derived, no need to change)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MV_DIR="${SCRIPT_DIR}/multi_view"
DATA_DIR="${SCRIPT_DIR}/data"
SMOOTH_DIR="${SCRIPT_DIR}/smooth"

MHR_CONV_DIR="/home/jiawei/timingyang/sam-3d-body/eval_ECCV/MHR/tools/mhr_smpl_conversion"
PIXI="/home/jiawei/.pixi/bin/pixi"
MANIFEST="/home/jiawei/timingyang/sam-3d-body/eval_ECCV/MHR/pyproject.toml"
PIXI_RUN="${PIXI} run --manifest-path ${MANIFEST} python"

CONDA_ENV="/home/jiawei/miniforge3/envs/sam_3d_body"
CONDA_PYTHON="${CONDA_ENV}/bin/python"

SMPL_MODEL="${DATA_DIR}/SMPL_NEUTRAL.pkl"
SMPL_MODEL_REL="./data/SMPL_NEUTRAL.pkl"

EMDB_DATA="/home/jiawei/timingyang/sam-3d-body/eval_ECCV/dataset/EMDB/data"
DPW_DATA="/home/jiawei/timingyang/sam-3d-body/eval_ECCV/dataset/3DPW"
EMDB_SAMPLES="${DATA_DIR}/outputs_EMDB/samples"
DPW_SAMPLES="${DATA_DIR}/outputs_3dpw/samples"
RICH_SAMPLES="/home/jiawei/timingyang/sam-3d-body/eval_ECCV/outputs_RICH/samples"
AIST_VIDEO_DIR="/home/jiawei/timingyang/sam-3d-body/eval_ECCV/dataset/AIST/videos"
AIST_DIR="/home/jiawei/timingyang/sam-3d-body/eval_ECCV/dataset/AIST"

EXP_DIR="${SCRIPT_DIR}/experiments/multiview_n${MAX_SAMPLES}_e${EPOCHS}"
SMOOTHER_CKPT_DIR="${SCRIPT_DIR}/experiments/smoother_w${SMOOTHER_WINDOW}"
AMASS_DATA="${SMOOTH_DIR}/data/amass_joints.npz"
YOLO_MODEL="${PROJECT_DIR}/checkpoints/yolo/yolo11m-pose.engine"

# ============================================================
# Print Config
# ============================================================
echo "========================================================"
echo "  MHR2SMPL Multi-View Pipeline"
echo "========================================================"
echo "  Steps:  0=${RUN_STEP0} 1=${RUN_STEP1} 2=${RUN_STEP2} 3=${RUN_STEP3} smooth=${RUN_SMOOTH} demo=${RUN_DEMO}"
echo "  Samples:    ${MAX_SAMPLES}"
echo "  Epochs:     ${EPOCHS} (lr=${LR}, bs=${BATCH_SIZE})"
echo "  Views:      select=${NUM_VIEWS_SELECT}, max=${MAX_VIEWS}"
echo "  Smoother:   window=${SMOOTHER_WINDOW}, epochs=${SMOOTHER_EPOCHS}"
echo "  Output:     ${EXP_DIR}"
echo "========================================================"

mkdir -p "${DATA_DIR}" "${EXP_DIR}"


# ============================================================
# Step 0: SAM-3D-Body inference → sample_*.npz
# ============================================================
if [ "${RUN_STEP0}" = "1" ]; then
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Step 0] Running SAM-3D-Body inference..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "${PROJECT_DIR}"

# EMDB
${CONDA_PYTHON} "${MV_DIR}/step0_infer_EMDB.py" \
    --data_dir "${EMDB_DATA}" \
    --output_dir "${DATA_DIR}/outputs_EMDB" \
    --detector_model "${YOLO_MODEL}" \
    --max_samples ${MAX_SAMPLES}

# 3DPW
${CONDA_PYTHON} "${MV_DIR}/step0_infer_3dpw.py" \
    --data_dir "${DPW_DATA}" \
    --output_dir "${DATA_DIR}/outputs_3dpw" \
    --detector_model "${YOLO_MODEL}" \
    --split test \
    --max_samples ${MAX_SAMPLES}
fi


# ============================================================
# Step 1: Data collection + merge
# ============================================================
if [ "${RUN_STEP1}" = "1" ]; then

# Step 1a: 3DPW + EMDB fitted
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Step 1a] Collecting fitted pairs: EMDB + 3DPW..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "${MHR_CONV_DIR}"

${PIXI_RUN} "${MV_DIR}/step1a_collect_fitted.py" \
    --input_dir "${EMDB_SAMPLES}" \
    --smpl_model "${SMPL_MODEL_REL}" \
    --output_path "${DATA_DIR}/pairs_EMDB_fitted.npz" \
    --max_samples ${MAX_SAMPLES} \
    --num_sampled_verts ${NUM_VERTS} \
    --max_views ${MAX_VIEWS} \
    --batch_size 256 \
    --use_fitted --no_viz

${PIXI_RUN} "${MV_DIR}/step1a_collect_fitted.py" \
    --input_dir "${DPW_SAMPLES}" \
    --smpl_model "${SMPL_MODEL_REL}" \
    --output_path "${DATA_DIR}/pairs_3dpw_fitted.npz" \
    --max_samples ${MAX_SAMPLES} \
    --num_sampled_verts ${NUM_VERTS} \
    --max_views ${MAX_VIEWS} \
    --batch_size 256 \
    --use_fitted --no_viz

# Step 1b: RICH multi-view (GT SMPL)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Step 1b] Collecting RICH multi-view pairs (GT)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "${MHR_CONV_DIR}"

${PIXI_RUN} "${MV_DIR}/step1b_collect_RICH.py" \
    --input_dirs "${RICH_SAMPLES}" \
    --smpl_model "${SMPL_MODEL_REL}" \
    --output_path "${DATA_DIR}/mv_pairs_RICH.npz" \
    --max_views ${MAX_VIEWS} \
    --num_sampled_verts ${NUM_VERTS} \
    --max_samples ${MAX_SAMPLES}

# Step 1b: AIST multi-view (GT SMPL + real Stage1)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Step 1b] Collecting AIST multi-view pairs (GT + real Stage1)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "${PROJECT_DIR}"

${CONDA_PYTHON} "${MV_DIR}/step1b_collect_AIST.py" \
    --video_dir "${AIST_VIDEO_DIR}" \
    --aist_dir "${AIST_DIR}" \
    --output_path "${DATA_DIR}/mv_pairs_AIST_stage1.npz" \
    --frame_stride ${AIST_FRAME_STRIDE} \
    --max_views ${MAX_VIEWS} \
    --max_motions ${MAX_SAMPLES}

# Step 1c: Merge all datasets
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Step 1c] Merging all datasets..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "${MHR_CONV_DIR}"

${PIXI_RUN} "${MV_DIR}/step1c_merge.py" \
    --inputs "${DATA_DIR}/pairs_3dpw_fitted.npz" \
             "${DATA_DIR}/pairs_EMDB_fitted.npz" \
             "${DATA_DIR}/mv_pairs_RICH.npz" \
             "${DATA_DIR}/mv_pairs_AIST_stage1.npz" \
    --output "${DATA_DIR}/pairs_all_merged.npz"
fi


# ============================================================
# Step 2: Multi-view network training
# ============================================================
if [ "${RUN_STEP2}" = "1" ]; then
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Step 2] Training multi-view network..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

${CONDA_PYTHON} -u "${MV_DIR}/step2_train.py" \
    --data_path "${DATA_DIR}/pairs_all_merged.npz" \
    --smpl_model "${SMPL_MODEL}" \
    --save_dir "${EXP_DIR}" \
    --from_scratch \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --num_views_select ${NUM_VIEWS_SELECT} \
    --sv_loss_weight ${SV_LOSS_WEIGHT}
fi


# ============================================================
# Step 3: Evaluation
# ============================================================
if [ "${RUN_STEP3}" = "1" ]; then
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Step 3] Evaluating..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "${MHR_CONV_DIR}"

${PIXI_RUN} "${MV_DIR}/step3_eval.py" \
    --input_dirs "${RICH_SAMPLES}" \
    --model_dir "${EXP_DIR}" \
    --smpl_model "${SMPL_MODEL_REL}" \
    --output_dir "${EXP_DIR}/eval_RICH" \
    --single_view_baseline
fi


# ============================================================
# Smooth: Train Pose Corrector (SmootherMLP on AMASS)
# ============================================================
if [ "${RUN_SMOOTH}" = "1" ]; then
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Smooth] Training Pose Corrector (SmootherMLP, window=${SMOOTHER_WINDOW})..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "${SMOOTH_DIR}"
${CONDA_PYTHON} -u "${SMOOTH_DIR}/train_smoother.py" \
    --data "${AMASS_DATA}" \
    --output_dir "${SMOOTHER_CKPT_DIR}" \
    --window_size ${SMOOTHER_WINDOW} \
    --hidden_dims ${SMOOTHER_HIDDEN} \
    --noise_std ${SMOOTHER_NOISE} \
    --epochs ${SMOOTHER_EPOCHS} \
    --batch_size 256 \
    --lr 1e-3 \
    --foot_weight ${SMOOTHER_FOOT_WEIGHT}
fi


# ============================================================
# Demo: Visualization (1V/2V/3V comparison)
# ============================================================
if [ "${RUN_DEMO}" = "1" ]; then
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Demo] Running visualization..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "${PROJECT_DIR}"

# AIST demo
${CONDA_PYTHON} "${MV_DIR}/step3_demo_AIST.py" \
    --motion ${AIST_MOTION} \
    --cams ${AIST_CAMS} \
    --model_path "${EXP_DIR}/best_model.pth" \
    --max_frames ${DEMO_MAX_FRAMES}

# RICH demo
${CONDA_PYTHON} "${MV_DIR}/step3_demo_RICH.py" \
    --scene ${RICH_SCENE} \
    --cams ${RICH_CAMS} \
    --model_path "${EXP_DIR}/best_model.pth" \
    --max_frames ${DEMO_MAX_FRAMES}
fi


echo ""
echo "========================================================"
echo "  Pipeline done!"
echo "  Model:     ${EXP_DIR}"
echo "  Eval:      ${EXP_DIR}/eval_RICH"
echo "  Videos:    ${SCRIPT_DIR}/output_visualization/"
echo "  Smoother:  ${SMOOTHER_CKPT_DIR}"
echo "========================================================"
