#!/bin/bash
# SAM 3D Body run script with optimized environment variables

# ============================================================
# Core Performance
# ============================================================
export GPU_HAND_PREP=1              # GPU hand preprocessing (faster)
export LAYER_DTYPE=fp32             # Layer dtype: fp32
# Multi-person scenarios require fp32; sam3dbody defaults to fp32 as well
export SKIP_KEYPOINT_PROMPT=1       # Skip keypoint prompt encoding
export IMG_SIZE=512                 # Input Backbone image size 448 (0=original 512)
# Backbone+decoder IMG_SIZE defaults to 512; important -- too small leads to inaccurate predictions!

# ============================================================
# torch.compile Optimization
# ============================================================
export USE_COMPILE=1                # Enable torch.compile
export USE_COMPILE_BACKBONE=1       # Compile backbone (DINOv3)
export DECODER_COMPILE=1            # Compile decoder
# export INTERM_COMPILE=1           # Compile intermediate layers (default=1)
export COMPILE_MODE=reduce-overhead  # Compile mode: default, reduce-overhead, max-autotune
export COMPILE_WARMUP_BATCH_SIZES=1  # Warmup batch sizes

# ============================================================
# CUDA Graph
# ============================================================
export MHR_USE_CUDA_GRAPH=0         # MHR CUDA Graph (0=off, 1=on)

# ============================================================
# Intermediate Layer Prediction
# ============================================================
export KEYPOINT_PROMPT_INTERM_INTERVAL=999  # Keypoint prompt interval (999=disable)
# export KEYPOINT_PROMPT_INTERM_LAYERS=0,1,2,3  # Specific layers for keypoint prompt

export BODY_INTERM_PRED_LAYERS=0,1,2        # Body decoder intermediate layers 
# Fewer layers = faster decoder; reducing layers significantly improves speed. Optimal: 0,1,2
export HAND_INTERM_PRED_LAYERS=0,1          # Hand decoder intermediate layers
# Fewer layers = faster decoder; reducing layers significantly improves speed. Optimal: 0,1

# export INTERM_PRED_LAYERS=0,1,2,3         # Generic intermediate layers (overridden by BODY/HAND)
# export INTERM_PRED_INTERVAL=1             # Generic interval (overridden by BODY/HAND)

# ============================================================
# MHR Head
# ============================================================
export MHR_NO_CORRECTIVES=1         # Disable correctives (faster)
# export MOMENTUM_ENABLED=1         # Enable momentum (default=1)

# ============================================================
# FOV Estimator (MoGe2)
# ============================================================
export FOV_TRT=1                    # Use TensorRT encoder
export FOV_FAST=1                   # Fast mode (skip normal_head)
export FOV_MODEL=s                  # Model size: s(35M), b(104M), l(331M)
export FOV_LEVEL=0                  # Resolution level: 0-9 (0=1200 tokens, 9=3600 tokens)
# export FOV_SIZE=512               # Input size (auto=512 when FOV_TRT=1)
# TRT Path: checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine

# ============================================================
# Backbone TensorRT (DINOv3) - requires engine file
# ============================================================
# export USE_TRT_BACKBONE=1
# export TRT_BACKBONE_PATH=./checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine

# ============================================================
# Decoder
# ============================================================
# export PARALLEL_DECODERS=1        # Parallel body/hand decoders (default=1)

# ============================================================
# Debug (set to 1 to enable)
# ============================================================
export DEBUG_NAN=0
export DEBUG_HAND_PREP=0
export DEBUG_BACKBONE_INPUT=0
export INTERM_TIMING=0            # Intermediate layer timing

# ============================================================
# Common Arguments
# ============================================================
IMAGE_PATH=./notebook/images/dancing.jpg
DETECTOR=yolo_pose
DETECTOR_MODEL=./checkpoints/yolo/yolo11m-pose.engine
HAND_BOX_SOURCE=yolo_pose

# ============================================================
# Run
# ============================================================

# Generate visualization results
python demo_human.py \
    --image_path $IMAGE_PATH \
    --detector $DETECTOR \
    --detector_model $DETECTOR_MODEL \
    --hand_box_source $HAND_BOX_SOURCE

# Speed profiling / inference benchmark
python profile_nsight.py \
    --image_path $IMAGE_PATH \
    --detector $DETECTOR \
    --detector_model $DETECTOR_MODEL \
    --hand_box_source $HAND_BOX_SOURCE
