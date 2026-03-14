#!/bin/bash
# Convert all models to TensorRT format.
#
# All generated engines are stored under ./checkpoints/:
#   checkpoints/yolo/yolo11m-pose.engine
#   checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine
#   checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine (optional)
#
# Usage:
#   bash build_tensorrt.sh

set -e
cd "$(dirname "$0")"

echo "============================================================"
echo " SAM 3D Body - TensorRT Engine Build"
echo "============================================================"
echo "Working directory: $(pwd)"
echo ""

# ============================================================
# 1. YOLO-Pose TensorRT
# ============================================================
echo "[Step 1] Converting YOLO-Pose to TensorRT..."
python convert_yolo_pose_trt.py --model ./checkpoints/yolo/yolo11m-pose.pt --imgsz 640 --half
echo ""

# ============================================================
# 2. MoGe2 FOV Estimator TensorRT (optional)
# ============================================================
echo "[Step 2] Converting MoGe2 DINOv2 Encoder to TensorRT..."
python convert_moge_encoder_trt.py --all
echo ""

# ============================================================
# 3. DINOv3 Backbone TensorRT (optional)
# ============================================================
echo "[Step 3] Converting DINOv3 Backbone to TensorRT..."
python convert_backbone_tensorrt.py --all
echo ""
