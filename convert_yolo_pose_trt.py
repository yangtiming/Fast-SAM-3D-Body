#!/usr/bin/env python3
"""
Convert YOLO-Pose model to TensorRT format.

Usage:
    python convert_yolo_pose_trt.py --model yolo11m-pose.pt --imgsz 640
    python convert_yolo_pose_trt.py --model yolo11n-pose.pt --imgsz 640 --half
"""

import argparse
import os


CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "yolo")


def convert_to_tensorrt(model_name: str, imgsz: int = 640, half: bool = True):
    """
    Convert YOLO-Pose model to TensorRT format.

    Args:
        model_name: Model name or path (e.g., "yolo11m-pose.pt")
        imgsz: Input image size
        half: Whether to use FP16
    """
    from ultralytics import YOLO

    # If model_name is just a filename, look in checkpoints/yolo/
    if not os.path.isabs(model_name) and not os.path.exists(model_name):
        ckpt_path = os.path.join(CHECKPOINT_DIR, model_name)
        if os.path.exists(ckpt_path):
            model_name = ckpt_path

    print(f"Loading model: {model_name}")
    model = YOLO(model_name)

    # Export to TensorRT
    print(f"Exporting to TensorRT (imgsz={imgsz}, half={half})...")
    output_path = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        device=0,  # GPU 0
        simplify=True,
        workspace=4,  # GB
    )

    # Move engine to checkpoints/yolo/ if not already there
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    import shutil
    engine_name = os.path.basename(output_path)
    target_path = os.path.join(CHECKPOINT_DIR, engine_name)
    if os.path.abspath(output_path) != os.path.abspath(target_path):
        shutil.move(output_path, target_path)
        output_path = target_path
        print(f"Moved engine to: {output_path}")

    # Also move intermediate .onnx file to checkpoints/yolo/ if it was left beside the .pt
    source_dir = os.path.dirname(os.path.abspath(model_name))
    onnx_name = os.path.splitext(os.path.basename(model_name))[0] + ".onnx"
    onnx_source = os.path.join(source_dir, onnx_name)
    onnx_target = os.path.join(CHECKPOINT_DIR, onnx_name)
    if os.path.exists(onnx_source) and os.path.abspath(onnx_source) != os.path.abspath(onnx_target):
        shutil.move(onnx_source, onnx_target)
        print(f"Moved ONNX to: {onnx_target}")

    print(f"TensorRT engine saved to: {output_path}")
    return output_path


def test_engine(engine_path: str, test_image: str = None):
    """Test TensorRT engine."""
    from ultralytics import YOLO
    import numpy as np

    print(f"\nTesting engine: {engine_path}")
    model = YOLO(engine_path)

    # Create test image
    if test_image and os.path.exists(test_image):
        import cv2
        img = cv2.imread(test_image)
    else:
        print("Using random test image...")
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Run inference
    results = model(img, verbose=False)

    print(f"Test passed! Detected {len(results[0].boxes)} objects")
    if results[0].keypoints is not None:
        print(f"Keypoints shape: {results[0].keypoints.data.shape}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO-Pose to TensorRT")
    parser.add_argument(
        "--model",
        type=str,
        default="./checkpoints/yolo/yolo11m-pose.pt",
        help="YOLO-Pose model name (e.g., yolo11n-pose.pt, yolo11m-pose.pt)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="Use FP16 (default: True)"
    )
    parser.add_argument(
        "--no-half",
        action="store_false",
        dest="half",
        help="Use FP32 instead of FP16"
    )
    parser.add_argument(
        "--test-image",
        type=str,
        default="./notebook/images/dancing.jpg",
        help="Test image path (optional)"
    )

    args = parser.parse_args()

    # Convert
    engine_path = convert_to_tensorrt(args.model, args.imgsz, args.half)

    # Test
    test_engine(engine_path, args.test_image)

    print("\n" + "=" * 60)
    print("Done! You can now use the TensorRT engine:")
    print(f"  --detector yolo_pose --detector_model {engine_path}")
    print("=" * 60)
