"""
Utility functions for SAM 3D Body demo notebook
"""

import os
import time
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from sam_3d_body import load_sam_3d_body_hf, load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def setup_sam_3d_body(
    hf_repo_id: str = "facebook/sam-3d-body-vith",
    detector_name: str = "vitdet",
    segmentor_name: str = "sam2",
    fov_name: str = "moge2",
    detector_path: str = "",
    segmentor_path: str = "",
    fov_path: str = "",
    device: str = "cuda",
    detector_model: str = "",
    local_checkpoint_path: str = "",  # Local checkpoint path (containing model.ckpt and model_config.yaml)
    local_mhr_path: str = "",  # Local MHR model path
):
    """
    Set up SAM 3D Body estimator with optional components.

    Args:
        hf_repo_id: HuggingFace repository ID for the model
        detector_name: Name of detector to use ("vitdet" or "yolo")
        segmentor_name: Name of segmentor to use (default: "sam2")
        fov_name: Name of FOV estimator to use (default: "moge2")
        detector_path: URL or path for human detector model
        segmentor_path: Path to human segmentor model (optional)
        fov_path: path for FOV estimator
        device: Device to use (default: auto-detect cuda/cpu)
        detector_model: For YOLO, specify model variant (e.g., "yolo11n.pt", "yolo11s.pt", "yolo11m.pt")
        local_checkpoint_path: Local checkpoint path (directory containing model.ckpt and model_config.yaml)
        local_mhr_path: Local MHR model path

    Returns:
        estimator: SAM3DBodyEstimator instance ready for inference
    """
    setup_total_start = time.time()

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load core model
    t0 = time.time()
    if local_checkpoint_path:
        # Use local checkpoint
        print(f"Loading SAM 3D Body model from LOCAL: {local_checkpoint_path}...")
        ckpt_file = os.path.join(local_checkpoint_path, "model.ckpt")
        mhr_file = local_mhr_path or os.path.join(local_checkpoint_path, "assets", "mhr_model.pt")
        model, model_cfg = load_sam_3d_body(checkpoint_path=ckpt_file, device=device, mhr_path=mhr_file)
    else:
        # Use HuggingFace
        print(f"Loading SAM 3D Body model from HuggingFace: {hf_repo_id}...")
        model, model_cfg = load_sam_3d_body_hf(hf_repo_id, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"    [setup_sam_3d_body] load_sam_3d_body: {time.time() - t0:.4f}s")

    # Initialize optional components
    human_detector, human_segmentor, fov_estimator = None, None, None

    if detector_name:
        print(f"Loading human detector from {detector_name}...")
        t0 = time.time()
        from tools.build_detector import HumanDetector

        detector_kwargs = {}
        # YOLO series detectors require the model parameter
        if detector_model and detector_name in ("yolo", "yolo11", "yolo_pose"):
            detector_kwargs["model"] = detector_model
        human_detector = HumanDetector(name=detector_name, device=device, **detector_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"    [setup_sam_3d_body] load_human_detector: {time.time() - t0:.4f}s")

    if segmentor_path:
        print(f"Loading human segmentor from {segmentor_path}...")
        t0 = time.time()
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=segmentor_name, device=device, path=segmentor_path
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"    [setup_sam_3d_body] load_human_segmentor: {time.time() - t0:.4f}s")

    if fov_name:
        print(f"Loading FOV estimator from {fov_name}...")
        t0 = time.time()
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=fov_name, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"    [setup_sam_3d_body] load_fov_estimator: {time.time() - t0:.4f}s")

    # Create estimator wrapper
    t0 = time.time()
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )
    print(f"    [setup_sam_3d_body] create_estimator: {time.time() - t0:.4f}s")

    print(f"    [setup_sam_3d_body] TOTAL: {time.time() - setup_total_start:.4f}s")
    print(f"Setup complete!")
    print(
        f"  Human detector: {'✓' if human_detector else '✗ (will use full image or manual bbox)'}"
    )
    print(
        f"  Human segmentor: {'✓' if human_segmentor else '✗ (mask inference disabled)'}"
    )
    print(f"  FOV estimator: {'✓' if fov_estimator else '✗ (will use default FOV)'}")

    # Print model precision summary
    print("\n" + "=" * 60)
    print("MODEL PRECISION SUMMARY")
    print("=" * 60)

    # SAM 3D Body backbone
    backbone_type = model_cfg.MODEL.BACKBONE.TYPE
    backbone_dtype = model.backbone_dtype
    print(f"  SAM3DBody-Backbone ({backbone_type}): {backbone_dtype}")

    # SAM 3D Body decoder
    decoder_dtype = next(model.decoder.parameters()).dtype
    decoder_hand_dtype = next(model.decoder_hand.parameters()).dtype
    print(f"  SAM3DBody-Decoder (Body): {decoder_dtype}")
    print(f"  SAM3DBody-Decoder (Hand): {decoder_hand_dtype}")

    # Human detector
    if human_detector is not None:
        if human_detector.name == "vitdet":
            detector_param = next(human_detector.detector.parameters())
            print(f"  ViTDet (Human Detector): {detector_param.dtype}")
        elif human_detector.name == "yolo" or human_detector.name.startswith("yolo11"):
            # Check if it's a TensorRT engine
            model_path = str(getattr(human_detector.detector, 'ckpt_path', ''))
            if model_path.endswith('.engine'):
                print(f"  YOLO11 (Human Detector): TensorRT FP16 engine")
            else:
                detector_param = next(human_detector.detector.model.parameters())
                print(f"  YOLO11 (Human Detector): {detector_param.dtype} (inference uses FP16)")

    # FOV estimator
    if fov_estimator is not None:
        fov_param = next(fov_estimator.fov_estimator.parameters())
        print(f"  MoGe2 (FOV Estimator): {fov_param.dtype}")

    print("=" * 60 + "\n")

    return estimator


def setup_visualizer():
    """Set up skeleton visualizer with MHR70 pose info"""
    visualizer = SkeletonVisualizer(line_width=2, radius=5)
    visualizer.set_pose_meta(mhr70_pose_info)
    return visualizer


def visualize_2d_results(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], visualizer: SkeletonVisualizer
) -> List[np.ndarray]:
    """Visualize 2D keypoints and bounding boxes"""
    results = []

    for pid, person_output in enumerate(outputs):
        img_vis = img_cv2.copy()

        # Draw keypoints
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d_vis = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_vis = visualizer.draw_skeleton(img_vis, keypoints_2d_vis)

        # Draw bounding box
        bbox = person_output["bbox"]
        img_vis = cv2.rectangle(
            img_vis,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),  # Green color
            2,
        )

        # Add person ID text
        cv2.putText(
            img_vis,
            f"Person {pid}",
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        results.append(img_vis)

    return results


def visualize_3d_mesh(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], faces: np.ndarray
) -> List[np.ndarray]:
    """Visualize 3D mesh overlaid on image and side view"""
    results = []
    viz_total_start = time.time()

    for pid, person_output in enumerate(outputs):
        person_start = time.time()

        # Create renderer for this person
        t0 = time.time()
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        print(f"      [visualize_3d_mesh] person_{pid}_create_renderer: {time.time() - t0:.4f}s")

        # 1. Original image
        img_orig = img_cv2.copy()

        # 2. Mesh overlay on original image
        t0 = time.time()
        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)
        print(f"      [visualize_3d_mesh] person_{pid}_render_overlay: {time.time() - t0:.4f}s")

        # 3. Mesh on white background (front view)
        t0 = time.time()
        white_img = np.ones_like(img_cv2) * 255
        img_mesh_white = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)
        print(f"      [visualize_3d_mesh] person_{pid}_render_front: {time.time() - t0:.4f}s")

        # 4. Side view
        t0 = time.time()
        img_mesh_side = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        ).astype(np.uint8)
        print(f"      [visualize_3d_mesh] person_{pid}_render_side: {time.time() - t0:.4f}s")

        # Combine all views
        combined = np.concatenate(
            [img_orig, img_mesh_overlay, img_mesh_white, img_mesh_side], axis=1
        )
        results.append(combined)
        print(f"      [visualize_3d_mesh] person_{pid}_total: {time.time() - person_start:.4f}s")

    print(f"      [visualize_3d_mesh] all_persons_total: {time.time() - viz_total_start:.4f}s")
    return results


def save_mesh_results(
    img_cv2: np.ndarray,
    outputs: List[Dict[str, Any]],
    faces: np.ndarray,
    save_dir: str,
    image_name: str,
) -> List[str]:
    """Save 3D mesh results to files and return PLY file paths"""
    import json

    save_total_start = time.time()
    os.makedirs(save_dir, exist_ok=True)
    ply_files = []

    # Save focal length
    if outputs:
        focal_length_data = {"focal_length": float(outputs[0]["focal_length"])}
        focal_length_path = os.path.join(save_dir, f"{image_name}_focal_length.json")
        with open(focal_length_path, "w") as f:
            json.dump(focal_length_data, f, indent=2)
        print(f"Saved focal length: {focal_length_path}")

    for pid, person_output in enumerate(outputs):
        person_start = time.time()

        # Skip persons with NaN vertices or empty vertices
        vertices = person_output["pred_vertices"]
        cam_t = person_output["pred_cam_t"]
        if vertices is None or len(vertices) == 0 or np.any(np.isnan(vertices)) or np.any(np.isnan(cam_t)):
            print(f"      [save_mesh_results] person_{pid} has invalid vertices, skipping")
            continue

        # Create renderer for this person
        t0 = time.time()
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        print(f"      [save_mesh_results] person_{pid}_create_renderer: {time.time() - t0:.4f}s")

        # Store individual mesh
        t0 = time.time()
        tmesh = renderer.vertices_to_trimesh(
            vertices, cam_t, LIGHT_BLUE
        )
        mesh_filename = f"{image_name}_mesh_{pid:03d}.ply"
        mesh_path = os.path.join(save_dir, mesh_filename)
        tmesh.export(mesh_path)
        ply_files.append(mesh_path)
        print(f"      [save_mesh_results] person_{pid}_export_mesh: {time.time() - t0:.4f}s")

        # Save individual overlay image
        t0 = time.time()
        img_mesh_overlay = (
            renderer(
                vertices,
                cam_t,
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        overlay_filename = f"{image_name}_overlay_{pid:03d}.png"
        cv2.imwrite(os.path.join(save_dir, overlay_filename), img_mesh_overlay)
        print(f"      [save_mesh_results] person_{pid}_save_overlay: {time.time() - t0:.4f}s")

        # Save bbox image
        t0 = time.time()
        img_bbox = img_cv2.copy()
        bbox = person_output["bbox"]
        img_bbox = cv2.rectangle(
            img_bbox,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            4,
        )
        bbox_filename = f"{image_name}_bbox_{pid:03d}.png"
        cv2.imwrite(os.path.join(save_dir, bbox_filename), img_bbox)
        print(f"      [save_mesh_results] person_{pid}_save_bbox: {time.time() - t0:.4f}s")

        print(f"      [save_mesh_results] person_{pid}_total: {time.time() - person_start:.4f}s")
        print(f"Saved mesh: {mesh_path}")
        print(f"Saved overlay: {os.path.join(save_dir, overlay_filename)}")
        print(f"Saved bbox: {os.path.join(save_dir, bbox_filename)}")

    print(f"      [save_mesh_results] all_persons_total: {time.time() - save_total_start:.4f}s")
    return ply_files


def display_results_grid(
    images: List[np.ndarray], titles: List[str], figsize_per_image: tuple = (6, 6)
):
    """Display multiple images in a grid"""
    n_images = len(images)
    if n_images == 0:
        print("No images to display")
        return

    # Calculate grid dimensions
    cols = min(3, n_images)  # Max 3 columns
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(figsize_per_image[0] * cols, figsize_per_image[1] * rows)
    )

    # Handle single image case
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convert BGR to RGB if needed
            if img.dtype == np.uint8 and np.mean(img) > 1:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
        else:
            img_rgb = img

        axes[i].imshow(img_rgb)
        axes[i].set_title(title)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def process_image_with_mask(estimator, image_path: str, mask_path: str):
    """
    Process image with external mask input.

    Note: The refactored code requires bboxes to be provided along with masks.
    This function automatically computes bboxes from the mask.
    """
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")

    # Ensure mask is binary (0 or 255)
    mask_binary = (mask > 127).astype(np.uint8) * 255

    print(f"Processing image with external mask: {mask_path}")
    print(f"Mask shape: {mask_binary.shape}, unique values: {np.unique(mask_binary)}")

    # Compute bounding box from mask (required by refactored code)
    # Find all non-zero pixels in the mask
    coords = cv2.findNonZero(mask_binary)
    if coords is None:
        print("Warning: Mask is empty, no objects detected")
        return []

    # Get bounding box from mask contours
    x, y, w, h = cv2.boundingRect(coords)
    bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)

    print(f"Computed bbox from mask: {bbox[0]}")

    # Process with external mask and computed bbox
    # Note: The mask needs to match the number of bboxes (1 bbox -> 1 mask)
    outputs = estimator.process_one_image(image_path, bboxes=bbox, masks=mask_binary)

    return outputs
