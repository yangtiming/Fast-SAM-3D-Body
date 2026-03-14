# Copyright (c) Meta Platforms, Inc. and affiliates.
import time
from typing import Optional, Union

import cv2

import numpy as np
import torch

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)

from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.utils import recursive_to
from torchvision.transforms import ToTensor


class SAM3DBodyEstimator:
    def __init__(
        self,
        sam_3d_body_model,
        model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    ):
        self.device = sam_3d_body_model.device
        self.model, self.cfg = sam_3d_body_model, model_cfg
        self.detector = human_detector
        self.sam = human_segmentor
        self.fov_estimator = fov_estimator
        self.thresh_wrist_angle = 1.4

        # For mesh visualization
        self.faces = self.model.head_pose.faces.cpu().numpy()

        if self.detector is None:
            print("No human detector is used...")
        if self.sam is None:
            print("Mask-condition inference is not supported...")
        if self.fov_estimator is None:
            print("No FOV estimator... Using the default FOV!")

        print(f"[SAM3DBodyEstimator] IMAGE_SIZE: {self.cfg.MODEL.IMAGE_SIZE}")

        # CUDA Graph warmup (capture before inference to avoid runtime overhead)
        self._warmup_cuda_graph()

        # Multi-person mode compile warmup (let torch.compile generate code for different batch sizes)
        self._warmup_multi_person_compile()

        self.transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        self.transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=0.9),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    def _warmup_cuda_graph(self):
        """Warmup and capture CUDA Graph (called before inference)."""
        import os
        # CUDA Graph is disabled by default; requires MHR_USE_CUDA_GRAPH=1 to enable
        if os.environ.get('MHR_USE_CUDA_GRAPH', '0') != '1':
            return

        cuda_graph_failed = False

        # Warmup body head
        try:
            if hasattr(self.model, 'head_pose') and hasattr(self.model.head_pose, 'warmup_cuda_graph'):
                print("[SAM3DBodyEstimator] Warming up CUDA Graph for body head...")
                self.model.head_pose.warmup_cuda_graph()
                # Check if successful
                if not getattr(self.model.head_pose, '_cuda_graph_captured', False):
                    cuda_graph_failed = True
        except Exception as e:
            print(f"[SAM3DBodyEstimator] Body head CUDA Graph warmup failed: {e}")
            cuda_graph_failed = True

        # If body head failed, clean up state and skip hand head
        if cuda_graph_failed:
            try:
                torch.cuda.synchronize()
            except:
                pass
            # Disable CUDA Graph for hand head
            if hasattr(self.model, 'head_pose_hand'):
                self.model.head_pose_hand.use_cuda_graph = False
            print("[SAM3DBodyEstimator] Skipping hand head CUDA Graph due to previous failure")
            return

        # Warmup hand head
        try:
            if hasattr(self.model, 'head_pose_hand') and hasattr(self.model.head_pose_hand, 'warmup_cuda_graph'):
                print("[SAM3DBodyEstimator] Warming up CUDA Graph for hand head...")
                self.model.head_pose_hand.warmup_cuda_graph()
        except Exception as e:
            print(f"[SAM3DBodyEstimator] Hand head CUDA Graph warmup failed: {e}")

        try:
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[SAM3DBodyEstimator] CUDA synchronize failed: {e}")

    def _warmup_multi_person_compile(self):
        """
        Compile warmup for multi-person mode, letting torch.compile generate code for batch_size > 1.
        This resolves the issue where torch.compile only sees batch_size=1 during warmup,
        causing NaN in multi-person mode.
        """
        import os
        if os.environ.get('USE_COMPILE', '0') != '1':
            return

        # Check if multi-person mode warmup is needed
        # Default includes 1,2,4 to cover single-person and multi-person scenarios
        warmup_batch_sizes = os.environ.get('COMPILE_WARMUP_BATCH_SIZES', '1,2,4')
        if not warmup_batch_sizes:
            return

        batch_sizes = [int(x.strip()) for x in warmup_batch_sizes.split(',') if x.strip()]
        if not batch_sizes:
            return

        print(f"[SAM3DBodyEstimator] Warming up torch.compile for multi-person mode (batch_sizes={batch_sizes})...")

        img_size = self.cfg.MODEL.IMAGE_SIZE
        device = self.device

        for batch_size in batch_sizes:
            try:
                # Create dummy batch for warmup
                # Combined batch: body (N) + left_hand (N) + right_hand (N) = 3N
                combined_size = 3 * batch_size

                dummy_batch = {
                    'img': torch.randn(1, combined_size, 3, img_size[0], img_size[1], device=device),
                    'bbox_center': torch.randn(1, combined_size, 2, device=device),
                    'bbox_scale': torch.randn(1, combined_size, 2, device=device).abs() + 100,
                    'ori_img_size': torch.tensor([[1920, 1080]], device=device).expand(1, combined_size, -1).clone(),
                    'img_size': torch.tensor([[img_size[1], img_size[0]]], device=device).expand(1, combined_size, -1).clone().float(),
                    'cam_int': torch.eye(3, device=device).unsqueeze(0),
                    'affine_trans': torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, combined_size, -1, -1).clone(),
                    # mask needs channel dimension: [B, N, 1, H, W] -> after flatten [B*N, 1, H, W]
                    'mask': torch.ones(1, combined_size, 1, img_size[0], img_size[1], device=device),
                    'mask_score': torch.ones(1, combined_size, device=device),
                    'person_valid': torch.ones(1, combined_size, device=device),
                }
                # Set focal length
                dummy_batch['cam_int'][0, 0, 0] = 1000.0
                dummy_batch['cam_int'][0, 1, 1] = 1000.0
                dummy_batch['cam_int'][0, 0, 2] = 960.0
                dummy_batch['cam_int'][0, 1, 2] = 540.0

                # Initialize batch
                self.model._initialize_batch(dummy_batch)

                # Set batch indices
                body_batch_idx = list(range(batch_size))
                hand_batch_idx = list(range(batch_size, 3 * batch_size))

                # Run forward_step_merged for warmup
                with torch.no_grad():
                    _ = self.model.forward_step_merged(
                        dummy_batch,
                        body_batch_idx=body_batch_idx,
                        hand_batch_idx=hand_batch_idx
                    )

                torch.cuda.synchronize()
                print(f"[SAM3DBodyEstimator] Warmup for batch_size={batch_size} completed")

            except Exception as e:
                print(f"[SAM3DBodyEstimator] Warmup for batch_size={batch_size} failed: {e}")
                import traceback
                traceback.print_exc()

        print("[SAM3DBodyEstimator] Multi-person compile warmup completed")

    @torch.no_grad()
    def process_one_image(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        cam_int: Optional[np.ndarray] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        inference_type: str = "full",
        hand_box_source: str = "body_decoder",  # "body_decoder" or "yolo_pose"
    ):
        """
        Perform model prediction in top-down format: assuming input is a full image.

        Args:
            img: Input image (path or numpy array)
            bboxes: Optional pre-computed bounding boxes
            masks: Optional pre-computed masks (numpy array). If provided, SAM2 will be skipped.
            det_cat_id: Detection category ID
            bbox_thr: Bounding box threshold
            nms_thr: NMS threshold
            inference_type:
                - full: full-body inference with both body and hand decoders
                - body: inference with body decoder only (still full-body output)
                - hand: inference with hand decoder only (only hand output)
            hand_box_source:
                - body_decoder: use hand boxes from body decoder output (default)
                - yolo_pose: use hand boxes computed from YOLO-Pose wrist keypoints
                  (requires detector to be yolo_pose type)
        """
        process_total_start = time.time()
        print("      [process_one_image] Starting...")

        # clear all cached results
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []
        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            # If state is corrupted after CUDA Graph capture failure, skip empty_cache
            print(f"        [process_one_image] Warning: empty_cache failed: {e}")

        t0 = time.time()
        if type(img) == str:
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            print("####### Please make sure the input image is in RGB format")
            image_format = "rgb"
        height, width = img.shape[:2]
        print(f"        [process_one_image] load_image: {time.time() - t0:.4f}s")

        t0 = time.time()
        yolo_pose_keypoints = None  # Will be set if using yolo_pose detector
        yolo_pose_body_boxes = None
        if bboxes is not None:
            boxes = bboxes.reshape(-1, 4)
            self.is_crop = True
        elif self.detector is not None:
            if image_format == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                image_format = "bgr"
            print("Running object detector...")
            torch.cuda.synchronize()
            detection_result = self.detector.run_human_detection(
                img,
                det_cat_id=det_cat_id,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                default_to_full_image=False,
            )
            torch.cuda.synchronize()

            # Handle yolo_pose detector which returns dict with boxes and keypoints
            if isinstance(detection_result, dict):
                boxes = detection_result["boxes"]
                yolo_pose_keypoints = detection_result.get("keypoints", None)
                yolo_pose_body_boxes = boxes.copy()  # Save body boxes for hand box computation
                print(f"Found boxes: {boxes}")
                if yolo_pose_keypoints is not None:
                    print(f"Found keypoints shape: {yolo_pose_keypoints.shape}")
            else:
                boxes = detection_result
                print("Found boxes:", boxes)

            self.is_crop = True
        else:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
            self.is_crop = False
        print(f"        [process_one_image] human_detection: {time.time() - t0:.4f}s")

        # If there are no detected humans, don't run prediction
        if len(boxes) == 0:
            return []

        # The following models expect RGB images instead of BGR
        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Handle masks - either provided externally or generated via SAM2
        t0 = time.time()
        masks_score = None
        if masks is not None:
            # Use provided masks - ensure they match the number of detected boxes
            print(f"Using provided masks: {masks.shape}")
            assert (
                bboxes is not None
            ), "Mask-conditioned inference requires bboxes input!"
            masks = masks.reshape(-1, height, width, 1).astype(np.uint8)
            masks_score = np.ones(
                len(masks), dtype=np.float32
            )  # Set high confidence for provided masks
            use_mask = True
        elif use_mask and self.sam is not None:
            print("Running SAM to get mask from bbox...")
            # Generate masks using SAM2
            torch.cuda.synchronize()
            masks, masks_score = self.sam.run_sam(img, boxes)
            torch.cuda.synchronize()
        else:
            masks, masks_score = None, None
        print(f"        [process_one_image] mask_processing: {time.time() - t0:.4f}s")

        #################### Construct batch data samples ####################
        t0 = time.time()
        batch = prepare_batch(img, self.transform, boxes, masks, masks_score)
        print(f"        [process_one_image] prepare_batch: {time.time() - t0:.4f}s")

        #################### Run model inference on an image ####################
        t0 = time.time()
        batch = recursive_to(batch, "cuda")
        self.model._initialize_batch(batch)
        print(f"        [process_one_image] initialize_batch: {time.time() - t0:.4f}s")

        # Handle camera intrinsics
        # - either provided externally or generated via default FOV estimator
        t0 = time.time()
        if cam_int is not None:
            print("Using provided camera intrinsics...")
            cam_int = cam_int.to(batch["img"])
            batch["cam_int"] = cam_int.clone()
        elif self.fov_estimator is not None:
            print("Running FOV estimator ...")
            input_image = batch["img_ori"][0].data
            torch.cuda.synchronize()
            cam_int = self.fov_estimator.get_cam_intrinsics(input_image).to(
                batch["img"]
            )
            torch.cuda.synchronize()
            batch["cam_int"] = cam_int.clone()
        else:
            cam_int = batch["cam_int"].clone()
        print(f"        [process_one_image] fov_estimation: {time.time() - t0:.4f}s")

        t0 = time.time()
        torch.cuda.synchronize()
        outputs = self.model.run_inference(
            img,
            batch,
            inference_type=inference_type,
            transform_hand=self.transform_hand,
            thresh_wrist_angle=self.thresh_wrist_angle,
            hand_box_source=hand_box_source,
            yolo_pose_keypoints=yolo_pose_keypoints,
            yolo_pose_body_boxes=yolo_pose_body_boxes,
        )
        torch.cuda.synchronize()
        print(f"        [process_one_image] model_run_inference: {time.time() - t0:.4f}s")
        if inference_type == "full":
            pose_output, batch_lhand, batch_rhand, _, _ = outputs
        else:
            pose_output = outputs

        t0 = time.time()
        out = pose_output["mhr"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")
        all_out = []
        for idx in range(batch["img"].shape[1]):
            all_out.append(
                {
                    "bbox": batch["bbox"][0, idx].cpu().numpy(),
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_vertices": out["pred_vertices"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                    "pred_pose_raw": out["pred_pose_raw"][idx],
                    "global_rot": out["global_rot"][idx],
                    "body_pose_params": out["body_pose"][idx],
                    "hand_pose_params": out["hand"][idx],
                    "scale_params": out["scale"][idx],
                    "shape_params": out["shape"][idx],
                    "expr_params": out["face"][idx],
                    "mask": masks[idx] if masks is not None else None,
                    "pred_joint_coords": out["pred_joint_coords"][idx],
                    "pred_global_rots": out["joint_global_rots"][idx],
                }
            )

            if inference_type == "full":
                all_out[-1]["lhand_bbox"] = np.array(
                    [
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_lhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_lhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )
                all_out[-1]["rhand_bbox"] = np.array(
                    [
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            - batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][0]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][0] / 2
                        ).item(),
                        (
                            batch_rhand["bbox_center"].flatten(0, 1)[idx][1]
                            + batch_rhand["bbox_scale"].flatten(0, 1)[idx][1] / 2
                        ).item(),
                    ]
                )

        print(f"        [process_one_image] postprocess_output: {time.time() - t0:.4f}s")
        print(f"      [process_one_image] TOTAL: {time.time() - process_total_start:.4f}s")
        return all_out
