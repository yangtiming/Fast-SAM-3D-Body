# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import torch

# FOV_MODEL: MoGe2 model size (s=35M, b=104M, l=331M)
#   - s: Ruicheng/moge-2-vits-normal (fastest)
#   - b: Ruicheng/moge-2-vitb-normal
#   - l: Ruicheng/moge-2-vitl-normal (default, most accurate)
# FOV_LEVEL: MoGe2 resolution_level (0-9), controls number of ViT tokens
#   - level=0 -> 1200 tokens, level=9 -> 3600 tokens
# FOV_SIZE: Optional, pre-resize input image size (reduces data transfer, 0=no resize)
# FOV_FAST: Enable fast mode, skip normal_head (can only skip normal, since intrinsics depends on points/mask)
# FOV_TRT: Use TensorRT to accelerate encoder (requires running convert_moge_encoder_trt.py first)
FOV_MODEL = os.environ.get("FOV_MODEL", "l")  # s, b, l
FOV_LEVEL = os.environ.get("FOV_LEVEL", "")
FOV_SIZE = int(os.environ.get("FOV_SIZE", "0"))  # 0 = no resize, keep original image
FOV_FAST = os.environ.get("FOV_FAST", "0") == "1"  # 1 = fast mode
FOV_TRT = os.environ.get("FOV_TRT", "0") == "1"  # 1 = use TensorRT encoder

MOGE_MODELS = {
    "s": "Ruicheng/moge-2-vits-normal",  # 35M params
    "b": "Ruicheng/moge-2-vitb-normal",  # 104M params
    "l": "Ruicheng/moge-2-vitl-normal",  # 331M params (default)
}

# TensorRT encoder config (for model 's' with level=0)
# Auto-find TRT engine path (relative to project root)
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TOOLS_DIR)
TRT_ENCODER_PATH = os.path.join(_PROJECT_ROOT, "checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine")
TRT_TOKEN_ROWS = 35
TRT_TOKEN_COLS = 35
TRT_EMBED_DIM = 384


class TRTEncoderWrapper:
    """TensorRT wrapper for MoGe2 DINOv2 encoder."""

    def __init__(self, engine_path, device="cuda"):
        import tensorrt as trt

        self.device = device

        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # I/O names
        self.input_name = "image"
        self.output_names = ["features", "cls_token"]

        # Pre-allocate output buffers
        self.features_buf = torch.empty(
            (4, TRT_EMBED_DIM, TRT_TOKEN_ROWS, TRT_TOKEN_COLS),
            device=device, dtype=torch.float16
        )
        self.cls_buf = torch.empty(
            (4, TRT_EMBED_DIM), device=device, dtype=torch.float16
        )

        print(f"  [FOV_TRT] Loaded TensorRT encoder: {engine_path}")

    def __call__(self, x, token_rows, token_cols, return_class_token=True):
        """Run TensorRT encoder inference."""
        batch_size = x.shape[0]

        # Set input shape
        self.context.set_input_shape(self.input_name, tuple(x.shape))

        # Slice output buffers for current batch
        features = self.features_buf[:batch_size]
        cls_token = self.cls_buf[:batch_size]

        # Ensure input is FP16 contiguous
        x_fp16 = x.half().contiguous() if x.dtype != torch.float16 else x.contiguous()

        # Set addresses and execute
        self.context.set_tensor_address(self.input_name, x_fp16.data_ptr())
        self.context.set_tensor_address(self.output_names[0], features.data_ptr())
        self.context.set_tensor_address(self.output_names[1], cls_token.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        if return_class_token:
            return features, cls_token
        return features


class FOVEstimator:
    def __init__(self, name="moge2", device="cuda", **kwargs):
        self.device = device
        self.fixed_size = FOV_SIZE
        self.resolution_level = int(FOV_LEVEL) if FOV_LEVEL else 9  # default 9
        self.model_size = FOV_MODEL
        self.fast_mode = FOV_FAST
        self.use_trt = FOV_TRT

        # TRT encoder requires fixed 512x512 input
        if self.use_trt and self.fixed_size == 0:
            print("  [FOV_TRT] Auto-setting FOV_SIZE=512 (TRT requires fixed input size)")
            self.fixed_size = 512

        if name == "moge2":
            model_path = MOGE_MODELS.get(self.model_size, MOGE_MODELS["l"])
            print("########### Using fov estimator: MoGe2...")
            print(f"########### FOV model: {model_path} ({self.model_size})")
            print(f"########### FOV input_size: {self.fixed_size}, resolution_level: {self.resolution_level}, fast_mode: {self.fast_mode}, use_trt: {self.use_trt}")
            self.fov_estimator = load_moge(device, path=model_path, use_trt=self.use_trt, **kwargs)

            # Use fast mode if enabled (skips unnecessary heads)
            if self.fast_mode:
                self.fov_estimator_func = run_moge_fast
            else:
                self.fov_estimator_func = run_moge

            self.fov_estimator.eval()
        else:
            raise NotImplementedError

    def get_cam_intrinsics(self, img, **kwargs):
        return self.fov_estimator_func(self.fov_estimator, img, self.device,
                                        fixed_size=self.fixed_size,
                                        resolution_level=self.resolution_level, **kwargs)


def load_moge(device, path="", half=True, use_trt=False):
    from moge.model.v2 import MoGeModel
    import os

    if path == "":
        path = "Ruicheng/moge-2-vits-normal"
    moge_model = MoGeModel.from_pretrained(path).to(device)

    # Convert to FP16 for faster inference
    if half and device == "cuda":
        moge_model.half()
        print("  MoGe2 converted to FP16")

    # Optionally use TensorRT for encoder
    if use_trt and os.path.exists(TRT_ENCODER_PATH):
        try:
            trt_encoder = TRTEncoderWrapper(TRT_ENCODER_PATH, device)
            # Save original and replace (bypass nn.Module attribute check)
            moge_model._orig_encoder = moge_model.encoder
            object.__setattr__(moge_model, 'encoder', trt_encoder)
            print("  [FOV_TRT] Replaced encoder with TensorRT")
        except Exception as e:
            print(f"  [FOV_TRT] Failed to load TensorRT encoder: {e}")
            print("  [FOV_TRT] Falling back to PyTorch encoder")
    elif use_trt:
        print(f"  [FOV_TRT] TensorRT engine not found: {TRT_ENCODER_PATH}")
        print("  [FOV_TRT] Run: python convert_moge_encoder_trt.py --all")

    return moge_model


def run_moge(model, input_image, device, fixed_size=0, resolution_level=9):
    """
    Run MoGe2 inference.

    Args:
        fixed_size: If > 0, resize input image to this size before inference (reduces data transfer)
        resolution_level: 0-9, controls num_tokens. Lower = faster.
                          level=0 -> 1200 tokens, level=9 -> 3600 tokens
    """
    import torch.nn.functional as F

    # We expect the image to be RGB already
    H, W, _ = input_image.shape

    # Match input dtype to model dtype (FP16 or FP32)
    model_dtype = next(model.parameters()).dtype

    # Optimized CPU to GPU transfer:
    # 1. Use from_numpy (zero-copy view)
    # 2. Transfer to GPU as uint8 (smaller)
    # 3. Convert to float and normalize on GPU (faster)
    input_tensor = torch.from_numpy(input_image).to(device=device, non_blocking=True)
    input_tensor = input_tensor.to(dtype=model_dtype).div_(255.0).permute(2, 0, 1)

    # Optional: Resize to fixed size (reduces data transfer for large images)
    if fixed_size > 0:
        input_tensor = F.interpolate(
            input_tensor.unsqueeze(0),
            size=(fixed_size, fixed_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        size_str = f"{H}x{W} -> {fixed_size}x{fixed_size}"
    else:
        size_str = f"{H}x{W}"

    # Print model dtype info
    print(f"          [DEBUG] Model: MoGe2 (FOV Estimator), input_dtype: {input_tensor.dtype}, param_dtype: {model_dtype}, input_size: {size_str}, resolution_level: {resolution_level}")

    # Infer w/ MoGe2
    moge_data = model.infer(input_tensor, resolution_level=resolution_level)

    # get intrinsics
    intrinsics = denormalize_f(moge_data["intrinsics"].cpu().numpy(), H, W)
    v_focal = intrinsics[1, 1]

    # override hfov with v_focal
    intrinsics[0, 0] = v_focal
    # add batch dim
    cam_intrinsics = intrinsics[None]

    return cam_intrinsics


def run_moge_fast(model, input_image, device, fixed_size=0, resolution_level=9):
    """
    Fast MoGe2 inference - patches normal_head to skip computation.
    Only skips: normal_head (saves ~2ms)

    Note: intrinsics requires points_head and mask_head to compute focal/shift.
    """
    import torch.nn.functional as F

    H, W, _ = input_image.shape
    model_dtype = next(model.parameters()).dtype

    # Prepare input tensor
    input_tensor = torch.from_numpy(input_image).to(device=device, non_blocking=True)
    input_tensor = input_tensor.to(dtype=model_dtype).div_(255.0).permute(2, 0, 1)

    if fixed_size > 0:
        input_tensor = F.interpolate(
            input_tensor.unsqueeze(0),
            size=(fixed_size, fixed_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    # Patch normal_head to skip computation (only on first call)
    # We can't skip points_head or mask_head because intrinsics depends on them
    if not hasattr(model, '_fast_mode_patched'):
        # Save original head
        model._orig_normal_head = model.normal_head

        # Create dummy head that returns [None] (MoGe2 forward uses [-1] indexing)
        class SkipHead(torch.nn.Module):
            """A head that returns [None] - actual computation is skipped."""
            def forward(self, *args, **kwargs):
                return [None]  # forward() uses [-1] to get last element

        # Only skip normal_head (points/mask are needed for intrinsics)
        model.normal_head = SkipHead()
        model._fast_mode_patched = True
        print("  [FOV_FAST] Patched MoGe2 (skipping normal_head only)")

    # Run normal infer (but normal_head is now a no-op)
    moge_data = model.infer(input_tensor, resolution_level=resolution_level)

    # Get intrinsics
    intrinsics = denormalize_f(moge_data["intrinsics"].cpu().numpy(), H, W)
    v_focal = intrinsics[1, 1]
    intrinsics[0, 0] = v_focal
    cam_intrinsics = intrinsics[None]

    return cam_intrinsics


def denormalize_f(norm_K, height, width):
    # Extract cx and cy from the normalized K matrix
    cx_norm = norm_K[0][2]  # c_x is at K[0][2]
    cy_norm = norm_K[1][2]  # c_y is at K[1][2]

    fx_norm = norm_K[0][0]  # Normalized fx
    fy_norm = norm_K[1][1]  # Normalized fy
    # s_norm = norm_K[0][1]   # Skew (usually 0)

    # Scale to absolute values
    fx_abs = fx_norm * width
    fy_abs = fy_norm * height
    cx_abs = cx_norm * width
    cy_abs = cy_norm * height
    # s_abs = s_norm * width
    s_abs = 0

    # Construct absolute K matrix
    abs_K = torch.tensor(
        [[fx_abs, s_abs, cx_abs], [0.0, fy_abs, cy_abs], [0.0, 0.0, 1.0]]
    )
    return abs_K
