# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch

from .models.meta_arch import SAM3DBody
from .utils.config import get_config
from .utils.checkpoint import load_state_dict


def load_sam_3d_body(checkpoint_path: str = "", device: str = "cuda", mhr_path: str = ""):
    print("Loading SAM 3D Body model...")
    
    # Check the current directory, and if not present check the parent dir.
    model_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if not os.path.exists(model_cfg):
        # Looks at parent dir
        model_cfg = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
        )

    model_cfg = get_config(model_cfg)

    # Disable face for inference
    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path

    # Support overriding IMAGE_SIZE config via IMG_SIZE environment variable
    img_size_env = os.environ.get("IMG_SIZE", "0")
    if img_size_env and int(img_size_env) > 0:
        size = int(img_size_env)
        model_cfg.MODEL.IMAGE_SIZE = [size, size]
        print(f"[build_models] IMAGE_SIZE overridden by IMG_SIZE env: {size}x{size}")

    model_cfg.freeze()

    # Initialze the model
    model = SAM3DBody(model_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    load_state_dict(model, state_dict, strict=False)

    model = model.to(device)
    model.eval()

    # Parse LAYER_DTYPE environment variable (only supports fp16/bf16, used for autocast)
    layer_dtype_str = os.environ.get("LAYER_DTYPE", "").lower()
    if layer_dtype_str in ("fp16", "float16"):
        layer_dtype = torch.float16
    elif layer_dtype_str in ("bf16", "bfloat16"):
        layer_dtype = torch.bfloat16
    elif layer_dtype_str and layer_dtype_str not in ("", "none", "fp32", "float32"):
        print(f"[build_models] WARNING: LAYER_DTYPE='{layer_dtype_str}' is not supported, only fp16/bf16 are supported. Ignoring this setting.")
        layer_dtype = None
    else:
        layer_dtype = None

    # Decide whether to apply torch.compile based on environment variable
    use_compile = os.environ.get("USE_COMPILE", "0")
    if use_compile.lower() in ("1", "true", "yes"):
        compile_mode = os.environ.get("COMPILE_MODE", "reduce-overhead")
        model.apply_compile(mode=compile_mode, dtype=layer_dtype)
    elif layer_dtype is not None:
        # Only convert precision, do not compile
        model.convert_decoder_dtype(layer_dtype)

    return model, model_cfg


def _hf_download(repo_id):
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=repo_id)
    return os.path.join(local_dir, "model.ckpt"), os.path.join(local_dir, "assets", "mhr_model.pt")


def load_sam_3d_body_hf(repo_id, **kwargs):
    ckpt_path, mhr_path = _hf_download(repo_id)
    return load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path)
