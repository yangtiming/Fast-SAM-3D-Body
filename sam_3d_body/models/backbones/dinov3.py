# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import torch
from torch import nn


# Environment variable to enable TensorRT backbone
USE_TRT_BACKBONE = os.environ.get("USE_TRT_BACKBONE", "0") == "1"
TRT_BACKBONE_PATH = os.environ.get("TRT_BACKBONE_PATH", "")

# Environment variable to enable torch.compile for backbone
USE_COMPILE_BACKBONE = os.environ.get("USE_COMPILE_BACKBONE", "0") == "1"


class Dinov3Backbone(nn.Module):
    def __init__(
        self, name="dinov2_vitb14", pretrained_weight=None, cfg=None, *args, **kwargs
    ):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self._use_tensorrt = False
        self._trt_backbone = None

        # Check if TensorRT mode is requested
        if USE_TRT_BACKBONE:
            engine_path = TRT_BACKBONE_PATH
            if not engine_path:
                # Default path
                checkpoint_dir = getattr(cfg, 'LOCAL_CHECKPOINT_PATH', None)
                if checkpoint_dir:
                    engine_path = os.path.join(checkpoint_dir, "backbone_trt", "backbone_dinov3_fp16.engine")
                else:
                    engine_path = "./checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine"

            if os.path.exists(engine_path):
                from .dinov3_tensorrt import create_tensorrt_backbone
                print(f"[Dinov3Backbone] Using TensorRT engine: {engine_path}")
                self._trt_backbone = create_tensorrt_backbone(
                    engine_path=engine_path,
                    name=name,
                )
                self._use_tensorrt = True
                self._compiled = False  # TensorRT mode does not need compile
                self.patch_size = self._trt_backbone.patch_size
                self.embed_dim = self.embed_dims = self._trt_backbone.embed_dim
                return
            else:
                print(f"[Dinov3Backbone] TensorRT engine not found: {engine_path}, falling back to PyTorch")

        # Prefer local cache to avoid network requests
        local_cache = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov3_main")
        if os.path.exists(local_cache):
            # Use local cache (offline mode)
            self.encoder = torch.hub.load(
                local_cache,
                self.name,
                source="local",
                pretrained=False,
                drop_path=self.cfg.MODEL.BACKBONE.DROP_PATH_RATE,
            )
        else:
            # Fall back to online loading
            self.encoder = torch.hub.load(
                "facebookresearch/dinov3",
                self.name,
                source="github",
                pretrained=False,
                drop_path=self.cfg.MODEL.BACKBONE.DROP_PATH_RATE,
                trust_repo=True,
            )
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.embed_dims = self.encoder.embed_dim
        self._compiled = False

        # If USE_COMPILE_BACKBONE env var is set, automatically apply torch.compile
        if USE_COMPILE_BACKBONE:
            self.apply_compile()

    def apply_compile(self, mode: str = "reduce-overhead"):
        """
        Apply torch.compile to DINOv3 backbone for faster inference.

        Args:
            mode: torch.compile mode, one of "default", "reduce-overhead", "max-autotune"
        """
        if self._use_tensorrt:
            print("[Dinov3Backbone] TensorRT mode enabled, skipping torch.compile")
            return

        if self._compiled:
            print("[Dinov3Backbone] Already compiled, skipping")
            return

        print(f"[Dinov3Backbone] Applying torch.compile with mode='{mode}'")

        # Compile _forward_impl, which includes the full get_intermediate_layers call
        # Use dynamic=True to support different batch sizes (single/multi-person)
        self._forward_compiled = torch.compile(
            self._forward_impl,
            mode=mode,
            fullgraph=False,  # Allow graph breaks for get_intermediate_layers compatibility
            dynamic=True,
        )
        self._compiled = True
        print("[Dinov3Backbone] torch.compile applied successfully")

    def _forward_impl(self, x):
        """Actual forward implementation, used for torch.compile."""
        return self.encoder.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]

    def forward(self, x, extra_embed=None):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        # Use TensorRT if available
        if self._use_tensorrt:
            return self._trt_backbone(x, extra_embed)

        assert extra_embed is None, "Not Implemented Yet"

        # Use compiled version or original version
        if self._compiled:
            return self._forward_compiled(x)
        else:
            return self._forward_impl(x)

    def get_layer_depth(self, param_name: str, prefix: str = "encoder."):
        """Get the layer-wise depth of a parameter.
        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.
        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        # TensorRT mode: return dummy values
        if self._use_tensorrt:
            return self._trt_backbone.get_layer_depth(param_name, prefix)

        num_layers = self.encoder.n_blocks + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix) :]

        if param_name in ("cls_token", "pos_embed", "storage_tokens"):
            layer_depth = 0
        elif param_name.startswith("patch_embed"):
            layer_depth = 0
        elif param_name.startswith("blocks"):
            layer_id = int(param_name.split(".")[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers
