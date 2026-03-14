# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import time as _time
from typing import Optional

import torch
import torch.nn as nn

from ..modules.transformer import build_norm_layer, TransformerDecoderLayer


def _should_compile():
    """Check whether torch.compile should be used."""
    use_compile = os.environ.get("USE_COMPILE", "0")
    return use_compile.lower() in ("1", "true", "yes")


# IntermPred timing control
_INTERM_TIMING_ENABLED = os.environ.get("INTERM_TIMING", "0") == "1"
_INTERM_TIMING_WARMUP = 3
_INTERM_TIMING_CALL_COUNT = 0


class PromptableDecoder(nn.Module):
    """Cross-attention based Transformer decoder with prompts input."""

    def __init__(
        self,
        dims: int,
        context_dims: int,
        depth: int,
        num_heads: int,
        head_dims: int,
        mlp_dims: int,
        layer_scale_init_value: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        ffn_type: str = "origin",
        norm_cfg: dict = dict(type="LN", eps=1e-6),
        enable_twoway: bool = True,
        repeat_pe: bool = False,
        frozen: bool = False,
        do_interm_preds: bool = False,
        interm_pred_interval: int = 1,
        interm_pred_layers: set = None,
        do_keypoint_tokens: bool = False,
        keypoint_token_update: str = None,
    ):
        super().__init__()
        self.dims = dims
        self.depth = depth
        self.do_interm_preds = do_interm_preds
        self.interm_pred_interval = interm_pred_interval
        self.interm_pred_layers = interm_pred_layers
        self.keypoint_token_update = keypoint_token_update

        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TransformerDecoderLayer(
                    token_dims=dims,
                    context_dims=context_dims,
                    num_heads=num_heads,
                    head_dims=head_dims,
                    mlp_dims=mlp_dims,
                    layer_scale_init_value=layer_scale_init_value,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rates[i],
                    ffn_type=ffn_type,
                    enable_twoway=enable_twoway,
                    repeat_pe=repeat_pe,
                )
            )
        self.norm_final = build_norm_layer(norm_cfg, dims)

        self.frozen = frozen
        self._freeze_stages()
        self._compiled = False
        self._layer_dtype = None

    def apply_compile(self, mode: str = "reduce-overhead", dtype: torch.dtype = None):
        """
        Apply torch.compile to Transformer layers to accelerate inference.
        This method should be called after weights are loaded.

        Args:
            mode: torch.compile mode, options are "default", "reduce-overhead", "max-autotune"
            dtype: Precision for autocast, options are torch.float16, torch.bfloat16, None (no autocast)
        """
        if self._compiled:
            return

        print(f"[PromptableDecoder] Applying torch.compile with mode='{mode}', autocast_dtype={dtype}")

        # Only compile individual TransformerDecoderLayers, not the entire decoder forward
        # This avoids issues caused by dynamic control flow (IntermPred)
        # Use dynamic=True to support different batch_sizes (single/multi-person mode)
        for i, layer in enumerate(self.layers):
            self.layers[i] = torch.compile(layer, mode=mode, dynamic=True)

        self._compiled = True
        self._layer_dtype = dtype  # Used for autocast in forward

    def convert_layers_dtype(self, dtype: torch.dtype):
        """
        Set autocast precision (without compiling).

        Args:
            dtype: Target precision, e.g. torch.float16, torch.bfloat16
        """
        print(f"[PromptableDecoder] Setting autocast dtype to {dtype}")
        # No longer manually converting layer precision, using autocast instead
        self._layer_dtype = dtype

    def forward(
        self,
        token_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        token_augment: Optional[torch.Tensor] = None,
        image_augment: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        channel_first: bool = True,
        token_to_pose_output_fn=None,
        keypoint_token_update_fn=None,
        hand_embeddings=None,
        hand_augment=None,
        decoder_name: str = "decoder",
        override_interm_interval: Optional[int] = None,
        override_interm_layers: Optional[set] = None,
    ):
        """
        Args:
            token_embedding: [B, N, C]
            image_embedding: [B, C, H, W]
        """
        # Determine precision mode
        use_autocast = hasattr(self, '_layer_dtype') and self._layer_dtype is not None
        if use_autocast:
            autocast_dtype = self._layer_dtype
            device_type = token_embedding.device.type

        # Preprocessing: flatten image embedding
        if channel_first:
            image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
            if image_augment is not None:
                image_augment = image_augment.flatten(2).permute(0, 2, 1)
            if hand_embeddings is not None:
                hand_embeddings = hand_embeddings.flatten(2).permute(0, 2, 1)
                hand_augment = hand_augment.flatten(2).permute(0, 2, 1)
                if len(hand_augment) == 1:
                    assert len(hand_augment.shape) == 3
                    hand_augment = hand_augment.repeat(len(hand_embeddings), 1, 1)

        # Determine IntermPred configuration
        if "body" in decoder_name.lower():
            prefix = "BODY"
        elif "hand" in decoder_name.lower():
            prefix = "HAND"
        else:
            prefix = None

        def get_env_config():
            if prefix:
                specific_layers = os.environ.get(f'{prefix}_INTERM_PRED_LAYERS')
                specific_interval = os.environ.get(f'{prefix}_INTERM_PRED_INTERVAL')
                if specific_layers:
                    return None, set(int(x.strip()) for x in specific_layers.split(',') if x.strip())
                if specific_interval:
                    return int(specific_interval), None
            generic_layers = os.environ.get('INTERM_PRED_LAYERS')
            generic_interval = os.environ.get('INTERM_PRED_INTERVAL')
            if generic_layers:
                return None, set(int(x.strip()) for x in generic_layers.split(',') if x.strip())
            if generic_interval:
                return int(generic_interval), None
            return self.interm_pred_interval, self.interm_pred_layers

        if override_interm_layers is not None:
            effective_interval = None
            effective_layers = override_interm_layers
        elif override_interm_interval is not None:
            effective_interval = override_interm_interval
            effective_layers = None
        else:
            effective_interval, effective_layers = get_env_config()

        do_final_pred = self.do_interm_preds
        if do_final_pred:
            assert token_to_pose_output_fn is not None
            all_pose_outputs = []

        # For reusing IntermPred results
        last_pose_output = None
        curr_pose_output = None

        # DEBUG: Check for NaN in decoder layers
        _debug_nan_decoder = os.environ.get('DEBUG_NAN', '0') == '1'
        _nan_first_detected_layer = None

        # Define single-layer execution function
        def _run_single_layer(layer, token_emb, image_emb, token_aug, image_aug, token_msk, hand_emb, hand_aug):
            if hand_emb is None:
                return layer(token_emb, image_emb, token_aug, image_aug, token_msk)
            else:
                tok_out, img_out = layer(
                    token_emb,
                    torch.cat([image_emb, hand_emb], dim=1),
                    token_aug,
                    torch.cat([image_aug, hand_aug], dim=1),
                    token_msk,
                )
                return tok_out, img_out[:, : image_aug.shape[1]]

        # Decoder Layers loop
        # Use a single autocast context wrapping the entire loop, avoiding per-layer autocast enter/exit
        # This prevents NaN issues with autocast + torch.compile
        _autocast_ctx = torch.autocast(device_type=device_type, dtype=autocast_dtype) if use_autocast else None
        if _autocast_ctx is not None:
            _autocast_ctx.__enter__()

        try:
            for layer_idx, layer in enumerate(self.layers):
                token_embedding, image_embedding = _run_single_layer(
                    layer, token_embedding, image_embedding,
                    token_augment, image_augment, token_mask,
                    hand_embeddings, hand_augment
                )

                # DEBUG: Check for NaN after each layer
                if _debug_nan_decoder and _nan_first_detected_layer is None:
                    if torch.isnan(token_embedding).any() or torch.isnan(image_embedding).any():
                        _nan_first_detected_layer = layer_idx
                        print(f"          [DEBUG {decoder_name}] NaN first detected at layer {layer_idx}!")
                        print(f"          [DEBUG {decoder_name}]   token_embedding has_nan={torch.isnan(token_embedding).any().item()}, shape={token_embedding.shape}")
                        print(f"          [DEBUG {decoder_name}]   image_embedding has_nan={torch.isnan(image_embedding).any().item()}, shape={image_embedding.shape}")

                # Intermediate prediction
                if effective_layers is not None:
                    should_do_interm = (
                        self.do_interm_preds and
                        layer_idx < len(self.layers) - 1 and
                        layer_idx in effective_layers
                    )
                else:
                    should_do_interm = (
                        self.do_interm_preds and
                        layer_idx < len(self.layers) - 1 and
                        (layer_idx + 1) % effective_interval == 0
                    )

                if should_do_interm:
                    # IntermPred timing
                    global _INTERM_TIMING_CALL_COUNT
                    do_interm_timing = False
                    if _INTERM_TIMING_ENABLED:
                        _INTERM_TIMING_CALL_COUNT += 1
                        do_interm_timing = _INTERM_TIMING_CALL_COUNT > _INTERM_TIMING_WARMUP
                        if do_interm_timing:
                            torch.cuda.synchronize()
                            t_interm_start = _time.perf_counter()

                    curr_pose_output = token_to_pose_output_fn(
                        self.norm_final(token_embedding),
                        prev_pose_output=(
                            all_pose_outputs[-1] if len(all_pose_outputs) > 0 else None
                        ),
                        layer_idx=layer_idx,
                    )
                    all_pose_outputs.append(curr_pose_output)
                    last_pose_output = curr_pose_output  # Save for reuse

                    if do_interm_timing:
                        torch.cuda.synchronize()
                        t_interm_end = _time.perf_counter()
                        print(f"[IntermPred] {decoder_name} layer={layer_idx}: {(t_interm_end - t_interm_start)*1000:.2f}ms")

                # keypoint_token_update: executed every layer, using the latest pose_output (or reusing the previous layer's)
                if self.keypoint_token_update and layer_idx < len(self.layers) - 1:
                    assert keypoint_token_update_fn is not None
                    pose_for_update = curr_pose_output if should_do_interm else last_pose_output
                    if pose_for_update is not None:
                        token_embedding, token_augment, _, _ = keypoint_token_update_fn(
                            token_embedding, token_augment, pose_for_update, layer_idx
                        )

        finally:
            # Ensure autocast context is properly closed
            if _autocast_ctx is not None:
                _autocast_ctx.__exit__(None, None, None)

        # Final LayerNorm (ensure output is float32)
        out = self.norm_final(token_embedding)
        if use_autocast:
            out = out.float()

        # Final layer's pose output
        if do_final_pred:
            # Final IntermPred timing
            do_final_timing = False
            if _INTERM_TIMING_ENABLED:
                _INTERM_TIMING_CALL_COUNT += 1
                do_final_timing = _INTERM_TIMING_CALL_COUNT > _INTERM_TIMING_WARMUP
                if do_final_timing:
                    torch.cuda.synchronize()
                    t_final_start = _time.perf_counter()

            curr_pose_output = token_to_pose_output_fn(
                out,
                prev_pose_output=(
                    all_pose_outputs[-1] if len(all_pose_outputs) > 0 else None
                ),
                layer_idx=layer_idx,
            )
            all_pose_outputs.append(curr_pose_output)

            if do_final_timing:
                torch.cuda.synchronize()
                t_final_end = _time.perf_counter()
                print(f"[IntermPred] {decoder_name} FINAL: {(t_final_end - t_final_start)*1000:.2f}ms")

            return out, all_pose_outputs
        else:
            return out

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen:
            for layer in self.layers:
                layer.eval()
            self.norm_final.eval()
            for param in self.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
