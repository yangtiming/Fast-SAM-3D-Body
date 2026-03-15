# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
SAM 3D Body GPU Profiling Script using NVIDIA Nsight Python
Usage: python profile_nsight.py --image_path <image_path>

First set the ncu path:
export PATH=/usr/local/cuda/bin:$PATH


# Activate your environment
conda activate fast_sam_3d_body

# Make sure ncu is in PATH
export PATH=/usr/local/cuda/bin:$PATH

# Basic profiling (timing + memory)
MHR_NO_CORRECTIVES=1 python profile_nsight.py --image_path ./notebook/images/dancing.jpg --detector yolo --detector_model ./checkpoints/yolo/yolo11m.engine



KEYPOINT_PROMPT_INTERM_INTERVAL=1 INTERM_PRED_INTERVAL=1 MHR_NO_CORRECTIVES=1  MHR_USE_CUDA_GRAPH=1



KEYPOINT_PROMPT_INTERM_INTERVAL=1 INTERM_PRED_INTERVAL=1 MHR_NO_CORRECTIVES=1 MHR_USE_CUDA_GRAPH=1 python profile_nsight.py --image_path ./notebook/images/dancing.jpg --detector yolo --detector_model ./checkpoints/yolo/yolo11m.engine


KEYPOINT_PROMPT_INTERM_LAYERS=0,3 INTERM_PRED_LAYERS=0,3 MHR_NO_CORRECTIVES=1 MHR_USE_CUDA_GRAPH=1  python demo_human.py --image_path ./notebook/images/dancing.jpg --detector yolo --detector_model ./checkpoints/yolo/yolo11m.engine


KEYPOINT_PROMPT_INTERM_LAYERS=0,1,2,3 BODY_INTERM_PRED_LAYERS=0,1,2,3 HAND_INTERM_PRED_LAYERS=0,1,2,3 MHR_NO_CORRECTIVES=1 MHR_USE_CUDA_GRAPH=1 



USE_TRT_BACKBONE=1 USE_COMPILE=1 KEYPOINT_PROMPT_INTERM_INTERVAL=1 INTERM_PRED_INTERVAL=1 MHR_NO_CORRECTIVES=1 MHR_USE_CUDA_GRAPH=1  python demo_human.py --image_path ./notebook/images/dancing.jpg --detector yolo --detector_model ./checkpoints/yolo/yolo11m.engine


PARALLEL_DECODERS=0


SKIP_KEYPOINT_PROMPT=1 FOV_TRT=1 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0 USE_COMPILE_BACKBONE=1 USE_COMPILE=1 KEYPOINT_PROMPT_INTERM_INTERVAL=1 INTERM_PRED_INTERVAL=1 MHR_NO_CORRECTIVES=1 MHR_USE_CUDA_GRAPH=1 
python profile_nsight.py --image_path ./notebook/images/dancing.jpg   --detector yolo_pose \
    --detector_model ./checkpoints/yolo/yolo11m-pose.engine \
    --hand_box_source yolo_pose




python demo_human.py --image_path ./notebook/images/dancing.jpg \
  --detector yolo_pose \
    --detector_model ./checkpoints/yolo/yolo11m-pose.engine \
    --hand_box_source yolo_pose



INTERM_TIMING=1 GPU_HAND_PREP=1 LAYER_DTYPE=fp16 SKIP_KEYPOINT_PROMPT=1 FOV_TRT=1 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0 USE_COMPILE_BACKBONE=1 USE_COMPILE=1 KEYPOINT_PROMPT_INTERM_INTERVAL=1 INTERM_PRED_INTERVAL=1 MHR_NO_CORRECTIVES=1 MHR_USE_CUDA_GRAPH=1   python profile_nsight.py --image_path ./notebook/images/dancing.jpg   --detector yolo_pose \
    --detector_model ./checkpoints/yolo/yolo11m-pose.engine \
    --hand_box_source yolo_pose




INTERM_TIMING=1 GPU_HAND_PREP=0 LAYER_DTYPE=fp16 SKIP_KEYPOINT_PROMPT=1 FOV_TRT=1 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0 USE_COMPILE_BACKBONE=1 USE_COMPILE=1 KEYPOINT_PROMPT_INTERM_INTERVAL=1 INTERM_PRED_INTERVAL=1 MHR_NO_CORRECTIVES=1 MHR_USE_CUDA_GRAPH=1  python demo_human.py --image_path ./notebook/images/dancing.jpg \
  --detector yolo_pose \
    --detector_model ./checkpoints/yolo/yolo11m-pose.engine \
    --hand_box_source yolo_pose










#####


 GPU_HAND_PREP=1 LAYER_DTYPE=fp32 SKIP_KEYPOINT_PROMPT=1 FOV_TRT=1 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0 USE_COMPILE_BACKBONE=1 USE_COMPILE=1 DECODER_COMPILE=1 MHR_USE_CUDA_GRAPH=0 KEYPOINT_PROMPT_INTERM_INTERVAL=999  BODY_INTERM_PRED_LAYERS=0,1,2 HAND_INTERM_PRED_LAYERS=0,1 MHR_NO_CORRECTIVES=1 python demo_human.py --image_path  ./notebook/images/dancing.jpg  --detector yolo_pose     --detector_model ./checkpoints/yolo/yolo11m-pose.engine     --hand_box_source yolo_pose


 GPU_HAND_PREP=1 LAYER_DTYPE=fp32 SKIP_KEYPOINT_PROMPT=1 FOV_TRT=1 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0 USE_COMPILE_BACKBONE=1 USE_COMPILE=1 DECODER_COMPILE=1 MHR_USE_CUDA_GRAPH=0 KEYPOINT_PROMPT_INTERM_INTERVAL=999  BODY_INTERM_PRED_LAYERS=0,1,2 HAND_INTERM_PRED_LAYERS=0,1 MHR_NO_CORRECTIVES=1 python demo_human.py --image_path  ./notebook/images/dancing.jpg  --detector yolo_pose     --detector_model ./checkpoints/yolo/yolo11m-pose.engine     --hand_box_source yolo_pose


 GPU_HAND_PREP=1 LAYER_DTYPE=fp32 SKIP_KEYPOINT_PROMPT=1 FOV_TRT=1 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0 USE_COMPILE_BACKBONE=1 USE_COMPILE=1 DECODER_COMPILE=1 MHR_USE_CUDA_GRAPH=0 KEYPOINT_PROMPT_INTERM_INTERVAL=999  BODY_INTERM_PRED_LAYERS=0,1,2 HAND_INTERM_PRED_LAYERS=0,1 MHR_NO_CORRECTIVES=1 python demo_human.py --image_path ./notebook/images/dancing.jpg  --detector yolo_pose     --detector_model ./checkpoints/yolo/yolo11m-pose.engine     --hand_box_source yolo_pose

"""

import argparse
import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

import cv2
import numpy as np
import torch

# Check if nsight-python is available
try:
    import nsight
    NSIGHT_AVAILABLE = True
    print("running nsight")
except ImportError:
    NSIGHT_AVAILABLE = False
    print("Warning: nsight-python not installed. Using basic timing only.")

import nvtx  # NVIDIA Tools Extension for custom markers


class SAM3DBodyProfiler:
    """SAM 3D Body GPU performance profiler"""

    def __init__(
        self,
        hf_repo_id: str = "facebook/sam-3d-body-dinov3",
        detector_name: str = "yolo",
        detector_model: str = "./checkpoints/yolo/yolo11n.pt",
        local_checkpoint_path: str = "",  # Local checkpoint path
        hand_box_source: str = "body_decoder",  # Hand box source
    ):
        self.hf_repo_id = hf_repo_id
        self.detector_name = detector_name
        self.detector_model = detector_model
        self.local_checkpoint_path = local_checkpoint_path
        self.hand_box_source = hand_box_source
        self.estimator = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_mask = False

    def setup(self, sam_path: str = ""):
        """Load models"""
        print("Loading SAM 3D Body model...")
        from notebook.utils import setup_sam_3d_body

        with nvtx.annotate("model_loading", color="blue"):
            self.estimator = setup_sam_3d_body(
                hf_repo_id=self.hf_repo_id,
                detector_name=self.detector_name,
                detector_model=self.detector_model,
                segmentor_path=sam_path,  # Pass SAM path
                fov_name="moge2",
                local_checkpoint_path=self.local_checkpoint_path,  # Use local checkpoint
            )
        self.use_mask = bool(sam_path)  # Enable mask if SAM path is provided
        print("Model loaded!")

    @nvtx.annotate("full_inference", color="green")
    def run_inference(self, image_path: str):
        """Run inference and return outputs"""
        # Note: Decoder uses float32 and cannot use Flash Attention
        # Backbone (DINOv3) uses bfloat16 and can automatically use Flash Attention
        outputs = self.estimator.process_one_image(
            image_path,
            use_mask=self.use_mask,
            hand_box_source=self.hand_box_source,
        )
        return outputs


def warmup_gpu(profiler: SAM3DBodyProfiler, image_path: str, n_warmup: int = 2):
    """GPU warmup"""
    print(f"\nWarming up GPU with {n_warmup} iterations...")
    for i in range(n_warmup):
        with nvtx.annotate(f"warmup_{i}", color="orange"):
            _ = profiler.run_inference(image_path)
            torch.cuda.synchronize()
    print("Warmup complete!")


def profile_with_nsight(profiler: SAM3DBodyProfiler, image_path: str, output_dir: str):
    """GPU kernel profiling using nsight-python"""

    if not NSIGHT_AVAILABLE:
        print("nsight-python not available, skipping kernel profiling")
        return

    print("\n" + "=" * 60)
    print("Running Nsight Compute Kernel Profiling")
    print("=" * 60)
    print("Note: nsight.analyze.kernel requires direct CUDA kernel annotation.")
    print("For PyTorch models, using torch.profiler is more effective.")
    print("Run with --layer_profile for detailed kernel analysis.")
    return None


def profile_with_nvtx(profiler: SAM3DBodyProfiler, image_path: str, output_dir: str, n_runs: int = 5):
    """Performance profiling using NVTX + torch.cuda.Event"""

    print("\n" + "=" * 60)
    print("Running NVTX + CUDA Event Profiling")
    print("=" * 60)

    timings = []

    for i in range(n_runs):
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with nvtx.annotate(f"inference_run_{i}", color="green"):
            outputs = profiler.run_inference(image_path)
        end_event.record()

        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        timings.append(elapsed_ms)
        print(f"  Run {i+1}/{n_runs}: {elapsed_ms:.2f} ms")

    # Statistics
    avg_time = np.mean(timings)
    std_time = np.std(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)

    print(f"\nTiming Statistics (n={n_runs}):")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Std Dev: {std_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")

    return timings


def profile_memory(profiler: SAM3DBodyProfiler, image_path: str):
    """GPU memory profiling"""

    print("\n" + "=" * 60)
    print("GPU Memory Profiling")
    print("=" * 60)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Record initial memory
    initial_mem = torch.cuda.memory_allocated() / 1024**2
    initial_reserved = torch.cuda.memory_reserved() / 1024**2

    print(f"Before inference:")
    print(f"  Allocated: {initial_mem:.2f} MB")
    print(f"  Reserved: {initial_reserved:.2f} MB")

    # Run inference
    with nvtx.annotate("memory_profile_inference", color="purple"):
        outputs = profiler.run_inference(image_path)
    torch.cuda.synchronize()

    # Record peak memory
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

    # Record post-inference memory
    after_mem = torch.cuda.memory_allocated() / 1024**2
    after_reserved = torch.cuda.memory_reserved() / 1024**2

    print(f"\nAfter inference:")
    print(f"  Allocated: {after_mem:.2f} MB")
    print(f"  Reserved: {after_reserved:.2f} MB")

    print(f"\nPeak memory:")
    print(f"  Allocated: {peak_mem:.2f} MB")
    print(f"  Reserved: {peak_reserved:.2f} MB")

    print(f"\nMemory delta:")
    print(f"  Inference used: {peak_mem - initial_mem:.2f} MB (peak)")

    return {
        "initial_allocated_mb": initial_mem,
        "after_allocated_mb": after_mem,
        "peak_allocated_mb": peak_mem,
        "peak_reserved_mb": peak_reserved,
    }


def profile_layer_by_layer(profiler: SAM3DBodyProfiler, image_path: str):
    """Layer-by-layer GPU kernel profiling (using torch.profiler)"""

    print("\n" + "=" * 60)
    print("Layer-by-Layer Profiling (torch.profiler)")
    print("=" * 60)

    # Use PyTorch profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with nvtx.annotate("torch_profiler_inference", color="red"):
            outputs = profiler.run_inference(image_path)
            torch.cuda.synchronize()

    # Print most time-consuming CUDA kernels
    print("\nTop 20 CUDA kernels by GPU time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Print most time-consuming operations (by self_cuda_time)
    print("\nTop 20 operations by self CUDA time:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

    return prof


def patch_human_detector(estimator):
    """Add timing instrumentation to Human Detector (ViTDet/YOLO)"""
    from sam_3d_body.models.modules.timing_utils import get_timing, time_cuda_op

    # Try multiple paths to find human_detector
    detector = None
    for attr in ['human_detector', 'detector', 'human_det']:
        detector = getattr(estimator, attr, None)
        if detector is not None:
            break

    if detector is None:
        print("  Warning: Could not find human_detector")
        return False

    # Check if it is a YOLO detector
    detector_name = getattr(detector, 'name', '')
    is_yolo = detector_name == 'yolo' or detector_name.startswith('yolo11')

    if is_yolo:
        # YOLO detector: patch run_human_detection method
        if hasattr(detector, '_original_run_human_detection'):
            return True  # Already patched

        original_func = detector.run_human_detection

        def timed_run_human_detection(*args, **kwargs):
            timing = get_timing()
            if timing.enabled:
                result, elapsed = time_cuda_op(lambda: original_func(*args, **kwargs))
                timing.add_module_time(elapsed, "human_detector")
                return result
            else:
                return original_func(*args, **kwargs)

        detector._original_run_human_detection = original_func
        detector.run_human_detection = timed_run_human_detection
        print("  Patched human_detector (YOLO11)")
        return True

    # ViTDet: try multiple paths to find a patchable object
    target = None
    for attr in ['predictor', 'model', 'detector']:
        target = getattr(detector, attr, None)
        if target is not None and callable(getattr(target, '__call__', None)):
            break

    if target is None:
        # Directly patch the detector itself
        target = detector

    # Save the original __call__ method
    if hasattr(target, '_original_call'):
        return True  # Already patched

    if hasattr(target, '__call__'):
        original_call = target.__call__

        def timed_call(*args, **kwargs):
            timing = get_timing()
            if timing.enabled:
                result, elapsed = time_cuda_op(lambda: original_call(*args, **kwargs))
                timing.add_module_time(elapsed, "human_detector")
                return result
            else:
                return original_call(*args, **kwargs)

        target._original_call = original_call
        target.__call__ = timed_call
        print("  Patched human_detector (ViTDet)")
        return True

    print(f"  Warning: Could not patch human_detector, target type: {type(target)}")
    return False


def patch_fov_estimator(estimator):
    """Add timing instrumentation to FOV Estimator (MoGe)"""
    from sam_3d_body.models.modules.timing_utils import get_timing, time_cuda_op

    # Try multiple paths to find fov_estimator
    fov_estimator = None
    for attr in ['fov_estimator', 'fov_est', 'moge']:
        fov_estimator = getattr(estimator, attr, None)
        if fov_estimator is not None:
            break

    if fov_estimator is None:
        print("  Warning: Could not find fov_estimator")
        return False

    # Try multiple paths to find the model
    # Note: FOVEstimator wrapper may have a nested fov_estimator attribute
    model = None
    search_targets = [fov_estimator]

    # If fov_estimator has a fov_estimator attribute, check that first
    if hasattr(fov_estimator, 'fov_estimator'):
        inner = fov_estimator.fov_estimator
        if inner is not None:
            search_targets.insert(0, inner)

    for target in search_targets:
        for attr in ['model', 'moge', 'net', 'network', 'backbone']:
            model = getattr(target, attr, None)
            if model is not None and isinstance(model, torch.nn.Module):
                break
        if model is not None:
            break
        # If target itself is an nn.Module
        if isinstance(target, torch.nn.Module):
            model = target
            break

    if model is None:
        print(f"  Warning: Could not find model in fov_estimator")
        print(f"    fov_estimator type: {type(fov_estimator)}")
        print(f"    fov_estimator attrs: {[a for a in dir(fov_estimator) if not a.startswith('_')][:20]}")
        if hasattr(fov_estimator, 'fov_estimator'):
            inner = fov_estimator.fov_estimator
            print(f"    inner fov_estimator type: {type(inner)}")
            print(f"    inner fov_estimator attrs: {[a for a in dir(inner) if not a.startswith('_')][:20]}")
        return False

    # Save the original forward method
    if hasattr(model, '_original_forward'):
        return True  # Already patched

    original_forward = model.forward

    def timed_forward(*args, **kwargs):
        timing = get_timing()
        if timing.enabled:
            result, elapsed = time_cuda_op(lambda: original_forward(*args, **kwargs))
            timing.add_module_time(elapsed, "fov_estimator")
            return result
        else:
            return original_forward(*args, **kwargs)

    model._original_forward = original_forward
    model.forward = timed_forward
    print("  Patched fov_estimator (MoGe)")
    return True


def patch_dinov3_blocks(estimator):
    """
    Add timing instrumentation to DINOv3 backbone Blocks by replacing forward methods
    """
    from sam_3d_body.models.modules.timing_utils import get_timing, time_cuda_op

    # Get the actual model from the estimator
    model = None
    if hasattr(estimator, 'model'):
        model = estimator.model
    elif hasattr(estimator, 'sam3d_body'):
        model = estimator.sam3d_body
    else:
        # Try to find nn.Module attributes
        for attr_name in dir(estimator):
            if attr_name.startswith('_'):
                continue
            attr = getattr(estimator, attr_name, None)
            if isinstance(attr, torch.nn.Module):
                model = attr
                print(f"  Found model at estimator.{attr_name}")
                break

    if model is None:
        print("  Warning: Could not find model in estimator")
        print(f"  Estimator type: {type(estimator)}")
        print(f"  Estimator attributes: {[a for a in dir(estimator) if not a.startswith('_')]}")
        return 0

    # Recursively find backbone
    def find_backbone(obj, path="model"):
        """Recursively find backbone"""
        if not isinstance(obj, torch.nn.Module):
            return None, None
        if hasattr(obj, 'encoder') and hasattr(obj.encoder, 'blocks'):
            return obj.encoder, path + ".encoder"
        if hasattr(obj, 'blocks'):
            blocks = obj.blocks
            if hasattr(blocks, '__len__') and len(blocks) > 0:
                first_block = blocks[0] if hasattr(blocks, '__getitem__') else list(blocks)[0]
                if hasattr(first_block, 'attn') and hasattr(first_block, 'mlp'):
                    return obj, path
        for name, child in obj.named_children():
            result = find_backbone(child, f"{path}.{name}")
            if result[0] is not None:
                return result
        return None, None

    backbone, backbone_path = find_backbone(model)

    if backbone is None:
        print("  Warning: Could not find DINOv3 backbone to patch")
        # Print top-level model structure for debugging
        print("  Model structure (top 2 levels):")
        for name, child in model.named_children():
            print(f"    - {name}: {type(child).__name__}")
            for name2, child2 in child.named_children():
                print(f"        - {name2}: {type(child2).__name__}")
        return 0

    print(f"  Found backbone at: {backbone_path}")

    # Find blocks
    blocks = None
    if hasattr(backbone, 'blocks'):
        blocks = backbone.blocks
    elif hasattr(backbone, 'transformer') and hasattr(backbone.transformer, 'resblocks'):
        blocks = backbone.transformer.resblocks

    if blocks is None:
        print("  Warning: Could not find blocks in backbone")
        return 0

    patched_count = 0
    for block in blocks:
        # Save original forward
        if hasattr(block, '_original_forward'):
            continue  # Already patched

        block._original_forward = block.forward

        # Create new forward method
        def make_timed_forward(blk):
            def timed_forward(x, *args, **kwargs):
                timing = get_timing()
                if timing.enabled:
                    # DINOv3 Block structure: norm1 -> attn -> norm2 -> mlp
                    # We need to time attn and mlp separately

                    # Attention part
                    if hasattr(blk, 'attn') and hasattr(blk, 'norm1'):
                        attn_input = blk.norm1(x)
                        attn_out, attn_time = time_cuda_op(lambda: blk.attn(attn_input))
                        if hasattr(blk, 'ls1'):
                            attn_out = blk.ls1(attn_out)
                        x = x + attn_out
                        timing.add_attention_time(attn_time, "self", component="backbone")

                        # FFN part
                        if hasattr(blk, 'mlp') and hasattr(blk, 'norm2'):
                            mlp_input = blk.norm2(x)
                            mlp_out, ffn_time = time_cuda_op(lambda: blk.mlp(mlp_input))
                            if hasattr(blk, 'ls2'):
                                mlp_out = blk.ls2(mlp_out)
                            x = x + mlp_out
                            timing.add_ffn_time(ffn_time, component="backbone")
                        return x
                    else:
                        # Fallback: use original forward
                        return blk._original_forward(x, *args, **kwargs)
                else:
                    return blk._original_forward(x, *args, **kwargs)
            return timed_forward

        block.forward = make_timed_forward(block)
        patched_count += 1

    return patched_count


def profile_ffn_vs_attention(profiler: SAM3DBodyProfiler, image_path: str, n_runs: int = 3):
    """
    Precisely analyze FFN and Attention time proportions.
    Uses built-in model timing to directly measure each attention and FFN module's time.
    """
    from sam_3d_body.models.modules.timing_utils import get_timing

    print("\n" + "=" * 60)
    print("FFN vs Attention Timing Analysis (Direct Timing)")
    print("=" * 60)

    timing = get_timing()

    # Patch individual modules
    print("\nPatching models for timing...")

    # Patch DINOv3 backbone blocks
    patched = patch_dinov3_blocks(profiler.estimator)
    print(f"  Patched {patched} blocks in DINOv3 backbone")

    # Patch human detector
    patch_human_detector(profiler.estimator)

    # Patch FOV estimator
    patch_fov_estimator(profiler.estimator)

    # Warmup
    print(f"\nWarming up with 1 iteration...")
    _ = profiler.run_inference(image_path)
    torch.cuda.synchronize()

    # Enable timing and run
    print(f"\nRunning {n_runs} iterations with timing enabled...")
    timing.enable()

    total_times = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = profiler.run_inference(image_path)
        end.record()
        torch.cuda.synchronize()

        elapsed = start.elapsed_time(end)
        total_times.append(elapsed)
        print(f"  Run {i+1}/{n_runs}: {elapsed:.2f} ms")

    timing.disable()

    # Print results
    avg_time = np.mean(total_times)
    summary = timing.get_summary()

    print("\n" + "=" * 60)
    print("FFN vs Attention Summary")
    print("=" * 60)

    print(f"\nEnd-to-end inference time (avg): {avg_time:.2f} ms")
    print(f"\nTransformer components (Attention + FFN):")
    print(f"  Total measured: {summary['total_attn_ffn_ms']:.2f} ms")

    print(f"\n  Attention:")
    print(f"    - Total time: {summary['attention_time_ms']:.2f} ms ({summary['attention_percentage']:.1f}%)")
    print(f"    - Self-attention: {summary['self_attn_time_ms']:.2f} ms")
    print(f"    - Cross-attention: {summary['cross_attn_time_ms']:.2f} ms")

    print(f"\n  FFN (MLP):")
    print(f"    - Total time: {summary['ffn_time_ms']:.2f} ms ({summary['ffn_percentage']:.1f}%)")

    if summary['attention_time_ms'] > 0:
        ratio = summary['ffn_time_ms'] / summary['attention_time_ms']
        print(f"\n  FFN / Attention ratio: {ratio:.2f}x")

    # Display breakdown by component
    by_component = summary.get('by_component', {})
    if by_component:
        print("\n" + "-" * 60)
        print("Breakdown by Component")
        print("-" * 60)

        for comp_name, comp_data in by_component.items():
            comp_attn = comp_data.get('attention_time_ms', 0)
            comp_ffn = comp_data.get('ffn_time_ms', 0)
            comp_total = comp_attn + comp_ffn
            comp_self_attn = comp_data.get('self_attn_time_ms', 0)
            comp_cross_attn = comp_data.get('cross_attn_time_ms', 0)
            comp_counts = comp_data.get('call_counts', {})

            print(f"\n  [{comp_name.upper()}] Total: {comp_total:.2f} ms")

            if comp_total > 0:
                attn_pct = 100 * comp_attn / comp_total
                ffn_pct = 100 * comp_ffn / comp_total
                print(f"    Attention: {comp_attn:.2f} ms ({attn_pct:.1f}%)")
                if comp_self_attn > 0:
                    print(f"      - Self-attention: {comp_self_attn:.2f} ms")
                if comp_cross_attn > 0:
                    print(f"      - Cross-attention: {comp_cross_attn:.2f} ms")
                print(f"    FFN: {comp_ffn:.2f} ms ({ffn_pct:.1f}%)")

                if comp_attn > 0:
                    comp_ratio = comp_ffn / comp_attn
                    print(f"    FFN/Attention ratio: {comp_ratio:.2f}x")

            print(f"    Call counts: {dict(comp_counts)}")

    # Display other module timings (human_detector, fov_estimator)
    by_module = summary.get('by_module', {})
    if by_module:
        print("\n" + "-" * 60)
        print("Other Modules Timing")
        print("-" * 60)

        for mod_name, mod_data in by_module.items():
            mod_time = mod_data.get('total_time_ms', 0)
            mod_count = mod_data.get('call_count', 0)
            mod_avg = mod_time / mod_count if mod_count > 0 else 0
            print(f"\n  [{mod_name.upper()}]")
            print(f"    Total time: {mod_time:.2f} ms")
            print(f"    Call count: {mod_count}")
            print(f"    Avg per call: {mod_avg:.2f} ms")

    # Calculate overall time distribution
    total_transformer = summary['total_attn_ffn_ms']
    total_other_modules = sum(m.get('total_time_ms', 0) for m in by_module.values())
    total_measured = total_transformer + total_other_modules
    e2e_total = avg_time * n_runs  # Total e2e time across all runs

    if total_measured > 0:
        print("\n" + "-" * 60)
        print("Overall Time Distribution (across all runs)")
        print("-" * 60)
        print(f"\n  Total measured: {total_measured:.2f} ms")
        print(f"  End-to-end total ({n_runs} runs): {e2e_total:.2f} ms")
        print(f"\n  Breakdown (% of e2e):")
        print(f"    - Transformer (Attn+FFN): {total_transformer:.2f} ms ({100*total_transformer/e2e_total:.1f}%)")

        for mod_name, mod_data in by_module.items():
            mod_time = mod_data.get('total_time_ms', 0)
            print(f"    - {mod_name}: {mod_time:.2f} ms ({100*mod_time/e2e_total:.1f}%)")

        unmeasured = e2e_total - total_measured
        if unmeasured > 0:
            print(f"    - Other (unmeasured): {unmeasured:.2f} ms ({100*unmeasured/e2e_total:.1f}%)")

    print("\n" + "=" * 60)

    return summary


def profile_ffn_vs_attention_profiler(profiler: SAM3DBodyProfiler, image_path: str, n_runs: int = 3):
    """
    Analyze FFN and Attention time proportions using torch.profiler (fallback method)
    """
    from torch.profiler import profile, ProfilerActivity

    print("\n" + "=" * 60)
    print("FFN vs Attention Timing Analysis (torch.profiler)")
    print("=" * 60)

    # Warmup
    print(f"\nWarming up with 1 iteration...")
    _ = profiler.run_inference(image_path)
    torch.cuda.synchronize()

    # Profile using torch.profiler
    print(f"\nProfiling with torch.profiler ({n_runs} runs)...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=True,  # Need stack to distinguish operation origins
    ) as prof:
        for _ in range(n_runs):
            _ = profiler.run_inference(image_path)
            torch.cuda.synchronize()

    # Analyze profiler results
    print("\n" + "-" * 60)
    print("Analyzing profiler results...")
    print("-" * 60)

    # Get CUDA time for all events
    key_averages = prof.key_averages()

    # Debug: print available attributes of the first event
    if len(key_averages) > 0:
        first_event = key_averages[0]
        cuda_attrs = [attr for attr in dir(first_event)
                      if ('cuda' in attr.lower() or 'device' in attr.lower())
                      and not attr.startswith('_')]
        print(f"  Available device/CUDA attributes: {cuda_attrs}")

        # Determine which attribute to use and print example values
        if hasattr(first_event, 'self_device_time_total'):
            print(f"  Using: self_device_time_total = {first_event.self_device_time_total}")
        elif hasattr(first_event, 'self_cuda_time_total'):
            print(f"  Using: self_cuda_time_total = {first_event.self_cuda_time_total}")
        elif hasattr(first_event, 'device_time'):
            print(f"  Using: device_time = {first_event.device_time}, count = {first_event.count}")
        elif hasattr(first_event, 'cuda_time'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print(f"  Using: cuda_time = {first_event.cuda_time}, count = {first_event.count}")

        # Print info for first few events for debugging
        print(f"\n  Sample events (first 5 with CUDA time):")
        count = 0
        for evt in key_averages:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cuda_t = getattr(evt, 'cuda_time', 0) * getattr(evt, 'count', 1)
            if cuda_t > 0 and count < 5:
                print(f"    {evt.key}: cuda_time={cuda_t:.0f}us, count={evt.count}")
                count += 1

    # Define operation categories
    # Attention-related ops: scaled_dot_product_attention, softmax, bmm (for QK^T and attn@V)
    ATTENTION_OPS = {
        'aten::scaled_dot_product_attention',
        'aten::_scaled_dot_product_attention',
        'aten::_efficient_attention_forward',
        'aten::_flash_attention_forward',
        'aten::softmax',
        'aten::_softmax',
        'aten::bmm',  # Used for attention
        'aten::baddbmm',
    }

    # FFN/MLP-related ops: linear, gelu, silu, relu
    FFN_OPS = {
        'aten::gelu',
        'aten::silu',
        'aten::silu_',
        'aten::relu',
        'aten::relu_',
    }

    # Linear ops (need stack trace to distinguish attention qkv/proj from FFN fc)
    LINEAR_OPS = {
        'aten::linear',
        'aten::addmm',
        'aten::mm',
    }

    attention_time = 0.0
    ffn_time = 0.0
    linear_time = 0.0
    other_time = 0.0
    total_cuda_time = 0.0

    attention_ops_detail = []
    ffn_ops_detail = []
    linear_ops_detail = []

    for event in key_averages:
        # Compatible with different PyTorch version attribute names
        # Prefer device_time (newer versions), then cuda_time (older versions)
        cuda_time_us = 0
        if hasattr(event, 'self_device_time_total'):
            cuda_time_us = event.self_device_time_total
        elif hasattr(event, 'self_cuda_time_total'):
            cuda_time_us = event.self_cuda_time_total
        elif hasattr(event, 'device_time_total'):
            cuda_time_us = event.device_time_total
        elif hasattr(event, 'cuda_time_total'):
            cuda_time_us = event.cuda_time_total
        elif hasattr(event, 'device_time'):
            cuda_time_us = event.device_time * event.count
        elif hasattr(event, 'cuda_time'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cuda_time_us = event.cuda_time * event.count  # cuda_time is the average

        cuda_time_ms = cuda_time_us / 1000.0 / n_runs  # Convert to ms and average

        if cuda_time_us <= 0:
            continue

        total_cuda_time += cuda_time_ms
        op_name = event.key

        # Use 'in' check for partial matching (some op names may have variants)
        is_attention = any(attn_op in op_name for attn_op in ATTENTION_OPS)
        is_ffn = any(ffn_op in op_name for ffn_op in FFN_OPS)
        is_linear = any(lin_op in op_name for lin_op in LINEAR_OPS)

        if is_attention:
            attention_time += cuda_time_ms
            attention_ops_detail.append((op_name, cuda_time_ms))
        elif is_ffn:
            ffn_time += cuda_time_ms
            ffn_ops_detail.append((op_name, cuda_time_ms))
        elif is_linear:
            linear_time += cuda_time_ms
            linear_ops_detail.append((op_name, cuda_time_ms))
        else:
            other_time += cuda_time_ms

    # Measure actual end-to-end time as reference
    print("\nMeasuring end-to-end inference time...")
    e2e_times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = profiler.run_inference(image_path)
        end.record()
        torch.cuda.synchronize()
        e2e_times.append(start.elapsed_time(end))

    avg_e2e_time = np.mean(e2e_times)

    # For Transformers, linear op distribution is roughly:
    # Attention: qkv (1x), proj (1x) = 2 linears
    # FFN: fc1 (1x), fc2 (1x) = 2 linears (for SwiGLU it's w12 + w3)
    # So linear time is roughly 50% attention, 50% FFN
    # But in practice FFN hidden_dim is typically 4x embed_dim, so FFN linears are heavier
    # With mlp_ratio=4, FFN linear time is about (4+1)/(1+1+4+1) = 5/7 ~ 71%
    # Attention linear time is about (1+1)/(1+1+4+1) = 2/7 ~ 29%
    # This is a rough estimate; actual values depend on specific architecture

    # Use a more conservative split, or let the user know linear is shared
    attn_linear_ratio = 0.3  # attention qkv + proj
    ffn_linear_ratio = 0.7   # FFN fc1 + fc2 (because hidden_dim is larger)

    attention_with_linear = attention_time + linear_time * attn_linear_ratio
    ffn_with_linear = ffn_time + linear_time * ffn_linear_ratio

    print("\n" + "=" * 60)
    print("FFN vs Attention Summary")
    print("=" * 60)

    print(f"\nEnd-to-end inference time (avg): {avg_e2e_time:.2f} ms")
    print(f"Total CUDA kernel time: {total_cuda_time:.2f} ms")

    print(f"\n--- Raw kernel breakdown ---")
    print(f"  Attention-specific ops (softmax, sdpa, bmm): {attention_time:.2f} ms")
    print(f"  FFN-specific ops (gelu, silu, relu): {ffn_time:.2f} ms")
    print(f"  Linear ops (mm, addmm, linear): {linear_time:.2f} ms")
    print(f"  Other ops: {other_time:.2f} ms")

    print(f"\n--- Estimated breakdown (linear split {attn_linear_ratio:.0%}/{ffn_linear_ratio:.0%}) ---")
    total_attn_ffn = attention_with_linear + ffn_with_linear
    if total_attn_ffn > 0:
        print(f"  Attention (estimated):")
        print(f"    - Total time: {attention_with_linear:.2f} ms")
        print(f"    - Percentage of (Attn+FFN): {100 * attention_with_linear / total_attn_ffn:.1f}%")

        print(f"\n  FFN (estimated):")
        print(f"    - Total time: {ffn_with_linear:.2f} ms")
        print(f"    - Percentage of (Attn+FFN): {100 * ffn_with_linear / total_attn_ffn:.1f}%")

        if attention_with_linear > 0:
            ffn_attn_ratio = ffn_with_linear / attention_with_linear
            print(f"\n  FFN / Attention ratio: {ffn_attn_ratio:.2f}x")

    # Detailed operation statistics
    print("\n" + "-" * 60)
    print("Detailed operation breakdown:")
    print("-" * 60)

    print("\n  Attention operations:")
    for op, t in sorted(attention_ops_detail, key=lambda x: -x[1]):
        print(f"    {op}: {t:.3f} ms")

    print("\n  FFN operations:")
    for op, t in sorted(ffn_ops_detail, key=lambda x: -x[1]):
        print(f"    {op}: {t:.3f} ms")

    print("\n  Linear operations (shared):")
    for op, t in sorted(linear_ops_detail, key=lambda x: -x[1]):
        print(f"    {op}: {t:.3f} ms")

    # Print top operations
    print("\n" + "-" * 60)
    print("Top 20 CUDA operations by time:")
    print("-" * 60)
    # Try different sort keys
    try:
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))
    except Exception:
        try:
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
        except Exception:
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    return {
        'e2e_time_ms': avg_e2e_time,
        'attention_time_ms': attention_with_linear,
        'ffn_time_ms': ffn_with_linear,
        'linear_time_ms': linear_time,
        'attention_percentage': 100 * attention_with_linear / total_attn_ffn if total_attn_ffn > 0 else 0,
        'ffn_percentage': 100 * ffn_with_linear / total_attn_ffn if total_attn_ffn > 0 else 0,
        'ffn_attn_ratio': ffn_with_linear / attention_with_linear if attention_with_linear > 0 else float('inf'),
    }


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SAM 3D Body GPU Profiler")
    print("=" * 60)
    print(f"Image: {args.image_path}")
    print(f"Model: {args.model}")
    print(f"Detector: {args.detector}" + (f" ({args.detector_model})" if args.detector in ["yolo", "yolo_pose"] else ""))
    print(f"Hand Box Source: {args.hand_box_source}")
    print(f"Output: {args.output_dir}")
    print(f"SAM2 Segmentor: {'✓ (' + args.sam_path + ')' if args.sam_path else '✗ (disabled)'}")
    print(f"Local Checkpoint: {'✓ (' + args.local_checkpoint + ')' if args.local_checkpoint else '✗ (using HuggingFace)'}")

    # Check image
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found - {args.image_path}")
        return

    # Initialize profiler
    profiler = SAM3DBodyProfiler(
        hf_repo_id=args.model,
        detector_name=args.detector,
        detector_model=args.detector_model,
        local_checkpoint_path=args.local_checkpoint,  # Local checkpoint
        hand_box_source=args.hand_box_source,
    )

    # Load model
    profiler.setup(sam_path=args.sam_path)

    # GPU warmup
    warmup_gpu(profiler, args.image_path, n_warmup=args.warmup)

    # 1. Basic timing analysis
    timings = profile_with_nvtx(profiler, args.image_path, args.output_dir, n_runs=args.runs)

    # 2. Memory profiling
    mem_stats = profile_memory(profiler, args.image_path)

    # 3. FFN vs Attention analysis
    if args.ffn_attention_profile:
        profile_ffn_vs_attention(profiler, args.image_path, n_runs=args.runs)

    # 4. Layer-by-layer analysis
    if args.layer_profile:
        prof = profile_layer_by_layer(profiler, args.image_path)

        # Export Chrome trace
        trace_path = os.path.join(args.output_dir, "torch_trace.json")
        prof.export_chrome_trace(trace_path)
        print(f"\nChrome trace saved to: {trace_path}")
        print("Open chrome://tracing in Chrome browser to view")

    # 5. Nsight Compute profiling (if available)
    if args.nsight_profile and NSIGHT_AVAILABLE:
        try:
            profile_with_nsight(profiler, args.image_path, args.output_dir)
        except Exception as e:
            print(f"Nsight profiling failed: {e}")
            print("Make sure 'ncu' is in PATH: export PATH=/usr/local/cuda/bin:$PATH")

    # Summary
    print("\n" + "=" * 60)
    print("PROFILING SUMMARY")
    print("=" * 60)
    print(f"Average inference time: {np.mean(timings):.2f} ms")
    print(f"Peak GPU memory: {mem_stats['peak_allocated_mb']:.2f} MB")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body GPU Profiler using Nsight Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic profiling (uses YOLO11 by default)
  python profile_nsight.py --image_path ./notebook/images/dancing.jpg

  # Use a different YOLO model variant
  python profile_nsight.py --image_path ./notebook/images/dancing.jpg --detector_model ./checkpoints/yolo/yolo11m.pt

  # Use ViTDet (Detectron2) detector
  python profile_nsight.py --image_path ./notebook/images/dancing.jpg --detector vitdet

  # Full profiling (including layer-by-layer and nsight)
  python profile_nsight.py --image_path ./notebook/images/dancing.jpg --layer_profile --nsight_profile --detector_model ./checkpoints/yolo/yolo11m.pt

  # More runs
  python profile_nsight.py --image_path ./notebook/images/dancing.jpg --runs 10

Note: Make sure ncu is in PATH before running:
  export PATH=/usr/local/cuda/bin:$PATH
        """
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default='./notebook/images/dancing.jpg',
        help="Input image path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./profile_output",
        help="Output directory (default: ./profile_output)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/sam-3d-body-dinov3",
        choices=["facebook/sam-3d-body-dinov3", "facebook/sam-3d-body-vith"],
        help="Model selection"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="yolo",
        choices=["vitdet", "yolo", "yolo_pose"],
        help="Human detector selection: vitdet (Detectron2), yolo (YOLO11), yolo_pose (YOLO11-Pose with keypoints) (default: yolo)"
    )
    parser.add_argument(
        "--hand_box_source",
        type=str,
        default="body_decoder",
        choices=["body_decoder", "yolo_pose"],
        help="Hand box source: body_decoder (from body decoder output), yolo_pose (computed from YOLO-Pose wrist positions) (default: body_decoder)"
    )
    parser.add_argument(
        "--detector_model",
        type=str,
        default="./checkpoints/yolo/yolo11n.pt",
        help="YOLO model path (default: ./checkpoints/yolo/yolo11n.pt)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of profiling runs (default: 5)"
    )
    parser.add_argument(
        "--layer_profile",
        action="store_true",
        help="Enable layer-by-layer profiling (torch.profiler)"
    )
    parser.add_argument(
        "--ffn_attention_profile",
        action="store_true",
        help="Enable FFN vs Attention time ratio analysis"
    )
    parser.add_argument(
        "--nsight_profile",
        action="store_true",
        help="Enable Nsight Compute kernel profiling"
    )
    parser.add_argument(
        "--sam_path",
        type=str,
        default="",#"./checkpoints/sam2.1_hiera_large.pt",
        help="SAM2 model path (e.g., /path/to/sam2); enables mask conditioning when provided"
    )
    parser.add_argument(
        "--local_checkpoint",
        type=str,
        default="./checkpoints/sam-3d-body-dinov3",
        help="Local checkpoint directory path (containing model.ckpt and model_config.yaml) to override HuggingFace config"
    )

    args = parser.parse_args()
    main(args)
