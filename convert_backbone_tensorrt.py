#!/usr/bin/env python3
"""
Convert DINOv3 Backbone to TensorRT.

Usage:
    python convert_backbone_tensorrt.py --export_onnx
    python convert_backbone_tensorrt.py --convert_trt
    python convert_backbone_tensorrt.py --benchmark
    python convert_backbone_tensorrt.py --all

The backbone accepts:
    Input: [B, 3, 512, 512] RGB image (normalized)
    Output: [B, 1280, 32, 32] feature map



    Instructions:
Step 1: Export and convert to TensorRT

# All-in-one: export ONNX + convert TensorRT + benchmark
python convert_backbone_tensorrt.py --all

# Or run steps individually:
python convert_backbone_tensorrt.py --export_onnx    # Export ONNX
python convert_backbone_tensorrt.py --convert_trt    # Convert to TensorRT
python convert_backbone_tensorrt.py --benchmark      # Performance comparison

Step 2: Run inference with TensorRT

# Set environment variable to enable TensorRT backbone
USE_TRT_BACKBONE=1 python profile_nsight.py --image_path ./notebook/images/dancing.jpg --detector yolo --detector_model ./checkpoints/yolo/yolo11m.engine

# Or specify a custom engine path
USE_TRT_BACKBONE=1 TRT_BACKBONE_PATH=/path/to/backbone.engine python demo_human.py ...


"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)


# Default paths
CHECKPOINT_DIR = os.path.join(parent_dir, "checkpoints", "sam-3d-body-dinov3")
TRT_OUTPUT_DIR = os.path.join(CHECKPOINT_DIR, "backbone_trt")
ONNX_PATH = os.path.join(TRT_OUTPUT_DIR, "backbone_dinov3.onnx")
TRT_PATH_BF16 = os.path.join(TRT_OUTPUT_DIR, "backbone_dinov3_bf16.engine")
TRT_PATH_FP16 = os.path.join(TRT_OUTPUT_DIR, "backbone_dinov3_fp16.engine")
TRT_PATH = TRT_PATH_FP16  # Default to FP16 for better TensorRT optimization

# Model config
IMAGE_SIZE = (512, 512)  # H, W
EMBED_DIM = 1280  # dinov3_vith16plus
PATCH_SIZE = 16
OUTPUT_SIZE = (32, 32)  # 512 / 16 = 32


class BackboneWrapper(nn.Module):
    """
    Wrapper for DINOv3 backbone that exposes a simple forward interface.
    This wraps the get_intermediate_layers call for ONNX export.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] normalized RGB image
        Returns:
            y: [B, C, H', W'] feature map
        """
        # get_intermediate_layers returns a list of features
        # We take the last one with reshape=True, norm=True
        y = self.encoder.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]
        return y


def load_backbone():
    """Load the DINOv3 backbone from checkpoint."""
    from sam_3d_body.build_models import load_sam_3d_body

    # Load model using checkpoint path
    ckpt_path = os.path.join(CHECKPOINT_DIR, "model.ckpt")
    mhr_path = os.path.join(CHECKPOINT_DIR, "assets", "mhr_model.pt")

    model, cfg = load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path)

    # Extract backbone
    backbone = model.backbone
    backbone.eval()

    print(f"Backbone type: {cfg.MODEL.BACKBONE.TYPE}")
    print(f"Embed dim: {backbone.embed_dim}")
    print(f"Patch size: {backbone.patch_size}")

    return backbone


def step1_export_onnx(backbone, batch_sizes=[1, 2, 4]):
    """Export backbone to ONNX with dynamic batch size."""
    print("=" * 60)
    print("Step 1: Export Backbone to ONNX")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(TRT_OUTPUT_DIR, exist_ok=True)
    print(f"  Output directory: {TRT_OUTPUT_DIR}")

    # Create wrapper and convert to FP32 for ONNX export
    # (TensorRT will handle FP16 conversion during engine build)
    wrapper = BackboneWrapper(backbone.encoder)
    wrapper.eval()
    wrapper.float()  # Convert to FP32
    wrapper.cuda()

    # Test input (FP32)
    dummy_input = torch.randn(1, 3, *IMAGE_SIZE, device="cuda", dtype=torch.float32)

    # Verify output
    with torch.no_grad():
        output = wrapper(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    # Export to ONNX with dynamic batch
    print("  Exporting to ONNX...")

    # Dynamic axes for batch dimension
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }

    torch.onnx.export(
        wrapper,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    print(f"  [SUCCESS] Saved to: {ONNX_PATH}")
    print(f"  File size: {os.path.getsize(ONNX_PATH) / 1024 / 1024:.1f} MB")

    # Verify ONNX
    try:
        import onnx
        model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(model)
        print("  ONNX model verified!")
    except Exception as e:
        print(f"  Warning: ONNX verification failed: {e}")

    return True


def step2_convert_tensorrt(batch_sizes=[1, 2, 4]):
    """Convert ONNX to TensorRT with FP16."""
    print("\n" + "=" * 60)
    print("Step 2: Convert to TensorRT (FP16)")
    print("=" * 60)

    try:
        import tensorrt as trt
    except ImportError:
        print("  [ERROR] TensorRT not installed")
        print("  Install with: pip install tensorrt")
        return False

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX (use parse_from_file for external data support)
    print("  Parsing ONNX...")
    print(f"  ONNX file: {ONNX_PATH}")

    # Use parse_from_file to handle ONNX with external data
    if not parser.parse_from_file(ONNX_PATH):
        for i in range(parser.num_errors):
            print(f"  [ERROR] {parser.get_error(i)}")
        return False

    # Print network info
    print(f"  Network inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"    {inp.name}: {inp.shape}")
    print(f"  Network outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"    {out.name}: {out.shape}")

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    # Use FP16 precision for internal compute and I/O
    # (FP16 is better optimized in TensorRT than BF16)
    config.set_flag(trt.BuilderFlag.FP16)

    # Set input/output layers to use FP16
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        inp.dtype = trt.float16
    for i in range(network.num_outputs):
        out = network.get_output(i)
        out.dtype = trt.float16
    print("  Using FP16 precision (compute + I/O)")

    # Optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()

    # Set min/opt/max shapes for batch dimension
    min_batch = min(batch_sizes)
    opt_batch = batch_sizes[len(batch_sizes) // 2] if len(batch_sizes) > 1 else batch_sizes[0]
    max_batch = max(batch_sizes)

    profile.set_shape(
        "input",
        (min_batch, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]),  # min
        (opt_batch, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]),  # opt
        (max_batch, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]),  # max
    )
    config.add_optimization_profile(profile)

    print(f"  Batch size range: [{min_batch}, {opt_batch}, {max_batch}]")

    # Build engine
    print("  Building TensorRT engine (this may take several minutes)...")
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        print("  [ERROR] Engine build failed")
        return False

    with open(TRT_PATH, "wb") as f:
        f.write(engine)

    print(f"  [SUCCESS] Saved to: {TRT_PATH}")
    print(f"  File size: {os.path.getsize(TRT_PATH) / 1024 / 1024:.1f} MB")
    return True


class TRTBackbone:
    """TensorRT inference wrapper for backbone."""

    def __init__(self, engine_path):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Get binding info
        self.input_name = "input"
        self.output_name = "output"

    def __call__(self, x):
        """
        Run TensorRT inference.
        Args:
            x: [B, 3, H, W] input tensor (BF16 or FP32)
        Returns:
            output: [B, C, H', W'] feature map (BF16)
        """
        batch_size = x.shape[0]

        # Set input shape for dynamic batch
        self.context.set_input_shape(self.input_name, x.shape)

        # Allocate output buffer (FP16)
        output = torch.empty(
            batch_size, EMBED_DIM, OUTPUT_SIZE[0], OUTPUT_SIZE[1],
            device=x.device, dtype=torch.float16
        )

        # Set tensor addresses
        x_fp16 = x.half() if x.dtype != torch.float16 else x
        self.context.set_tensor_address(self.input_name, x_fp16.data_ptr())
        self.context.set_tensor_address(self.output_name, output.data_ptr())

        # Execute
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        return output


def step3_benchmark(backbone):
    """Benchmark PyTorch vs TensorRT."""
    print("\n" + "=" * 60)
    print("Step 3: Benchmark")
    print("=" * 60)

    # Test different batch sizes
    for batch_size in [1, 2]:
        print(f"\n  Batch size: {batch_size}")

        # Test input
        x = torch.randn(batch_size, 3, *IMAGE_SIZE, device="cuda", dtype=torch.float32)

        # PyTorch (BF16)
        print("  [PyTorch BF16]")
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = backbone(x)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(50):
                with torch.no_grad():
                    _ = backbone(x)
            torch.cuda.synchronize()
            pt_time = (time.perf_counter() - start) * 1000 / 50
            print(f"    Time: {pt_time:.3f} ms/call")

        # TensorRT
        if os.path.exists(TRT_PATH):
            print("  [TensorRT BF16]")
            trt_backbone = TRTBackbone(TRT_PATH)

            # Warmup
            for _ in range(5):
                _ = trt_backbone(x)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(50):
                _ = trt_backbone(x)
            torch.cuda.synchronize()
            trt_time = (time.perf_counter() - start) * 1000 / 50
            print(f"    Time: {trt_time:.3f} ms/call")
            print(f"    Speedup: {pt_time / trt_time:.2f}x")

            # Verify output
            with torch.cuda.amp.autocast(dtype=torch.float16):
                with torch.no_grad():
                    pt_out = backbone(x)
            trt_out = trt_backbone(x)

            diff = (pt_out.float() - trt_out.float()).abs()
            print(f"    Max diff: {diff.max().item():.6f}")
            print(f"    Mean diff: {diff.mean().item():.6f}")
        else:
            print("  [TensorRT] Engine not found, skipping...")


def main():
    parser = argparse.ArgumentParser(description="Convert DINOv3 Backbone to TensorRT")
    parser.add_argument("--export_onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--convert_trt", action="store_true", help="Convert ONNX to TensorRT")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark PyTorch vs TensorRT")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4", help="Batch sizes for optimization")

    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    if args.all:
        args.export_onnx = True
        args.convert_trt = True
        args.benchmark = True

    if not any([args.export_onnx, args.convert_trt, args.benchmark]):
        parser.print_help()
        return

    backbone = None

    if args.export_onnx or args.benchmark:
        print("Loading backbone...")
        backbone = load_backbone()

    if args.export_onnx:
        step1_export_onnx(backbone, batch_sizes)

    if args.convert_trt:
        step2_convert_tensorrt(batch_sizes)

    if args.benchmark:
        step3_benchmark(backbone)


if __name__ == "__main__":
    main()
