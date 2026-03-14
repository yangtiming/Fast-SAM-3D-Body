# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Simple timing utility for profiling Attention vs FFN time in Transformers.
"""

import torch
from collections import defaultdict


class TransformerTiming:
    """Global timing accumulator for Transformer components."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.reset()
        self.enabled = False

    def reset(self):
        """Reset all timing accumulators."""
        # Overall statistics
        self.attention_time_ms = 0.0
        self.ffn_time_ms = 0.0
        self.self_attn_time_ms = 0.0
        self.cross_attn_time_ms = 0.0
        self.other_time_ms = 0.0
        self.call_counts = defaultdict(int)

        # Per-component statistics
        self.component_timing = defaultdict(lambda: {
            "attention_time_ms": 0.0,
            "ffn_time_ms": 0.0,
            "self_attn_time_ms": 0.0,
            "cross_attn_time_ms": 0.0,
            "call_counts": defaultdict(int),
        })

        # Other module timing statistics
        self.module_timing = defaultdict(lambda: {
            "total_time_ms": 0.0,
            "call_count": 0,
        })

    def enable(self):
        """Enable timing collection."""
        self.enabled = True
        self.reset()

    def disable(self):
        """Disable timing collection."""
        self.enabled = False

    def add_attention_time(self, time_ms: float, attn_type: str = "self", component: str = "unknown"):
        """Add attention time."""
        if not self.enabled:
            return
        self.attention_time_ms += time_ms
        if attn_type == "self":
            self.self_attn_time_ms += time_ms
        elif attn_type == "cross":
            self.cross_attn_time_ms += time_ms
        self.call_counts[f"attn_{attn_type}"] += 1

        # Per-component statistics
        comp = self.component_timing[component]
        comp["attention_time_ms"] += time_ms
        if attn_type == "self":
            comp["self_attn_time_ms"] += time_ms
        elif attn_type == "cross":
            comp["cross_attn_time_ms"] += time_ms
        comp["call_counts"][f"attn_{attn_type}"] += 1

    def add_ffn_time(self, time_ms: float, component: str = "unknown"):
        """Add FFN time."""
        if not self.enabled:
            return
        self.ffn_time_ms += time_ms
        self.call_counts["ffn"] += 1

        # Per-component statistics
        comp = self.component_timing[component]
        comp["ffn_time_ms"] += time_ms
        comp["call_counts"]["ffn"] += 1

    def add_other_time(self, time_ms: float, name: str = "other"):
        """Add other time."""
        if not self.enabled:
            return
        self.other_time_ms += time_ms
        self.call_counts[name] += 1

    def add_module_time(self, time_ms: float, module_name: str):
        """Add time for a module (e.g., human_detector, fov_estimator)."""
        if not self.enabled:
            return
        mod = self.module_timing[module_name]
        mod["total_time_ms"] += time_ms
        mod["call_count"] += 1

    def get_summary(self) -> dict:
        """Get timing summary."""
        total = self.attention_time_ms + self.ffn_time_ms
        return {
            "attention_time_ms": self.attention_time_ms,
            "ffn_time_ms": self.ffn_time_ms,
            "self_attn_time_ms": self.self_attn_time_ms,
            "cross_attn_time_ms": self.cross_attn_time_ms,
            "other_time_ms": self.other_time_ms,
            "total_attn_ffn_ms": total,
            "attention_percentage": 100 * self.attention_time_ms / total if total > 0 else 0,
            "ffn_percentage": 100 * self.ffn_time_ms / total if total > 0 else 0,
            "call_counts": dict(self.call_counts),
            "by_component": {k: dict(v) for k, v in self.component_timing.items()},
            "by_module": {k: dict(v) for k, v in self.module_timing.items()},
        }

    def print_summary(self):
        """Print timing summary."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("Transformer Timing Summary (Attention vs FFN)")
        print("=" * 60)
        print(f"\nAttention total: {summary['attention_time_ms']:.2f} ms ({summary['attention_percentage']:.1f}%)")
        print(f"  - Self-attention: {summary['self_attn_time_ms']:.2f} ms")
        print(f"  - Cross-attention: {summary['cross_attn_time_ms']:.2f} ms")
        print(f"\nFFN total: {summary['ffn_time_ms']:.2f} ms ({summary['ffn_percentage']:.1f}%)")
        print(f"\nOther: {summary['other_time_ms']:.2f} ms")
        print(f"\nCall counts: {dict(summary['call_counts'])}")

        if summary['attention_time_ms'] > 0:
            ratio = summary['ffn_time_ms'] / summary['attention_time_ms']
            print(f"\nFFN / Attention ratio: {ratio:.2f}x")
        print("=" * 60)


# Global instance
_timing = TransformerTiming()


def get_timing() -> TransformerTiming:
    """Get the global timing instance."""
    return _timing


def time_cuda_op(func, *args, **kwargs):
    """
    Time a CUDA operation with proper synchronization.
    Returns (result, time_ms).
    """
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    result = func(*args, **kwargs)
    end.record()

    torch.cuda.synchronize()
    time_ms = start.elapsed_time(end)

    return result, time_ms
