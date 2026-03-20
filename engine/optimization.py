"""
Performance Optimization Utilities for CLIP Few-Shot Classification

This module provides tools for optimizing the performance of the classification system,
including model quantization, batch processing optimization, and memory management.

Key Features:
- Model quantization (FP16, INT8)
- Batch size optimization
- Memory usage monitoring
- Performance benchmarking
- Cache management
"""

import time
import psutil
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
import numpy as np

from ..core.config import Config
from ..features.feature_extractor import MultiScaleFeatureExtractor
from .classifier import FineGrainedClassifier
from ..core.utils import Timer, ensure_dir


class ModelOptimizer:
    """Optimizer for CLIP model performance"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def quantize_model(
        self, model: nn.Module, quantization_type: str = "fp16"
    ) -> nn.Module:
        """
        Quantize model for faster inference

        Args:
            model: PyTorch model to quantize
            quantization_type: Type of quantization ("fp16", "int8")

        Returns:
            Quantized model
        """
        if quantization_type == "fp16":
            if torch.cuda.is_available():
                model = model.half()
                self.logger.info("Model quantized to FP16")
            else:
                self.logger.warning("FP16 quantization requires CUDA")

        elif quantization_type == "int8":
            if hasattr(torch.quantization, "quantize_dynamic"):
                model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
                self.logger.info("Model quantized to INT8")
            else:
                self.logger.warning("INT8 quantization not available")

        return model

    def optimize_batch_size(
        self,
        classifier: FineGrainedClassifier,
        test_images: List[str],
        max_batch_size: int = 128,
    ) -> int:
        """
        Find optimal batch size for the system

        Args:
            classifier: Classifier instance
            test_images: List of test image paths
            max_batch_size: Maximum batch size to test

        Returns:
            Optimal batch size
        """
        self.logger.info("Optimizing batch size...")

        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        if max_batch_size > 64:
            batch_sizes.extend([128, 256])

        batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]

        results = {}
        test_subset = test_images[: min(32, len(test_images))]  # Use subset for testing

        for batch_size in batch_sizes:
            try:
                # Measure processing time
                start_time = time.time()

                # Clear cache before each test
                if hasattr(classifier, "clear_caches"):
                    classifier.clear_caches()

                # Process batch
                _ = classifier.classify_batch(
                    test_subset, batch_size=batch_size, return_scores=False
                )

                end_time = time.time()
                processing_time = end_time - start_time
                throughput = len(test_subset) / processing_time

                # Monitor memory usage
                memory_usage = self._get_memory_usage()

                results[batch_size] = {
                    "processing_time": processing_time,
                    "throughput": throughput,
                    "memory_usage": memory_usage,
                }

                self.logger.info(
                    f"Batch size {batch_size}: "
                    f"{throughput:.2f} images/sec, "
                    f"{memory_usage['gpu_memory']:.1f}GB GPU memory"
                )

            except Exception as e:
                self.logger.warning(f"Batch size {batch_size} failed: {e}")
                continue

        # Find optimal batch size (highest throughput within memory limits)
        if not results:
            return 1

        # Sort by throughput
        sorted_results = sorted(
            results.items(), key=lambda x: x[1]["throughput"], reverse=True
        )

        optimal_batch_size = sorted_results[0][0]
        self.logger.info(f"Optimal batch size: {optimal_batch_size}")

        return optimal_batch_size

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = {}

        # CPU memory
        cpu_memory = psutil.virtual_memory()
        memory_info["cpu_memory"] = cpu_memory.used / (1024**3)  # GB
        memory_info["cpu_memory_percent"] = cpu_memory.percent

        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)  # GB
            memory_info["gpu_memory"] = gpu_memory
            memory_info["gpu_memory_cached"] = gpu_memory_cached
        else:
            memory_info["gpu_memory"] = 0.0
            memory_info["gpu_memory_cached"] = 0.0

        return memory_info


class PerformanceBenchmark:
    """Performance benchmarking utilities"""

    def __init__(self, classifier: FineGrainedClassifier):
        self.classifier = classifier
        self.logger = logging.getLogger(__name__)

    def benchmark_classification(
        self, test_images: List[str], num_runs: int = 3, warmup_runs: int = 1
    ) -> Dict[str, Any]:
        """
        Benchmark classification performance

        Args:
            test_images: List of test image paths
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs

        Returns:
            Benchmark results
        """
        self.logger.info(f"Benchmarking classification on {len(test_images)} images")

        # Warmup runs
        for i in range(warmup_runs):
            self.logger.info(f"Warmup run {i + 1}/{warmup_runs}")
            _ = self.classifier.classify_batch(test_images[:5], return_scores=False)

        # Benchmark runs
        run_times = []
        memory_usage = []

        for run in range(num_runs):
            self.logger.info(f"Benchmark run {run + 1}/{num_runs}")

            # Clear caches
            if hasattr(self.classifier, "clear_caches"):
                self.classifier.clear_caches()

            # Measure performance
            start_time = time.time()
            start_memory = self._get_peak_memory()

            results = self.classifier.classify_batch(test_images, return_scores=True)

            end_time = time.time()
            end_memory = self._get_peak_memory()

            run_time = end_time - start_time
            memory_used = end_memory - start_memory

            run_times.append(run_time)
            memory_usage.append(memory_used)

            self.logger.info(
                f"Run {run + 1}: {run_time:.3f}s, "
                f"{len(test_images) / run_time:.2f} images/sec"
            )

        # Calculate statistics
        avg_time = np.mean(run_times)
        std_time = np.std(run_times)
        avg_throughput = len(test_images) / avg_time
        avg_memory = np.mean(memory_usage)

        benchmark_results = {
            "num_images": len(test_images),
            "num_runs": num_runs,
            "avg_time": avg_time,
            "std_time": std_time,
            "min_time": np.min(run_times),
            "max_time": np.max(run_times),
            "avg_throughput": avg_throughput,
            "avg_memory_usage": avg_memory,
            "run_times": run_times,
            "system_info": self._get_system_info(),
        }

        return benchmark_results

    def benchmark_feature_extraction(
        self, test_images: List[str], batch_sizes: List[int] = [1, 8, 16, 32]
    ) -> Dict[str, Any]:
        """Benchmark feature extraction performance"""
        self.logger.info("Benchmarking feature extraction")

        feature_extractor = self.classifier.feature_extractor
        results = {}

        for batch_size in batch_sizes:
            self.logger.info(f"Testing batch size: {batch_size}")

            # Select test images
            test_subset = test_images[: min(batch_size * 4, len(test_images))]

            try:
                start_time = time.time()

                # Extract features in batches
                for i in range(0, len(test_subset), batch_size):
                    batch = test_subset[i : i + batch_size]
                    for img_path in batch:
                        _ = feature_extractor.extract_features(img_path)

                end_time = time.time()

                processing_time = end_time - start_time
                throughput = len(test_subset) / processing_time

                results[batch_size] = {
                    "processing_time": processing_time,
                    "throughput": throughput,
                    "images_processed": len(test_subset),
                }

                self.logger.info(
                    f"Batch size {batch_size}: {throughput:.2f} images/sec"
                )

            except Exception as e:
                self.logger.warning(f"Batch size {batch_size} failed: {e}")
                continue

        return results

    def _get_peak_memory(self) -> float:
        """Get peak memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)  # GB
        else:
            return psutil.virtual_memory().used / (1024**3)  # GB

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
            "torch_version": torch.__version__,
        }

        if torch.cuda.is_available():
            info.update(
                {
                    "cuda_available": True,
                    "cuda_version": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(),
                    "gpu_memory": torch.cuda.get_device_properties(0).total_memory
                    / (1024**3),
                }
            )
        else:
            info["cuda_available"] = False

        return info


class CacheManager:
    """Cache management utilities"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        ensure_dir(self.cache_dir)
        self.logger = logging.getLogger(__name__)

    def clear_all_caches(self):
        """Clear all cache directories"""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            ensure_dir(self.cache_dir)
            self.logger.info("All caches cleared")

    def get_cache_size(self) -> Dict[str, float]:
        """Get cache size information"""
        cache_info = {}

        if self.cache_dir.exists():
            total_size = 0
            for file_path in self.cache_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

            cache_info["total_size_mb"] = total_size / (1024**2)
            cache_info["total_size_gb"] = total_size / (1024**3)

            # Count files by type
            file_counts = {}
            for file_path in self.cache_dir.rglob("*"):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    file_counts[ext] = file_counts.get(ext, 0) + 1

            cache_info["file_counts"] = file_counts
        else:
            cache_info = {"total_size_mb": 0, "total_size_gb": 0, "file_counts": {}}

        return cache_info

    def optimize_cache(self, max_size_gb: float = 5.0):
        """Optimize cache by removing old files if size exceeds limit"""
        cache_info = self.get_cache_size()

        if cache_info["total_size_gb"] > max_size_gb:
            self.logger.info(
                f"Cache size ({cache_info['total_size_gb']:.2f}GB) "
                f"exceeds limit ({max_size_gb}GB), cleaning up..."
            )

            # Get all files with modification times
            files_with_times = []
            for file_path in self.cache_dir.rglob("*"):
                if file_path.is_file():
                    mtime = file_path.stat().st_mtime
                    size = file_path.stat().st_size
                    files_with_times.append((file_path, mtime, size))

            # Sort by modification time (oldest first)
            files_with_times.sort(key=lambda x: x[1])

            # Remove files until under limit
            current_size = cache_info["total_size_gb"] * (1024**3)
            target_size = max_size_gb * 0.8 * (1024**3)  # Target 80% of limit

            removed_count = 0
            for file_path, mtime, size in files_with_times:
                if current_size <= target_size:
                    break

                try:
                    file_path.unlink()
                    current_size -= size
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to remove {file_path}: {e}")

            self.logger.info(f"Removed {removed_count} cache files")


def create_optimization_report(
    classifier: FineGrainedClassifier,
    test_images: List[str],
    output_path: str = "optimization_report.json",
):
    """Create comprehensive optimization report"""
    logger = logging.getLogger(__name__)
    logger.info("Creating optimization report...")

    # Initialize components
    config = classifier.config
    optimizer = ModelOptimizer(config)
    benchmark = PerformanceBenchmark(classifier)
    cache_manager = CacheManager()

    report = {
        "timestamp": time.time(),
        "system_info": benchmark._get_system_info(),
        "config": config.to_dict(),
        "cache_info": cache_manager.get_cache_size(),
    }

    # Run benchmarks
    try:
        # Classification benchmark
        classification_results = benchmark.benchmark_classification(
            test_images[:20], num_runs=3
        )
        report["classification_benchmark"] = classification_results

        # Feature extraction benchmark
        feature_results = benchmark.benchmark_feature_extraction(test_images[:16])
        report["feature_extraction_benchmark"] = feature_results

        # Batch size optimization
        optimal_batch_size = optimizer.optimize_batch_size(
            classifier, test_images[:32], max_batch_size=64
        )
        report["optimal_batch_size"] = optimal_batch_size

        # Memory usage analysis
        memory_usage = optimizer._get_memory_usage()
        report["memory_usage"] = memory_usage

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        report["benchmark_error"] = str(e)

    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Optimization report saved to: {output_path}")
    return report


# -------- Small Head (cosine/ArcFace/CosFace) lightweight training ---------
class CosineClassifierHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        head_type: str = "cosine",
        margin: float = 0.5,
        scale: float = 30.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.head_type = head_type.lower()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = nn.functional.normalize(x, dim=1)
        W = nn.functional.normalize(self.weight, dim=1)
        logits = x @ W.t()  # cosine similarities
        if labels is not None and self.head_type in ("arcface", "cosface"):
            # apply margin only for target class during training
            logits_target = logits[torch.arange(logits.size(0)), labels]
            if self.head_type == "arcface":
                logits_target = torch.clamp(logits_target, -1.0 + 1e-7, 1.0 - 1e-7)
                theta = torch.acos(logits_target)
                logits_target = torch.cos(theta + self.margin)
            else:  # cosface
                logits_target = logits_target - self.margin
            logits = logits.clone()
            logits[torch.arange(logits.size(0)), labels] = logits_target
        return logits * self.scale


def train_small_head(
    support_manager, config: Config, device: Optional[torch.device] = None
) -> Optional[Dict[str, Any]]:
    """Train a lightweight small head using adapter_support_bank.
    Returns dict with learned weights and meta; also updates support_manager.small_head.
    """
    logger = logging.getLogger(__name__)
    adapter_cfg = getattr(config, "adapter", None)
    if adapter_cfg is None or not getattr(adapter_cfg, "train_small_head", False):
        logger.info(
            "train_small_head skipped: config.adapter.train_small_head is False"
        )
        return None

    # Build dataset from support bank
    class_names: List[str] = support_manager.class_names
    X_list, y_list = [], []
    for idx, cname in enumerate(class_names):
        feats = support_manager.adapter_support_bank.get(cname)
        if feats is None:
            continue
        # feats: [num_shots, D]
        X_list.append(feats)
        y_list.append(torch.full((feats.shape[0],), idx, dtype=torch.long))
    if len(X_list) == 0:
        logger.warning("No features in adapter_support_bank; cannot train small head")
        return None

    X = torch.cat(X_list, dim=0).float()
    y = torch.cat(y_list, dim=0)

    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = nn.functional.normalize(X, dim=1)
    feat_dim = X.shape[1]

    head_type = getattr(adapter_cfg, "head_type", "arcface")
    margin = float(getattr(adapter_cfg, "margin", 0.5))
    scale = float(getattr(adapter_cfg, "scale", 30.0))
    epochs = int(getattr(adapter_cfg, "train_epochs", 2))
    lr = float(getattr(adapter_cfg, "lr", 1e-3))
    weight_decay = float(getattr(adapter_cfg, "weight_decay", 1e-4))
    batch_size = min(128, max(8, X.shape[0]))

    model = CosineClassifierHead(
        num_classes=len(class_names),
        feat_dim=feat_dim,
        head_type=head_type,
        margin=margin,
        scale=scale,
    ).to(device)

    # Initialize weights by class means for stability
    with torch.no_grad():
        W_init = torch.zeros(len(class_names), feat_dim, device=device)
        for idx, cname in enumerate(class_names):
            feats = support_manager.adapter_support_bank.get(cname)
            if feats is not None and feats.numel() > 0:
                mu = nn.functional.normalize(feats.float().mean(0), dim=0)
                W_init[idx] = mu
            else:
                W_init[idx] = nn.functional.normalize(
                    torch.randn(feat_dim, device=device), dim=0
                )
        model.weight.copy_(W_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    X_dev = X.to(device)
    y_dev = y.to(device)

    model.train()
    for ep in range(epochs):
        # simple shuffling
        perm = torch.randperm(X_dev.size(0), device=device)
        X_ep = X_dev[perm]
        y_ep = y_dev[perm]
        losses = []
        for i in range(0, X_ep.size(0), batch_size):
            xb = X_ep[i : i + batch_size]
            yb = y_ep[i : i + batch_size]
            logits = model(xb, yb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        logger.info(f"Small head epoch {ep + 1}/{epochs} loss={np.mean(losses):.4f}")

    # Save to support_manager
    with torch.no_grad():
        W = nn.functional.normalize(model.weight, dim=1).detach().cpu().numpy()
    small_head = {"type": head_type, "scale": scale, "margin": margin, "W": W}
    support_manager.small_head = small_head
    logger.info("Small head trained and attached to support_manager.small_head")
    return small_head
