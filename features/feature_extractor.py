"""Feature extraction module providing multi-scale CLIP feature extraction.

This module provides a unified interface for creating feature extractors
that support multi-scale feature extraction from CLIP models, with
automatic device resolution and caching capabilities.
"""

import os
import importlib

from ..core.config import Config
from ..modalities.image import CLIPImageEncoderAdapter

torch = importlib.import_module("torch")


class MultiScaleFeatureExtractor(CLIPImageEncoderAdapter):
    """Multi-scale feature extractor extending CLIP image encoder.

    This class inherits from CLIPImageEncoderAdapter to provide multi-scale
    feature extraction capabilities. Currently acts as a thin wrapper around
    the base adapter, with potential for future multi-scale enhancements.

    Args:
        model_name: CLIP model identifier (e.g., 'ViT-B/32')
        device: Computation device ('cuda', 'cpu', or 'auto')
        cache_features: Whether to enable feature caching
        cache_backend: Cache storage backend ('disk', 'mmap', or 'memory')
        cache_dir: Directory path for disk-based caching
    """

    pass


def create_feature_extractor(config: Config) -> MultiScaleFeatureExtractor:
    """Create a configured MultiScaleFeatureExtractor instance.

    Factory function that creates a feature extractor with automatic device
    resolution (respecting NC_FORCE_CPU environment variable) and configured
    caching backend.

    Args:
        config: System configuration containing model and caching settings.
            Uses config.model.clip_model_name, device, cache_features,
            cache_backend, and cache_dir.

    Returns:
        Configured MultiScaleFeatureExtractor instance ready for feature extraction.

    Note:
        Sets config.model.device to the resolved device for downstream consistency.
    """
    try:
        if str(os.environ.get("NC_FORCE_CPU", "")).strip() == "1":
            resolved_device = "cpu"
        else:
            device = str(getattr(config.model, "device", "auto")).lower()
            if device == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device.startswith("cuda") and not torch.cuda.is_available():
                resolved_device = "cpu"
            else:
                resolved_device = getattr(config.model, "device", "cpu")
        config.model.device = resolved_device
    except Exception:
        resolved_device = getattr(config.model, "device", "cpu")

    return MultiScaleFeatureExtractor(
        model_name=config.model.clip_model_name,
        device=resolved_device,
        cache_features=getattr(config.model, "cache_features", True),
        cache_backend=getattr(config.model, "cache_backend", "disk"),
        cache_dir=getattr(config.model, "cache_dir", "cache/clip_features"),
    )


__all__ = ["MultiScaleFeatureExtractor", "create_feature_extractor"]
