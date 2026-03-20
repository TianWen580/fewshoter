"""
Multi-scale Feature Extraction Engine for CLIP Few-Shot Classification

This module implements the MultiScaleFeatureExtractor class that extracts
hierarchical features from CLIP models, including global features, local patch features,
and attention maps for fine-grained classification tasks.

Key Features:
- Multi-scale feature pyramid extraction
- Attention map computation from transformer layers
- Support for different CLIP model architectures
- Efficient batch processing and caching
- Feature normalization and preprocessing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip as clip
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import os

from ..core.config import Config
from ..core.utils import load_image, normalize_tensor, Timer


class DiskFeatureCache:
    """Simple disk-backed feature cache using torch.save/torch.load.
    简单的磁盘特征缓存，通过 torch.save/torch.load 实现。
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _path(self, key: str) -> Path:
        safe = str(key).replace("/", "_").replace(":", "_")
        return self.root / f"{safe}.pt"

    def get(self, key: str):
        p = self._path(key)
        if p.exists():
            try:
                return torch.load(p, map_location="cpu")
            except Exception:
                return None
        return None

    def set(self, key: str, data: Dict[str, torch.Tensor]) -> None:
        p = self._path(key)
        try:
            torch.save(data, p)
        except Exception:
            pass


class MmapFeatureCache:
    """True memory-mapped feature cache using NumPy .npy with mmap_mode.
    真实的内存映射缓存：每个特征张量单独保存为 .npy，并以 mmap 方式只读加载。
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _key_dir(self, key: str) -> Path:
        safe = str(key).replace("/", "_").replace(":", "_")
        return self.root / safe

    def get(self, key: str):
        d = self._key_dir(key)
        if not d.exists() or not d.is_dir():
            return None
        result: Dict[str, torch.Tensor] = {}
        try:
            for npy_path in sorted(d.glob("*.npy")):
                name = npy_path.stem
                arr = np.load(npy_path, mmap_mode="r")  # memmap array
                # Important: avoid returning tensors that share storage with np.memmap to prevent
                # lifetime and non-writable issues that can lead to segfaults. Copy into a fresh tensor.
                t = torch.tensor(np.asarray(arr))
                result[name] = t
        except Exception:
            return None
        return result if result else None

    def set(self, key: str, data: Dict[str, torch.Tensor]) -> None:
        d = self._key_dir(key)
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        for name, tensor in (data or {}).items():
            p = d / f"{name}.npy"
            try:
                np.save(p, tensor.detach().cpu().numpy())
            except Exception:
                pass


class MultiScaleFeatureExtractor:
    """
    Multi-scale feature extractor for CLIP/OpenCLIP models

    Extracts features at different scales:
    - Global: Full image representation from final layer
    - Local: Patch-level features from intermediate layers
    - Attention: Attention maps from transformer blocks
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        cache_features: bool = True,
        cache_backend: str = "mmap",
        cache_dir: Union[str, Path] = "cache/clip_features",
    ):
        """
        Initialize the feature extractor

        Args:
            model_name: CLIP model name (e.g., "ViT-B/32", "ViT-B/16")
            device: Device to run the model on
            cache_features: Whether to cache extracted features
            cache_backend: Cache backend ('memory' | 'disk' | 'mmap')
            cache_dir: Directory for on-disk feature cache
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.cache_features = cache_features
        self.cache_backend = (cache_backend or "disk").lower()
        self.cache_dir = Path(cache_dir)
        if self.cache_features:
            if self.cache_backend in ("disk", "mmap"):
                try:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                if self.cache_backend == "mmap":
                    self.feature_cache = MmapFeatureCache(self.cache_dir)
                else:
                    self.feature_cache = DiskFeatureCache(self.cache_dir)
            else:
                self.feature_cache = {}
        else:
            self.feature_cache = None

        # Load CLIP/OpenCLIP model
        # Map friendly names to open_clip model/pretrained tags
        def _resolve_openclip(model_name: str):
            name = model_name
            pretrained = None
            # Common mappings
            mappings = {
                "ViT-B/32": ("ViT-B-32", "openai"),
                "ViT-B/16": ("ViT-B-16", "openai"),
                "ViT-L/14": ("ViT-L-14", "openai"),
                "ViT-L/14@336px": ("ViT-L-14@336px", "openai"),
                "RN50": ("RN50", "openai"),
                "RN101": ("RN101", "openai"),
                "RN50x4": ("RN50x4", "openai"),
                "RN50x16": ("RN50x16", "openai"),
                "RN50x64": ("RN50x64", "openai"),
                # OpenCLIP defaults
                "ViT-H-14-quickgelu": ("ViT-H-14-quickgelu", "dfn5b"),
                "ViT-L-14-quickgelu": ("ViT-L-14-quickgelu", "dfn2b"),
                "ViT-H-14-laion2B": (
                    "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                    None,
                ),
                "ViT-B-32-laion2B": (
                    "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                    None,
                ),
            }
            if name in mappings:
                return mappings[name]
            # Best-effort normalization for other names
            norm = name.replace("/", "-")
            return norm, pretrained or "laion2b_s32b_b79k"

        arch, pretrained = _resolve_openclip(self.model_name)
        try:
            model, preprocess_train, preprocess_val = clip.create_model_and_transforms(
                arch, pretrained=pretrained, device=self.device.type
            )
        except TypeError:
            # Fallback for open_clip versions without device kwarg
            model, preprocess_train, preprocess_val = clip.create_model_and_transforms(
                arch, pretrained=pretrained
            )
            model = model.to(self.device)
        self.model = model.eval()
        # Prefer validation preprocessing for inference
        self.preprocess = preprocess_val or preprocess_train

        # Get model architecture info
        self._setup_model_info()

        # Setup hooks for intermediate features
        self.intermediate_features = {}
        self.attention_maps = {}
        self._register_hooks()

        self.logger = logging.getLogger(__name__)

    def _setup_model_info(self):
        """Setup model architecture information"""
        if hasattr(self.model, "visual") and hasattr(self.model.visual, "transformer"):
            # Vision Transformer
            self.is_vit = True
            # Feature dimension
            if hasattr(self.model.visual, "width"):
                self.feature_dim = self.model.visual.width
            elif hasattr(self.model.visual, "transformer") and hasattr(
                self.model.visual.transformer, "width"
            ):
                self.feature_dim = self.model.visual.transformer.width
            elif hasattr(self.model.visual, "embed_dim"):
                self.feature_dim = self.model.visual.embed_dim
            else:
                self.feature_dim = 1024
            # Number of layers
            if hasattr(self.model.visual.transformer, "resblocks"):
                self.num_layers = len(self.model.visual.transformer.resblocks)
            elif hasattr(self.model.visual.transformer, "blocks"):
                self.num_layers = len(self.model.visual.transformer.blocks)
            else:
                self.num_layers = 12
            # Grid size from positional embeddings if available
            grid_size = None
            if hasattr(self.model.visual, "positional_embedding"):
                try:
                    grid_size = int(
                        (self.model.visual.positional_embedding.shape[1] - 1) ** 0.5
                    )
                except Exception:
                    grid_size = None
            if grid_size is None and hasattr(self.model.visual, "pos_embed"):
                try:
                    grid_size = int((self.model.visual.pos_embed.shape[1] - 1) ** 0.5)
                except Exception:
                    grid_size = None
            if grid_size is None:
                # Fallback by patch size hint in model name
                if any(s in self.model_name for s in ["/32", "-32"]):
                    grid_size = 7
                elif any(s in self.model_name for s in ["/16", "-16", "-14", "/14"]):
                    # 224 / 16 = 14, 224 / 14 = 16; pick 14 as conservative default
                    grid_size = 14
                else:
                    grid_size = 14
            self.grid_size = grid_size
        else:
            # ResNet-style encoder
            self.is_vit = False
            try:
                self.feature_dim = (
                    self.model.visual.attnpool.positional_embedding.shape[-1]
                )
            except Exception:
                self.feature_dim = 1024
            self.num_layers = 4  # ResNet has 4 main blocks
            self.grid_size = 7  # ResNet feature map size

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features"""
        if self.is_vit:
            # Register hooks for Vision Transformer layers
            target_layers = [3, 6, 9] if self.num_layers >= 12 else [1, 2, 3]

            for i, layer_idx in enumerate(target_layers):
                if layer_idx < self.num_layers:
                    if hasattr(self.model.visual.transformer, "resblocks"):
                        layer = self.model.visual.transformer.resblocks[layer_idx]
                    else:
                        layer = self.model.visual.transformer.blocks[layer_idx]
                    layer.register_forward_hook(self._make_hook(f"layer_{layer_idx}"))

                    # Register attention hook if available
                    if hasattr(layer, "attn"):
                        layer.attn.register_forward_hook(
                            self._make_attention_hook(f"attn_{layer_idx}")
                        )
        else:
            # Register hooks for ResNet layers
            layers = [
                getattr(self.model.visual, "layer1", None),
                getattr(self.model.visual, "layer2", None),
                getattr(self.model.visual, "layer3", None),
                getattr(self.model.visual, "layer4", None),
            ]

            for i, layer in enumerate([l for l in layers if l is not None]):
                layer.register_forward_hook(self._make_hook(f"resnet_layer_{i + 1}"))

    def _make_hook(self, name: str):
        """Create a forward hook function"""

        def hook(module, input, output):
            self.intermediate_features[name] = output.detach()

        return hook

    def _make_attention_hook(self, name: str):
        """Create an attention hook function"""

        def hook(module, input, output):
            # For ViT, attention weights are in the attention module
            if hasattr(module, "attention_weights"):
                self.attention_maps[name] = module.attention_weights.detach()

        return hook

    def extract_features(
        self,
        image_input: Union[str, Path, Image.Image, torch.Tensor],
        return_attention: bool = True,
        normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from input image

        Args:
            image_input: Input image (path, PIL Image, or tensor)
            return_attention: Whether to return attention maps
            normalize: Whether to normalize features

        Returns:
            Dictionary containing extracted features at different scales
        """
        # Check cache first
        cache_key = None
        if self.cache_features and isinstance(image_input, (str, Path)):
            cache_key = str(image_input)
            if self.feature_cache:
                try:
                    if hasattr(self.feature_cache, "get"):
                        cached = self.feature_cache.get(cache_key)
                        if cached is not None:
                            return cached
                    elif (
                        isinstance(self.feature_cache, dict)
                        and cache_key in self.feature_cache
                    ):
                        return self.feature_cache[cache_key]
                except Exception:
                    pass

        # Preprocess image
        if isinstance(image_input, (str, Path)):
            image = load_image(image_input)
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        elif isinstance(image_input, Image.Image):
            image_tensor = self.preprocess(image_input).unsqueeze(0).to(self.device)
        elif isinstance(image_input, torch.Tensor):
            image_tensor = image_input.to(self.device)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Clear previous intermediate features
        self.intermediate_features.clear()
        self.attention_maps.clear()

        # Extract features
        with torch.no_grad():
            # Get global features
            global_features = self.model.encode_image(image_tensor)

            if normalize:
                global_features = normalize_tensor(global_features)

        # Prepare output dictionary
        features = {
            "global": global_features,
            "image_size": image_tensor.shape[-2:],
            "grid_size": self.grid_size,
        }

        # Add intermediate features
        for name, feat in self.intermediate_features.items():
            if self.is_vit:
                # For ViT, remove CLS token and reshape to spatial format
                if feat.dim() == 3 and feat.shape[1] > self.grid_size * self.grid_size:
                    spatial_feat = feat[:, 1:, :]  # Remove CLS token
                    batch_size, seq_len, feat_dim = spatial_feat.shape
                    grid_size = int(seq_len**0.5)
                    spatial_feat = spatial_feat.reshape(
                        batch_size, grid_size, grid_size, feat_dim
                    )
                    spatial_feat = spatial_feat.permute(0, 3, 1, 2)  # [B, C, H, W]
                    features[name] = spatial_feat
                else:
                    features[name] = feat
            else:
                # For ResNet, features are already in spatial format
                features[name] = feat

        # Add attention maps if requested
        if return_attention and self.is_vit:
            attention_features = self._process_attention_maps()
            features.update(attention_features)

        # Normalize all features if requested
        if normalize:
            for key, feat in features.items():
                if isinstance(feat, torch.Tensor) and feat.dim() >= 2:
                    if key not in ["image_size", "grid_size"]:
                        features[key] = normalize_tensor(feat, dim=-1)

        # Cache results
        if cache_key and self.cache_features and self.feature_cache:
            try:
                if hasattr(self.feature_cache, "set"):
                    self.feature_cache.set(cache_key, features)
                else:
                    self.feature_cache[cache_key] = features
            except Exception:
                pass

        return features

    def _process_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Process attention maps from transformer layers"""
        attention_features = {}

        for name, attn_map in self.attention_maps.items():
            if attn_map is not None:
                # Average over attention heads
                if attn_map.dim() == 4:  # [batch, heads, seq, seq]
                    attn_map = attn_map.mean(dim=1)  # Average over heads

                # Extract CLS token attention to patches
                if attn_map.shape[-1] > self.grid_size * self.grid_size:
                    cls_attention = attn_map[:, 0, 1:]  # CLS to patches

                    # Reshape to spatial format
                    batch_size = cls_attention.shape[0]
                    cls_attention = cls_attention.reshape(
                        batch_size, self.grid_size, self.grid_size
                    )

                    attention_features[name] = cls_attention

        return attention_features

    def extract_batch_features(
        self, image_paths: List[Union[str, Path]], batch_size: int = 32, **kwargs
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Extract features from a batch of images

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            **kwargs: Additional arguments for extract_features

        Returns:
            List of feature dictionaries
        """
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]

            with Timer(f"Processing batch {i // batch_size + 1}"):
                batch_results = []
                for path in batch_paths:
                    features = self.extract_features(path, **kwargs)
                    batch_results.append(features)

                results.extend(batch_results)

        return results

    def get_feature_dimensions(self) -> Dict[str, Tuple[int, ...]]:
        """Get feature dimensions for each scale"""
        dimensions = {"global": (self.feature_dim,), "grid_size": self.grid_size}

        # Add intermediate layer dimensions
        if self.is_vit:
            dimensions.update(
                {
                    "layer_3": (self.feature_dim, self.grid_size, self.grid_size),
                    "layer_6": (self.feature_dim, self.grid_size, self.grid_size),
                    "layer_9": (self.feature_dim, self.grid_size, self.grid_size),
                }
            )
        else:
            # ResNet dimensions vary by layer
            dimensions.update(
                {
                    "resnet_layer_1": (256, 56, 56),
                    "resnet_layer_2": (512, 28, 28),
                    "resnet_layer_3": (1024, 14, 14),
                    "resnet_layer_4": (2048, 7, 7),
                }
            )

        return dimensions

    def clear_cache(self):
        """Clear feature cache"""
        if self.feature_cache:
            self.feature_cache.clear()
            self.logger.info("Feature cache cleared")

    def get_cache_size(self) -> int:
        """Get number of cached features"""
        return len(self.feature_cache) if self.feature_cache else 0

    def __repr__(self):
        return (
            f"MultiScaleFeatureExtractor("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"cache_size={self.get_cache_size()})"
        )


def create_feature_extractor(config: Config) -> MultiScaleFeatureExtractor:
    """Factory function to create feature extractor from config with safe device fallback"""
    # Resolve device safely: honor 'auto' and fall back to CPU if CUDA is unavailable
    try:
        # Env override to force CPU when CUDA env is unstable
        if str(os.environ.get("NC_FORCE_CPU", "")).strip() == "1":
            resolved_device = "cpu"
        else:
            dev = str(getattr(config.model, "device", "auto")).lower()
            if dev == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            elif dev.startswith("cuda") and not torch.cuda.is_available():
                resolved_device = "cpu"
            else:
                resolved_device = getattr(config.model, "device", "cpu")
        # Persist the resolved device back to config to keep downstream consistent
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
