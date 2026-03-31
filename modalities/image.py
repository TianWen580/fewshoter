import logging
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image

from ..core.utils import load_image, normalize_tensor, Timer

np = importlib.import_module("numpy")
clip = importlib.import_module("open_clip")
torch = importlib.import_module("torch")
nn = importlib.import_module("torch.nn")


class DiskFeatureCache:
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

    def set(self, key: str, data: Dict[str, Any]) -> None:
        p = self._path(key)
        try:
            torch.save(data, p)
        except Exception:
            pass

    def clear(self) -> None:
        try:
            for p in self.root.glob("*.pt"):
                p.unlink(missing_ok=True)
        except Exception:
            pass


class MmapFeatureCache:
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
        result: Dict[str, Any] = {}
        try:
            for npy_path in sorted(d.glob("*.npy")):
                name = npy_path.stem
                arr = np.load(npy_path, mmap_mode="r")
                result[name] = torch.tensor(np.asarray(arr))
        except Exception:
            return None
        return result if result else None

    def set(self, key: str, data: Dict[str, Any]) -> None:
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

    def clear(self) -> None:
        try:
            for d in self.root.glob("*"):
                if d.is_dir():
                    for p in d.glob("*.npy"):
                        p.unlink(missing_ok=True)
                    d.rmdir()
        except Exception:
            pass


def _resolve_openclip(model_name: str) -> Tuple[str, Optional[str]]:
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
        "ViT-H-14-quickgelu": ("ViT-H-14-quickgelu", "dfn5b"),
        "ViT-L-14-quickgelu": ("ViT-L-14-quickgelu", "dfn2b"),
        "ViT-H-14-laion2B": ("hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K", None),
        "ViT-B-32-laion2B": ("hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K", None),
        "BioCLIP": ("hf-hub:imageomics/bioclip", None),
        "BioCLIP-B/16": ("hf-hub:imageomics/bioclip", None),
        "bioclip": ("hf-hub:imageomics/bioclip", None),
        "BioCLIP-2": ("hf-hub:imageomics/bioclip-2", None),
        "BioCLIP-L/14": ("hf-hub:imageomics/bioclip-2", None),
        "bioclip-2": ("hf-hub:imageomics/bioclip-2", None),
        "BioCLIP-2.5": ("hf-hub:imageomics/bioclip-2.5-vith14", None),
        "BioCLIP-H/14": ("hf-hub:imageomics/bioclip-2.5-vith14", None),
        "bioclip-2.5": ("hf-hub:imageomics/bioclip-2.5-vith14", None),
    }
    if model_name in mappings:
        return mappings[model_name]
    return model_name.replace("/", "-"), "laion2b_s32b_b79k"


def _validate_openclip_selection(model_name: str, arch: str, pretrained: Optional[str]) -> None:
    if arch.startswith("hf-hub:"):
        return
    try:
        available = clip.list_pretrained()
    except Exception:
        available = None
    if not available:
        return

    pairs = {(str(a), str(p)) for a, p in available if a is not None and p is not None}
    names = {str(a) for a, _ in available if a is not None}

    if pretrained is None:
        valid = arch in names
    else:
        valid = (arch, pretrained) in pairs
    if valid:
        return

    shortlist = sorted(list(pairs))[:10]
    raise ValueError(
        "Invalid CLIP model configuration: "
        f"model_name='{model_name}' resolved to arch='{arch}', pretrained='{pretrained}'. "
        f"Sample available model/pretrained pairs: {shortlist}"
    )


class CLIPImageEncoderAdapter:
    modality = "image"

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        cache_features: bool = True,
        cache_backend: str = "mmap",
        cache_dir: Union[str, Path] = "cache/clip_features",
    ):
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
                self.feature_cache = (
                    MmapFeatureCache(self.cache_dir)
                    if self.cache_backend == "mmap"
                    else DiskFeatureCache(self.cache_dir)
                )
            else:
                self.feature_cache = {}
        else:
            self.feature_cache = None

        arch, pretrained = _resolve_openclip(self.model_name)
        _validate_openclip_selection(self.model_name, arch, pretrained)

        try:
            model, preprocess_train, preprocess_val = clip.create_model_and_transforms(
                arch,
                pretrained=pretrained,
                device=self.device.type,
            )
        except TypeError:
            try:
                model, preprocess_train, preprocess_val = clip.create_model_and_transforms(
                    arch,
                    pretrained=pretrained,
                )
            except Exception as exc:
                raise ValueError(
                    "Invalid CLIP model configuration: "
                    f"model_name='{self.model_name}' resolved to arch='{arch}', pretrained='{pretrained}'. "
                    f"Original error: {exc}"
                ) from exc
            model = model.to(self.device)
        except Exception as exc:
            raise ValueError(
                "Invalid CLIP model configuration: "
                f"model_name='{self.model_name}' resolved to arch='{arch}', pretrained='{pretrained}'. "
                f"Original error: {exc}"
            ) from exc

        self.model = model.eval()
        self.preprocess = preprocess_val or preprocess_train

        self.intermediate_features: Dict[str, Any] = {}
        self.attention_maps: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

        self._setup_model_info()
        self.embedding_dim = self.feature_dim
        self._register_hooks()

    def _setup_model_info(self) -> None:
        if hasattr(self.model, "visual") and hasattr(self.model.visual, "transformer"):
            self.is_vit = True
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

            if hasattr(self.model.visual.transformer, "resblocks"):
                self.num_layers = len(self.model.visual.transformer.resblocks)
            elif hasattr(self.model.visual.transformer, "blocks"):
                self.num_layers = len(self.model.visual.transformer.blocks)
            else:
                self.num_layers = 12

            grid_size = None
            if hasattr(self.model.visual, "positional_embedding"):
                try:
                    grid_size = int((self.model.visual.positional_embedding.shape[1] - 1) ** 0.5)
                except Exception:
                    grid_size = None
            if grid_size is None and hasattr(self.model.visual, "pos_embed"):
                try:
                    grid_size = int((self.model.visual.pos_embed.shape[1] - 1) ** 0.5)
                except Exception:
                    grid_size = None
            if grid_size is None:
                if any(s in self.model_name for s in ["/32", "-32"]):
                    grid_size = 7
                elif any(s in self.model_name for s in ["/16", "-16", "-14", "/14"]):
                    grid_size = 14
                else:
                    grid_size = 14
            self.grid_size = grid_size
        else:
            self.is_vit = False
            try:
                self.feature_dim = self.model.visual.attnpool.positional_embedding.shape[-1]
            except Exception:
                self.feature_dim = 1024
            self.num_layers = 4
            self.grid_size = 7

    def _register_hooks(self) -> None:
        if self.is_vit:
            target_layers = [3, 6, 9] if self.num_layers >= 12 else [1, 2, 3]
            for layer_idx in target_layers:
                if layer_idx < self.num_layers:
                    if hasattr(self.model.visual.transformer, "resblocks"):
                        layer = self.model.visual.transformer.resblocks[layer_idx]
                    else:
                        layer = self.model.visual.transformer.blocks[layer_idx]
                    layer.register_forward_hook(self._make_hook(f"layer_{layer_idx}"))
                    if hasattr(layer, "attn"):
                        layer.attn.register_forward_hook(
                            self._make_attention_hook(f"attn_{layer_idx}")
                        )
        else:
            layers = [
                getattr(self.model.visual, "layer1", None),
                getattr(self.model.visual, "layer2", None),
                getattr(self.model.visual, "layer3", None),
                getattr(self.model.visual, "layer4", None),
            ]
            for i, layer in enumerate([l for l in layers if l is not None]):
                layer.register_forward_hook(self._make_hook(f"resnet_layer_{i + 1}"))

    def _make_hook(self, name: str):
        def hook(module, input, output):
            _ = (module, input)
            self.intermediate_features[name] = output.detach()

        return hook

    def _make_attention_hook(self, name: str):
        def hook(module, input, output):
            _ = (input, output)
            if hasattr(module, "attention_weights"):
                self.attention_maps[name] = module.attention_weights.detach()

        return hook

    def preprocess_images(self, images: Sequence[Union[str, Path, Image.Image]]) -> Any:
        processed = []
        for image in images:
            if isinstance(image, (str, Path)):
                pil = load_image(image)
                processed.append(self.preprocess(pil))
            elif isinstance(image, Image.Image):
                processed.append(self.preprocess(image))
            else:
                processed.append(torch.as_tensor(image))
        return torch.stack(processed, dim=0)

    def encode(self, images: Sequence[Union[str, Path, Image.Image, Any]]) -> Dict[str, Any]:
        if len(images) == 0:
            return {"global": torch.empty(0, self.embedding_dim)}
        if isinstance(images[0], torch.Tensor):
            batch = torch.stack([torch.as_tensor(image) for image in images], dim=0).to(self.device)
        else:
            batch = self.preprocess_images(images).to(self.device)
        with torch.no_grad():
            global_features = self.model.encode_image(batch)
        return {"global": normalize_tensor(global_features)}

    def extract_features(
        self,
        image_input: Union[str, Path, Image.Image, Any],
        return_attention: bool = True,
        normalize: bool = True,
    ) -> Dict[str, Any]:
        cache_key = None
        if self.cache_features and isinstance(image_input, (str, Path)):
            cache_key = str(image_input)
            if self.feature_cache:
                try:
                    if hasattr(self.feature_cache, "get"):
                        cached = self.feature_cache.get(cache_key)
                        if cached is not None:
                            return cached
                    elif isinstance(self.feature_cache, dict) and cache_key in self.feature_cache:
                        return self.feature_cache[cache_key]
                except Exception:
                    pass

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

        self.intermediate_features.clear()
        self.attention_maps.clear()

        with torch.no_grad():
            global_features = self.model.encode_image(image_tensor)
            if normalize:
                global_features = normalize_tensor(global_features)

        features: Dict[str, Any] = {
            "global": global_features,
            "image_size": image_tensor.shape[-2:],
            "grid_size": self.grid_size,
        }

        for name, feat in self.intermediate_features.items():
            if self.is_vit:
                if feat.dim() == 3 and feat.shape[1] > self.grid_size * self.grid_size:
                    spatial_feat = feat[:, 1:, :]
                    batch_size, seq_len, feat_dim = spatial_feat.shape
                    grid_size = int(seq_len**0.5)
                    spatial_feat = spatial_feat.reshape(batch_size, grid_size, grid_size, feat_dim)
                    spatial_feat = spatial_feat.permute(0, 3, 1, 2)
                    features[name] = spatial_feat
                else:
                    features[name] = feat
            else:
                features[name] = feat

        if return_attention and self.is_vit:
            features.update(self._process_attention_maps())

        if normalize:
            for key, feat in list(features.items()):
                if (
                    isinstance(feat, torch.Tensor)
                    and feat.dim() >= 2
                    and key not in ["image_size", "grid_size"]
                ):
                    features[key] = normalize_tensor(feat, dim=-1)

        if cache_key and self.cache_features and self.feature_cache:
            try:
                if isinstance(self.feature_cache, dict):
                    self.feature_cache[cache_key] = features
                elif hasattr(self.feature_cache, "set"):
                    self.feature_cache.set(cache_key, features)
            except Exception:
                pass

        return features

    def _process_attention_maps(self) -> Dict[str, Any]:
        attention_features: Dict[str, Any] = {}
        for name, attn_map in self.attention_maps.items():
            if attn_map is not None:
                if attn_map.dim() == 4:
                    attn_map = attn_map.mean(dim=1)
                if attn_map.shape[-1] > self.grid_size * self.grid_size:
                    cls_attention = attn_map[:, 0, 1:]
                    batch_size = cls_attention.shape[0]
                    cls_attention = cls_attention.reshape(
                        batch_size, self.grid_size, self.grid_size
                    )
                    attention_features[name] = cls_attention
        return attention_features

    def extract_batch_features(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        results = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            with Timer(f"Processing batch {i // batch_size + 1}"):
                batch_results = [self.extract_features(path, **kwargs) for path in batch_paths]
                results.extend(batch_results)
        return results

    def get_feature_dimensions(self) -> Dict[str, Any]:
        dimensions: Dict[str, Any] = {}
        dimensions["global"] = (self.feature_dim,)
        dimensions["grid_size"] = self.grid_size
        if self.is_vit:
            dimensions["layer_3"] = (self.feature_dim, self.grid_size, self.grid_size)
            dimensions["layer_6"] = (self.feature_dim, self.grid_size, self.grid_size)
            dimensions["layer_9"] = (self.feature_dim, self.grid_size, self.grid_size)
        else:
            dimensions["resnet_layer_1"] = (256, 56, 56)
            dimensions["resnet_layer_2"] = (512, 28, 28)
            dimensions["resnet_layer_3"] = (1024, 14, 14)
            dimensions["resnet_layer_4"] = (2048, 7, 7)
        return dimensions

    def clear_cache(self) -> None:
        if not self.feature_cache:
            return
        try:
            self.feature_cache.clear()
        except Exception:
            pass

    def get_cache_size(self) -> int:
        if not self.feature_cache:
            return 0
        if isinstance(self.feature_cache, dict):
            return len(self.feature_cache)
        try:
            if isinstance(self.feature_cache, DiskFeatureCache):
                return len(list(self.feature_cache.root.glob("*.pt")))
            if isinstance(self.feature_cache, MmapFeatureCache):
                return len([d for d in self.feature_cache.root.glob("*") if d.is_dir()])
        except Exception:
            return 0
        return 0

    def __repr__(self):
        return (
            "CLIPImageEncoderAdapter("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"cache_size={self.get_cache_size()})"
        )


__all__ = ["CLIPImageEncoderAdapter"]
