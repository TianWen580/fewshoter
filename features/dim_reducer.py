"""
Sklearn-driven dimensionality reduction utilities for CLIP few-shot embeddings.

- Default method: PCA
- Can be fit on support-set embeddings, then reused to transform query/prototypes

English code + bilingual comments; use English in code/comments, docs bilingual externally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Any

import numpy as np
import pickle

try:  # sklearn is already a dependency in this package
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None  # type: ignore

# Torch is optional at runtime here; we accept/return tensors if provided
try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


@dataclass
class DimReduceConfig:
    method: str = "pca"  # only 'pca' supported for now
    n_components: Union[int, float, None] = (
        0.95  # int or float in (0,1] for variance ratio
    )
    whiten: bool = False
    random_state: Optional[int] = 42


class DimensionalityReducer:
    """Thin wrapper over sklearn DR models with Torch/NumPy IO.

    API:
      - fit(X)
      - transform(x_or_X) -> same type and batch handling
      - fit_transform(X)
      - serialize()/deserialize()
    """

    def __init__(self, cfg: Optional[DimReduceConfig] = None):
        if PCA is None:
            raise ImportError("scikit-learn is required for DimensionalityReducer")
        self.cfg = cfg or DimReduceConfig()
        self.method = (self.cfg.method or "pca").lower()
        self.estimator: Any = None

    # -------- Core helpers --------
    @staticmethod
    def _to_numpy(x: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        if _HAS_TORCH and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

    @staticmethod
    def _back_to_type(
        x_np: np.ndarray, like: Union[np.ndarray, "torch.Tensor"]
    ) -> Union[np.ndarray, "torch.Tensor"]:
        if _HAS_TORCH and isinstance(like, torch.Tensor):
            return torch.tensor(x_np, dtype=like.dtype)
        return x_np

    def _make_estimator(self):
        if self.method == "pca":
            return PCA(
                n_components=self.cfg.n_components,
                whiten=self.cfg.whiten,
                random_state=self.cfg.random_state,
            )
        raise ValueError(f"Unsupported dim reduction method: {self.method}")

    # -------- Public API --------
    def fit(self, X: Union[np.ndarray, "torch.Tensor"]) -> "DimensionalityReducer":
        X_np = self._to_numpy(X)
        X_np = X_np.reshape(X_np.shape[0], -1)
        self.estimator = self._make_estimator()
        self.estimator.fit(X_np)
        return self

    def transform(
        self, X: Union[np.ndarray, "torch.Tensor"]
    ) -> Union[np.ndarray, "torch.Tensor"]:
        if self.estimator is None:
            raise RuntimeError("DimensionalityReducer is not fit")
        inp = X
        X_np = self._to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
            single = True
        else:
            X_np = X_np.reshape(X_np.shape[0], -1)
            single = False
        Y = self.estimator.transform(X_np)
        if single:
            Y = Y.reshape(-1)
        return self._back_to_type(Y, inp)

    def fit_transform(
        self, X: Union[np.ndarray, "torch.Tensor"]
    ) -> Union[np.ndarray, "torch.Tensor"]:
        self.fit(X)
        return self.transform(X)

    @property
    def output_dim(self) -> Optional[int]:
        try:
            if hasattr(self.estimator, "n_components_"):
                return int(self.estimator.n_components_)
            if hasattr(self.estimator, "components_"):
                return int(getattr(self.estimator, "components_").shape[0])
        except Exception:
            return None
        return None

    # -------- Persistence --------
    def serialize(self) -> bytes:
        payload = {
            "cfg": self.cfg.__dict__,
            "method": self.method,
            "estimator": self.estimator,
        }
        return pickle.dumps(payload)

    @staticmethod
    def deserialize(blob: bytes) -> "DimensionalityReducer":
        payload = pickle.loads(blob)
        cfg = DimReduceConfig(**payload.get("cfg", {}))
        inst = DimensionalityReducer(cfg)
        inst.method = payload.get("method", cfg.method)
        inst.estimator = payload.get("estimator", None)
        return inst
