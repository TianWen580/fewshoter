import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

torch = importlib.import_module("torch")
Tensor = Any


class EncoderContract(ABC):
    """Base protocol-like contract for modality encoders."""

    embedding_dim: int
    modality: str

    @classmethod
    def __subclasshook__(cls, C):
        """Support structural checks for objects implementing ``encode``.

        Args:
            C: Candidate class for structural protocol checks.

        Returns:
            bool | NotImplemented: ``True`` when required methods are found,
            otherwise ``NotImplemented``.
        """
        if cls is EncoderContract and any("encode" in B.__dict__ for B in C.__mro__):
            return True
        return NotImplemented

    @abstractmethod
    def encode(self, images: Sequence[Any]) -> Dict[str, Tensor]:
        """Encode modality inputs into named embeddings.

        Args:
            images: Input sequence for the underlying modality.

        Returns:
            Dict[str, Tensor]: Mapping from embedding key to tensor-like outputs.
        """
        raise NotImplementedError


class ImageEncoder(EncoderContract, ABC):
    """Encoder contract specialization for image modalities."""

    @classmethod
    def __subclasshook__(cls, C):
        """Support structural checks for image encoder implementations.

        Args:
            C: Candidate class for structural protocol checks.

        Returns:
            bool | NotImplemented: ``True`` when required methods are found,
            otherwise ``NotImplemented``.
        """
        if cls is ImageEncoder and all(
            any(name in B.__dict__ for B in C.__mro__) for name in ("encode", "preprocess_images")
        ):
            return True
        return NotImplemented

    @abstractmethod
    def preprocess_images(self, images: Sequence[Any]) -> Tensor:
        """Preprocess raw images into model-ready tensors.

        Args:
            images: Raw image payloads.

        Returns:
            Tensor: Preprocessed image batch.
        """
        raise NotImplementedError
