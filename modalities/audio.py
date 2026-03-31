from dataclasses import dataclass
import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Sequence

from .base import EncoderContract

torch = importlib.import_module("torch")
Tensor = Any


@dataclass(frozen=True)
class AudioEmbeddingMetadata:
    """Metadata describing audio embedding format and provenance."""

    sample_rate_hz: int
    window_seconds: float
    embedding_dim: int
    encoder_name: str
    checkpoint: str
    preprocessing_policy: str
    provenance_tag: str


def validate_audio_metadata(metadata: AudioEmbeddingMetadata) -> None:
    """Validate constraints for audio embedding metadata.

    Args:
        metadata: Metadata object to validate.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    if metadata.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if metadata.window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if metadata.embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")

    text_fields = {
        "encoder_name": metadata.encoder_name,
        "checkpoint": metadata.checkpoint,
        "preprocessing_policy": metadata.preprocessing_policy,
        "provenance_tag": metadata.provenance_tag,
    }
    for key, value in text_fields.items():
        if not isinstance(value, str) or len(value.strip()) == 0:
            raise ValueError(f"{key} must be a non-empty string")


class AudioEncoder(EncoderContract, ABC):
    """Abstract contract for audio encoder implementations."""

    metadata: AudioEmbeddingMetadata

    @classmethod
    def __subclasshook__(cls, C):
        """Support structural checks for audio encoder implementations.

        Args:
            C: Candidate class for structural protocol checks.

        Returns:
            bool | NotImplemented: ``True`` when required methods are found,
            otherwise ``NotImplemented``.
        """
        if cls is AudioEncoder and all(
            any(name in B.__dict__ for B in C.__mro__)
            for name in ("preprocess_audio", "encode", "provenance")
        ):
            return True
        return NotImplemented

    @abstractmethod
    def preprocess_audio(self, audio: Sequence[Any]) -> Tensor:
        """Preprocess raw audio payloads into model-ready tensors.

        Args:
            audio: Sequence of audio payloads.

        Returns:
            Tensor: Preprocessed audio batch.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, images: Sequence[Any]) -> Dict[str, Tensor]:
        """Encode audio payloads into named embedding tensors.

        Args:
            images: Sequence of input audio payloads.

        Returns:
            Dict[str, Tensor]: Mapping from embedding key to tensor output.
        """
        raise NotImplementedError

    @abstractmethod
    def provenance(self) -> Mapping[str, str]:
        """Return provenance metadata describing encoder/runtime details.

        Returns:
            Mapping[str, str]: Serializable metadata fields for reproducibility.
        """
        raise NotImplementedError
