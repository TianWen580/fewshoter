from dataclasses import dataclass
import importlib
from typing import Any, Dict, List, Mapping, Sequence

from ..modalities.audio import AudioEmbeddingMetadata, validate_audio_metadata

torch = importlib.import_module("torch")


@dataclass(frozen=True)
class AudioPreprocessingConfig:
    """Static preprocessing configuration for audio boundary pipelines."""

    sample_rate_hz: int = 32000
    window_seconds: float = 5.0
    channel_count: int = 1
    window_hop_seconds: float = 5.0
    policy_name: str = "perch2-default"


@dataclass(frozen=True)
class AudioProvenance:
    """Provenance fields identifying encoder/checkpoint/runtime metadata."""

    encoder_name: str = "perch2"
    checkpoint: str = "placeholder"
    provenance_tag: str = "offline-placeholder"
    runtime_framework: str = "none"


class AudioPipelineBoundary:
    """Boundary object that validates and preprocesses audio integration data."""

    def __init__(
        self,
        preprocessing: AudioPreprocessingConfig,
        provenance: AudioProvenance,
        embedding_dim: int,
    ):
        """Initialize pipeline boundary and validate metadata contract.

        Args:
            preprocessing: Audio preprocessing configuration.
            provenance: Encoder provenance metadata.
            embedding_dim: Output embedding dimension.

        Returns:
            None: Initializes boundary state and validates configuration.

        Raises:
            ValueError: If boundary configuration is invalid.
        """
        self.preprocessing = preprocessing
        self.provenance = provenance
        self.embedding_dim = int(embedding_dim)
        self._validate_boundary()

    def _validate_boundary(self) -> None:
        if self.preprocessing.channel_count != 1:
            raise ValueError("channel_count must be 1 for the placeholder boundary")
        if self.preprocessing.window_hop_seconds <= 0:
            raise ValueError("window_hop_seconds must be positive")
        metadata = self.metadata()
        validate_audio_metadata(metadata)

    def metadata(self) -> AudioEmbeddingMetadata:
        """Build audio embedding metadata for downstream contract checks."""
        return AudioEmbeddingMetadata(
            sample_rate_hz=self.preprocessing.sample_rate_hz,
            window_seconds=self.preprocessing.window_seconds,
            embedding_dim=self.embedding_dim,
            encoder_name=self.provenance.encoder_name,
            checkpoint=self.provenance.checkpoint,
            preprocessing_policy=self.preprocessing.policy_name,
            provenance_tag=self.provenance.provenance_tag,
        )

    def preprocess(self, audio: Sequence[Any]) -> List[Any]:
        """Convert input audio payloads to float tensors.

        Args:
            audio: Sequence of raw audio payloads.

        Returns:
            List[Any]: Normalized tensor-like audio payloads.

        Raises:
            ValueError: If any payload is scalar rather than waveform-like.
        """
        normalized: List[Any] = []
        for item in audio:
            tensor = torch.as_tensor(item, dtype=torch.float32)
            if tensor.dim() == 0:
                raise ValueError("audio items must be at least 1D waveforms")
            normalized.append(tensor)
        return normalized

    def validate_expected(
        self,
        *,
        sample_rate_hz: int,
        window_seconds: float,
        embedding_dim: int,
    ) -> None:
        """Validate metadata matches expected sample/window/dimension values.

        Args:
            sample_rate_hz: Expected sample rate.
            window_seconds: Expected window duration.
            embedding_dim: Expected embedding dimension.

        Returns:
            None: Raises when metadata constraints are violated.

        Raises:
            ValueError: If metadata values differ from expectations.
        """
        metadata = self.metadata()
        if metadata.sample_rate_hz != int(sample_rate_hz):
            raise ValueError(
                f"Invalid metadata: expected sample_rate_hz={sample_rate_hz}, "
                f"got {metadata.sample_rate_hz}"
            )
        if abs(float(metadata.window_seconds) - float(window_seconds)) > 1e-9:
            raise ValueError(
                f"Invalid metadata: expected window_seconds={window_seconds}, "
                f"got {metadata.window_seconds}"
            )
        if metadata.embedding_dim != int(embedding_dim):
            raise ValueError(
                f"Invalid metadata: expected embedding_dim={embedding_dim}, "
                f"got {metadata.embedding_dim}"
            )

    def provenance_payload(self) -> Dict[str, str]:
        """Return provenance payload as a serializable mapping."""
        return {
            "encoder_name": self.provenance.encoder_name,
            "checkpoint": self.provenance.checkpoint,
            "preprocessing_policy": self.preprocessing.policy_name,
            "provenance_tag": self.provenance.provenance_tag,
            "runtime_framework": self.provenance.runtime_framework,
        }

    def cache_key(self) -> str:
        """Return deterministic cache key from preprocessing/provenance settings."""
        return (
            f"{self.provenance.encoder_name}|{self.provenance.checkpoint}|"
            f"{self.preprocessing.policy_name}|{self.preprocessing.sample_rate_hz}|"
            f"{self.preprocessing.window_seconds}|{self.embedding_dim}"
        )
