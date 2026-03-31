import importlib
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from ..modalities.audio import AudioEmbeddingMetadata
from .audio_pipeline import AudioPipelineBoundary

torch = importlib.import_module("torch")


class PerchEncoderPlaceholder:
    """Placeholder PERCH-like audio encoder for integration boundaries."""

    modality = "audio"

    def __init__(
        self,
        pipeline: AudioPipelineBoundary,
        embedding_fn: Optional[Callable[[Sequence[Any], int], Any]] = None,
        *,
        expected_sample_rate_hz: int = 32000,
        expected_window_seconds: float = 5.0,
        expected_embedding_dim: int = 1536,
    ):
        """Initialize encoder boundary and validate expected metadata contract.

        Args:
            pipeline: Audio preprocessing/provenance boundary implementation.
            embedding_fn: Optional function producing embeddings from preprocessed audio.
            expected_sample_rate_hz: Required sample rate for boundary validation.
            expected_window_seconds: Required window duration for boundary validation.
            expected_embedding_dim: Output embedding dimension.

        Returns:
            None: Initializes encoder state and validates boundary metadata.

        Raises:
            ValueError: If pipeline metadata does not match expected contract.
        """
        self.pipeline = pipeline
        self.embedding_fn = embedding_fn
        self.expected_sample_rate_hz = int(expected_sample_rate_hz)
        self.expected_window_seconds = float(expected_window_seconds)
        self.embedding_dim = int(expected_embedding_dim)
        self.pipeline.validate_expected(
            sample_rate_hz=self.expected_sample_rate_hz,
            window_seconds=self.expected_window_seconds,
            embedding_dim=self.embedding_dim,
        )
        self.metadata = self.pipeline.metadata()

    def preprocess_audio(self, audio: Sequence[Any]):
        """Preprocess audio via pipeline after metadata validation.

        Args:
            audio: Sequence of raw audio payloads.

        Returns:
            Any: Pipeline-specific preprocessed audio batch.
        """
        self._validate_metadata(self.pipeline.metadata())
        return self.pipeline.preprocess(audio)

    def provenance(self) -> Mapping[str, str]:
        """Return provenance payload emitted by the boundary pipeline."""
        return self.pipeline.provenance_payload()

    def encode(self, audio: Sequence[Any]) -> Dict[str, Any]:
        """Encode audio into a ``{"global": embeddings}`` mapping.

        Args:
            audio: Sequence of raw audio payloads.

        Returns:
            Dict[str, Any]: Dictionary with a rank-2 global embedding tensor.

        Raises:
            ValueError: If embedding output rank or dimension is invalid.
        """
        processed = self.preprocess_audio(audio)
        if self.embedding_fn is None:
            embeddings = self._default_embeddings(processed)
        else:
            embeddings = torch.as_tensor(
                self.embedding_fn(processed, self.embedding_dim),
                dtype=torch.float32,
            )

        if embeddings.dim() != 2:
            raise ValueError("Placeholder embedding output must be rank-2 [batch, dim]")
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Placeholder embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[-1]}"
            )
        return {"global": embeddings}

    def _default_embeddings(self, processed: Sequence[Any]):
        if len(processed) == 0:
            return torch.empty(0, self.embedding_dim, dtype=torch.float32)
        rows = []
        for waveform in processed:
            tensor = torch.as_tensor(waveform, dtype=torch.float32).reshape(-1)
            mean = tensor.mean() if tensor.numel() > 0 else torch.tensor(0.0)
            std = tensor.std(unbiased=False) if tensor.numel() > 1 else torch.tensor(0.0)
            row = torch.zeros(self.embedding_dim, dtype=torch.float32)
            row[0] = mean
            if self.embedding_dim > 1:
                row[1] = std
            rows.append(row)
        return torch.stack(rows, dim=0)

    def _validate_metadata(self, metadata: AudioEmbeddingMetadata) -> None:
        self.pipeline.validate_expected(
            sample_rate_hz=self.expected_sample_rate_hz,
            window_seconds=self.expected_window_seconds,
            embedding_dim=self.embedding_dim,
        )
        if metadata.embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Invalid metadata: expected embedding_dim={self.embedding_dim}, "
                f"got {metadata.embedding_dim}"
            )
