import importlib
from abc import ABC, abstractmethod
from typing import Any, Iterable

torch = importlib.import_module("torch")
Tensor = Any

from ..core.episode import Episode
from ..modalities.base import EncoderContract


class LearnerContract(ABC):
    """Abstract contract for few-shot learners."""

    @classmethod
    def __subclasshook__(cls, C):
        """Enable structural checks for learner-like implementations.

        Args:
            C: Candidate class to test for protocol conformance.

        Returns:
            bool | NotImplemented: ``True`` when required methods exist, otherwise
            ``NotImplemented``.
        """
        if cls is LearnerContract and all(
            any(name in B.__dict__ for B in C.__mro__)
            for name in ("fit", "predict", "from_encoder")
        ):
            return True
        return NotImplemented

    @abstractmethod
    def fit(self, episodes: Iterable[Episode]) -> None:
        """Fit learner state using episodic support data.

        Args:
            episodes: Episodes used to build internal learner state.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, embeddings: Tensor) -> Any:
        """Predict class identifiers from embeddings.

        Args:
            embeddings: Batch embedding tensor-like object.

        Returns:
            Any: Predicted class labels or indices.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_encoder(cls, encoder: EncoderContract, config: Any) -> "LearnerContract":
        """Create learner from encoder and configuration.

        Args:
            encoder: Modality encoder instance.
            config: Runtime configuration object.

        Returns:
            LearnerContract: Constructed learner implementation.
        """
        ...
