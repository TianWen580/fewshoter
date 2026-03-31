from collections.abc import Iterable
from typing import Any

import importlib

from ..core.episode import Episode
from .utils import l2_normalize, squared_euclidean_distance, validate_episode_tensors


torch = importlib.import_module("torch")


class PrototypicalLearner:
    """Minimal prototypical-network style learner over embedding tensors."""

    def __init__(self):
        """Initialize empty prototype and class-id state."""
        self._class_ids: Any | None = None
        self._prototypes: Any | None = None

    @classmethod
    def from_encoder(cls, encoder: Any, config: Any) -> "PrototypicalLearner":
        """Construct learner from encoder/config inputs.

        Args:
            encoder: Unused encoder placeholder for API compatibility.
            config: Unused config placeholder for API compatibility.

        Returns:
            PrototypicalLearner: New learner instance.
        """
        _ = encoder
        _ = config
        return cls()

    def fit(self, episodes: Iterable[Episode]) -> None:
        """Fit class prototypes from support embeddings across episodes.

        Args:
            episodes: Episodes with support embeddings and labels.

        Raises:
            ValueError: If episodes are empty or contain invalid tensors.
        """
        support_embeddings_batches: list[Any] = []
        support_labels_batches: list[Any] = []

        for episode in episodes:
            if episode.support_embeddings is None or episode.support_labels is None:
                raise ValueError("episode must define support embeddings and support labels")

            support_embeddings = torch.as_tensor(episode.support_embeddings)
            support_labels = torch.as_tensor(episode.support_labels).to(torch.long)

            validate_episode_tensors(
                support_embeddings=support_embeddings,
                support_labels=support_labels,
                query_embeddings=episode.query_embeddings,
                query_labels=None
                if episode.query_labels is None
                else torch.as_tensor(episode.query_labels),
            )

            support_embeddings_batches.append(support_embeddings)
            support_labels_batches.append(support_labels)

        if not support_embeddings_batches:
            raise ValueError("fit requires at least one episode")

        support_embeddings = torch.cat(support_embeddings_batches, dim=0)
        support_labels = torch.cat(support_labels_batches, dim=0)
        self._class_ids, self._prototypes = self._build_prototypes(
            support_embeddings=support_embeddings,
            support_labels=support_labels,
        )

    def predict(self, embeddings: Any):
        """Predict labels for query embeddings using fitted prototypes.

        Args:
            embeddings: Query embedding tensor-like input.

        Returns:
            Any: Predicted class ids aligned with fitted support labels.

        Raises:
            ValueError: If learner is unfitted or dimensions are incompatible.
        """
        if self._prototypes is None or self._class_ids is None:
            raise ValueError("learner must be fit before calling predict")

        query_embeddings = torch.as_tensor(embeddings)
        if query_embeddings.ndim != 2:
            raise ValueError("query embeddings must be a 2D tensor")
        if query_embeddings.shape[1] != self._prototypes.shape[1]:
            raise ValueError(
                "query embeddings must share embedding dimension with fitted prototypes"
            )

        normalized_query = l2_normalize(query_embeddings)
        distances = squared_euclidean_distance(normalized_query, self._prototypes)
        nearest_indices = distances.argmin(dim=1)
        return self._class_ids[nearest_indices]

    def predict_from_tensors(
        self,
        support_embeddings: Any,
        support_labels: Any,
        query_embeddings: Any,
    ):
        """Predict labels directly from support/query tensors.

        Args:
            support_embeddings: Support embedding matrix.
            support_labels: Support labels vector.
            query_embeddings: Query embedding matrix.

        Returns:
            Any: Predicted class ids for each query sample.

        Raises:
            ValueError: If tensor shapes are invalid.
        """
        support_embeddings = torch.as_tensor(support_embeddings)
        support_labels = torch.as_tensor(support_labels).to(torch.long)
        query_embeddings = torch.as_tensor(query_embeddings)

        validate_episode_tensors(
            support_embeddings=support_embeddings,
            support_labels=support_labels,
            query_embeddings=query_embeddings,
        )

        class_ids, prototypes = self._build_prototypes(
            support_embeddings=support_embeddings,
            support_labels=support_labels,
        )

        normalized_query = l2_normalize(query_embeddings)
        distances = squared_euclidean_distance(normalized_query, prototypes)
        nearest_indices = distances.argmin(dim=1)
        return class_ids[nearest_indices]

    def predict_episode(self, episode: Episode):
        """Predict query labels for a complete episode.

        Args:
            episode: Episode containing support and query embeddings/labels.

        Returns:
            Any: Predicted class ids for the episode query set.

        Raises:
            ValueError: If required embeddings/labels are missing.
        """
        if episode.support_embeddings is None or episode.support_labels is None:
            raise ValueError("episode must define support embeddings and support labels")
        if episode.query_embeddings is None:
            raise ValueError("episode must define query embeddings")

        return self.predict_from_tensors(
            support_embeddings=torch.as_tensor(episode.support_embeddings),
            support_labels=torch.as_tensor(episode.support_labels),
            query_embeddings=torch.as_tensor(episode.query_embeddings),
        )

    @staticmethod
    def _build_prototypes(support_embeddings: Any, support_labels: Any):
        """Build normalized class prototypes from labeled support embeddings."""
        normalized_support = l2_normalize(support_embeddings)
        class_ids = torch.unique(support_labels, sorted=True)
        prototypes = []

        for class_id in class_ids:
            class_mask = support_labels == class_id
            if not torch.any(class_mask):
                raise ValueError("each class must have at least one support embedding")
            class_embeddings = normalized_support[class_mask]
            prototypes.append(class_embeddings.mean(dim=0))

        return class_ids, torch.stack(prototypes, dim=0)
