from dataclasses import dataclass
from collections.abc import Sized
from typing import Optional

import importlib


@dataclass
class Episode:
    """Container for one few-shot evaluation/training episode.

    Stores support/query embeddings, labels, class names, and optional source item
    references used for leakage checks.
    """

    support_embeddings: object | None = None
    support_labels: Sized | None = None
    query_embeddings: object | None = None
    query_labels: Sized | None = None
    class_names: list[str] | None = None
    n_way: int = 0
    k_shot: int = 0
    num_queries: int = 0
    seed: Optional[int] = None
    support_items: list[object] | None = None
    query_items: list[object] | None = None

    def validate(self) -> "Episode":
        """Validate episode consistency constraints.

        Returns:
            Episode: The same episode instance when validation succeeds.

        Raises:
            ValueError: If labels/classes are missing, dimensions are inconsistent,
                class counts do not match ``n_way``, or support/query items overlap.
        """
        if self.support_labels is None or self.query_labels is None or self.class_names is None:
            raise ValueError("episode must define support labels, query labels, and class_names")

        if self.support_items is not None and len(self.support_items) != len(self.support_labels):
            raise ValueError("support items and labels must have the same length")
        if self.query_items is not None and len(self.query_items) != len(self.query_labels):
            raise ValueError("query items and labels must have the same length")

        if len(self.class_names) != self.n_way:
            raise ValueError("class_names must contain exactly n_way classes")

        if self.support_items is not None and self.query_items is not None:
            overlap = set(self.support_items).intersection(self.query_items)
            if overlap:
                raise ValueError("support and query items must not overlap")

        return self


@dataclass
class EpisodeConfig:
    """Configuration values for constructing episodic few-shot tasks."""

    n_way: int = 5
    k_shot: int = 1
    num_queries: int = 5
    seed: int = 42


torch = importlib.import_module("torch")
