from __future__ import annotations

import random
from dataclasses import dataclass
from collections.abc import Mapping, Sequence, Sized
from typing import Optional

import importlib


torch = importlib.import_module("torch")

from ..core.episode import Episode


class EpisodeSamplingError(ValueError):
    """Raised when an episode cannot be sampled from available data."""

    pass


@dataclass(frozen=True)
class EpisodeSampler:
    """Sampler for constructing N-way K-shot episodic tasks."""

    class_to_samples: Mapping[str, Sequence[object]]
    base_classes: Optional[Sequence[str]] = None
    novel_classes: Optional[Sequence[str]] = None

    def __post_init__(self):
        """Materialize class sample mappings into mutable lists."""
        object.__setattr__(
            self, "class_to_samples", {k: list(v) for k, v in self.class_to_samples.items()}
        )

    def _candidate_classes(self, split: Optional[str]) -> list[str]:
        all_classes = sorted(self.class_to_samples)
        if split is None:
            return all_classes
        if split == "base":
            return list(self.base_classes) if self.base_classes is not None else all_classes
        if split == "novel":
            return list(self.novel_classes) if self.novel_classes is not None else all_classes
        raise ValueError("split must be one of: None, 'base', 'novel'")

    def sample_episode(
        self,
        *,
        n_way: int,
        k_shot: int,
        num_queries: int,
        seed: Optional[int] = None,
        split: Optional[str] = None,
        classes: Optional[Sequence[str]] = None,
    ) -> Episode:
        """Sample a single episode from the configured class-to-sample mapping.

        Args:
            n_way: Number of classes in the episode.
            k_shot: Number of support examples per class.
            num_queries: Number of query examples per class.
            seed: Optional random seed for deterministic sampling.
            split: Optional split selector (``None``, ``base``, or ``novel``).
            classes: Optional allowlist of classes to sample from.

        Returns:
            Episode: A validated episode with support/query items and labels.

        Raises:
            ValueError: If input shape constraints are invalid.
            EpisodeSamplingError: If not enough classes or samples are available.
        """
        if n_way < 1:
            raise ValueError("n_way must be at least 1")
        if k_shot < 1:
            raise ValueError("k_shot must be at least 1")
        if num_queries < 1:
            raise ValueError("num_queries must be at least 1")

        rng = random.Random(seed)
        candidate_classes = self._candidate_classes(split)
        if classes is not None:
            candidate_classes = [c for c in candidate_classes if c in set(classes)]

        if len(candidate_classes) < n_way:
            raise EpisodeSamplingError(
                f"requested {n_way}-way episode but only {len(candidate_classes)} classes are available"
            )

        selected_classes = rng.sample(candidate_classes, n_way)
        support_items: list[object] = []
        query_items: list[object] = []
        support_labels: list[int] = []
        query_labels: list[int] = []

        required = k_shot + num_queries
        for label_idx, class_name in enumerate(selected_classes):
            samples = list(self.class_to_samples[class_name])
            if len(samples) < required:
                raise EpisodeSamplingError(
                    f"Class '{class_name}' has {len(samples)} items, needs {required} for {k_shot}-shot {num_queries}-query episodes"
                )

            shuffled = list(samples)
            rng.shuffle(shuffled)
            support = shuffled[:k_shot]
            query = shuffled[k_shot : k_shot + num_queries]

            support_items.extend(support)
            query_items.extend(query)
            support_labels.extend([label_idx] * len(support))
            query_labels.extend([label_idx] * len(query))

        episode = Episode(
            support_embeddings=None,
            support_labels=_to_tensor(support_labels),
            query_embeddings=None,
            query_labels=_to_tensor(query_labels),
            class_names=selected_classes,
            n_way=n_way,
            k_shot=k_shot,
            num_queries=num_queries,
            seed=seed,
            support_items=support_items,
            query_items=query_items,
        )
        return episode.validate()


def _to_tensor(values: Sequence[int]):
    return torch.tensor(list(values), dtype=torch.long)
