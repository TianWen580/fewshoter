from __future__ import annotations

import random
from dataclasses import dataclass
from collections.abc import Iterator, Mapping, Sequence

from ..core.episode import Episode
from .samplers import EpisodeSampler


@dataclass(frozen=True)
class EpisodeDatasetConfig:
    """Configuration for generating episodes from an :class:`EpisodeSampler`."""

    n_way: int
    k_shot: int
    num_queries: int
    num_episodes: int | None = None
    split: str | None = None
    classes: Sequence[str] | None = None
    base_seed: int | None = None


class EpisodeDataset:
    """Dataset-like wrapper that lazily samples few-shot episodes by index."""

    def __init__(
        self,
        sampler: EpisodeSampler,
        *,
        n_way: int,
        k_shot: int,
        num_queries: int,
        num_episodes: int | None = None,
        split: str | None = None,
        classes: Sequence[str] | None = None,
        base_seed: int | None = None,
    ):
        """Initialize episodic sampling dataset parameters.

        Args:
            sampler: Episode sampler used to materialize episodes.
            n_way: Number of classes per episode.
            k_shot: Number of support samples per class.
            num_queries: Number of query samples per class.
            num_episodes: Optional finite dataset length.
            split: Optional class split selector.
            classes: Optional class allowlist.
            base_seed: Optional base seed offset by sample index.

        Returns:
            None: Initializes dataset state.
        """
        self.sampler: EpisodeSampler = sampler
        self.config: EpisodeDatasetConfig = EpisodeDatasetConfig(
            n_way=n_way,
            k_shot=k_shot,
            num_queries=num_queries,
            num_episodes=num_episodes,
            split=split,
            classes=classes,
            base_seed=base_seed,
        )

    def __len__(self) -> int:
        """Return finite number of available episodes.

        Returns:
            int: Configured episode count.

        Raises:
            TypeError: If ``num_episodes`` is not configured.
        """
        if self.config.num_episodes is None:
            raise TypeError("EpisodeDataset has no finite length when num_episodes is None")
        return self.config.num_episodes

    def __getitem__(self, index: int) -> Episode:
        """Sample an episode for a specific index.

        Args:
            index: Zero-based episode index.

        Returns:
            Episode: Sampled and validated episode.

        Raises:
            IndexError: If index is negative or exceeds finite bounds.
        """
        if index < 0:
            raise IndexError("EpisodeDataset index must be non-negative")
        if self.config.num_episodes is not None and index >= self.config.num_episodes:
            raise IndexError("EpisodeDataset index out of range")

        seed = None
        if self.config.base_seed is not None:
            seed = self.config.base_seed + index

        return self.sampler.sample_episode(
            n_way=self.config.n_way,
            k_shot=self.config.k_shot,
            num_queries=self.config.num_queries,
            seed=seed,
            split=self.config.split,
            classes=self.config.classes,
        )

    def __iter__(self) -> Iterator[Episode]:
        """Iterate indefinitely or until configured finite length.

        Returns:
            Iterator[Episode]: Iterator yielding sampled episodes.
        """
        idx = 0
        while self.config.num_episodes is None or idx < self.config.num_episodes:
            yield self[idx]
            idx += 1


@dataclass(frozen=True)
class DatasetSplitter:
    """Split class-indexed samples into train/val/test partitions."""

    class_to_samples: Mapping[str, Sequence[object]]
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    def __post_init__(self):
        """Validate split ratios and normalize class sample mappings.

        Returns:
            None: Mutates dataclass internals in-place.

        Raises:
            ValueError: If split ratios do not sum to 1.0.
        """
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        object.__setattr__(
            self, "class_to_samples", {k: list(v) for k, v in self.class_to_samples.items()}
        )

    def split(self) -> dict[str, dict[str, list[object]]]:
        """Split classes into train/val/test dictionaries.

        Returns:
            dict[str, dict[str, list[object]]]: Nested mapping keyed by split name.
        """
        classes = sorted(self.class_to_samples.keys())
        rng = random.Random(self.seed)
        rng.shuffle(classes)

        n_train = int(len(classes) * self.train_ratio)
        n_val = int(len(classes) * self.val_ratio)
        train_classes = classes[:n_train]
        val_classes = classes[n_train : n_train + n_val]
        test_classes = classes[n_train + n_val :]

        return {
            "train": {c: list(self.class_to_samples[c]) for c in train_classes},
            "val": {c: list(self.class_to_samples[c]) for c in val_classes},
            "test": {c: list(self.class_to_samples[c]) for c in test_classes},
        }

    def create_episode_samplers(self) -> dict[str, EpisodeSampler]:
        """Create one :class:`EpisodeSampler` per split.

        Returns:
            dict[str, EpisodeSampler]: Mapping of split names to samplers.
        """
        split_map = self.split()
        return {
            split_name: EpisodeSampler(class_map) for split_name, class_map in split_map.items()
        }
