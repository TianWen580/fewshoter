from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import importlib

from ..core.episode import Episode
from .base import Metrics


torch = importlib.import_module("torch")


@dataclass
class ComparisonMetrics:
    """Side-by-side metrics for legacy and prototypical learners."""

    legacy: Metrics
    prototypical: Metrics

    @property
    def accuracy_delta(self) -> float:
        """Return ``prototypical.accuracy - legacy.accuracy``."""
        return self.prototypical.accuracy - self.legacy.accuracy


class EpisodicEvaluator:
    """Evaluator for episodic few-shot benchmarks and comparisons."""

    def __init__(self, confidence_level: float = 0.95):
        """Initialize evaluator.

        Args:
            confidence_level: Confidence level used for CI half-width estimates.

        Returns:
            None: Initializes evaluator state.

        Raises:
            ValueError: If confidence level is outside ``(0, 1)``.
        """
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        self.confidence_level = confidence_level

    def compute_metrics(
        self,
        learner: Any,
        episodes: Iterable[Episode],
        *,
        split: str | None = None,
        allowed_classes: Sequence[str] | None = None,
    ) -> Metrics:
        """Compute episodic metrics for a learner.

        Args:
            learner: Learner implementing ``predict`` or ``predict_episode``.
            episodes: Episodes to evaluate.
            split: Optional split name used in validation errors.
            allowed_classes: Optional class allowlist for split validation.

        Returns:
            Metrics: Aggregate accuracy, CI half-width, and per-class accuracy.

        Raises:
            ValueError: If episodes are invalid or predictions are inconsistent.
        """
        self._validate_split(split)

        allowed_set = set(allowed_classes) if allowed_classes is not None else None
        episode_accuracies: list[float] = []
        class_correct: dict[str, int] = {}
        class_total: dict[str, int] = {}

        for episode in episodes:
            self._validate_episode(episode)
            self._validate_episode_classes(episode, allowed_set, split)

            predictions = self._predict_episode(learner, episode)
            query_labels = torch.as_tensor(episode.query_labels, dtype=torch.long)

            if predictions.shape[0] != query_labels.shape[0]:
                raise ValueError("predictions and query labels must have the same length")

            episode_correct = predictions.eq(query_labels)
            episode_accuracies.append(float(episode_correct.float().mean().item()))

            for index in range(query_labels.shape[0]):
                label_idx = int(query_labels[index].item())
                class_name = self._class_name_for_label(episode, label_idx)
                class_total[class_name] = class_total.get(class_name, 0) + 1
                class_correct[class_name] = class_correct.get(class_name, 0) + int(
                    episode_correct[index].item()
                )

        if not episode_accuracies:
            raise ValueError("episodes must contain at least one episode")

        mean_accuracy = float(sum(episode_accuracies) / len(episode_accuracies))
        confidence_interval = self._confidence_interval_half_width(episode_accuracies)
        per_class_accuracy = {
            class_name: class_correct[class_name] / class_total[class_name]
            for class_name in sorted(class_total)
        }

        return Metrics(
            accuracy=mean_accuracy,
            confidence_interval=confidence_interval,
            per_class_accuracy=per_class_accuracy,
        )

    def evaluate(
        self,
        learner: Any,
        episodes: Iterable[Episode],
        *,
        split: str | None = None,
        allowed_classes: Sequence[str] | None = None,
    ) -> Metrics:
        """Alias of :meth:`compute_metrics` for evaluator-style APIs.

        Args:
            learner: Learner implementing ``predict`` or ``predict_episode``.
            episodes: Episodes to evaluate.
            split: Optional split name used in validation errors.
            allowed_classes: Optional class allowlist for split validation.

        Returns:
            Metrics: Aggregate accuracy, confidence interval, and per-class scores.
        """
        return self.compute_metrics(
            learner,
            episodes,
            split=split,
            allowed_classes=allowed_classes,
        )

    def compare(
        self,
        *,
        legacy_learner: Any,
        prototypical_learner: Any,
        episodes: Sequence[Episode],
        split: str | None = None,
        allowed_classes: Sequence[str] | None = None,
    ) -> ComparisonMetrics:
        """Compare legacy and prototypical learners on identical episodes.

        Args:
            legacy_learner: Learner implementing legacy prediction behavior.
            prototypical_learner: Learner implementing prototypical behavior.
            episodes: Shared episodes used for both learners.
            split: Optional split name used in validation errors.
            allowed_classes: Optional class allowlist for split validation.

        Returns:
            ComparisonMetrics: Metrics for both learners and accuracy delta.
        """
        episode_list = list(episodes)
        legacy_metrics = self.evaluate(
            legacy_learner,
            episode_list,
            split=split,
            allowed_classes=allowed_classes,
        )
        proto_metrics = self.evaluate(
            prototypical_learner,
            episode_list,
            split=split,
            allowed_classes=allowed_classes,
        )
        return ComparisonMetrics(legacy=legacy_metrics, prototypical=proto_metrics)

    @staticmethod
    def _validate_split(split: str | None) -> None:
        if split is None:
            return
        if split not in {"base", "novel"}:
            raise ValueError("split must be one of: None, 'base', 'novel'")

    @staticmethod
    def _validate_episode(episode: Episode) -> None:
        if episode.query_labels is None:
            raise ValueError("episode must define query labels")
        if episode.query_embeddings is None:
            raise ValueError("episode must define query embeddings")

        if episode.support_items is not None and episode.query_items is not None:
            leakage = set(episode.support_items).intersection(episode.query_items)
            if leakage:
                raise ValueError("support/query leakage detected in episode items")

    @staticmethod
    def _validate_episode_classes(
        episode: Episode,
        allowed_set: set[str] | None,
        split: str | None,
    ) -> None:
        if allowed_set is None:
            return
        class_names = episode.class_names or []
        invalid = sorted(set(class_names) - allowed_set)
        if invalid:
            split_name = split or "provided"
            raise ValueError(
                f"episode includes classes outside {split_name} split: {', '.join(invalid)}"
            )

    @staticmethod
    def _predict_episode(learner: Any, episode: Episode):
        if hasattr(learner, "predict_episode"):
            predictions = learner.predict_episode(episode)
        elif hasattr(learner, "predict"):
            predictions = learner.predict(episode.query_embeddings)
        else:
            raise ValueError("learner must implement predict or predict_episode")
        return torch.as_tensor(predictions, dtype=torch.long)

    @staticmethod
    def _class_name_for_label(episode: Episode, label_idx: int) -> str:
        class_names = episode.class_names or []
        if 0 <= label_idx < len(class_names):
            return str(class_names[label_idx])
        return str(label_idx)

    def _confidence_interval_half_width(self, values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.0

        data = torch.as_tensor(values, dtype=torch.float64)
        std = float(torch.std(data, unbiased=True).item())
        sem = std / (len(values) ** 0.5)

        scipy_stats = importlib.import_module("scipy.stats")
        alpha = 1.0 - self.confidence_level
        t_critical = float(scipy_stats.t.ppf(1.0 - alpha / 2.0, df=len(values) - 1))
        return t_critical * sem
