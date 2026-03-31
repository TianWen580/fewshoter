from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Iterable

from ..core.episode import Episode
from ..learners.base import LearnerContract


@dataclass
class Metrics:
    """Standard metric payload returned by evaluators."""

    accuracy: float
    confidence_interval: float
    per_class_accuracy: Dict[str, float]


class EvaluatorContract(ABC):
    """Abstract interface for episodic evaluators."""

    @classmethod
    def __subclasshook__(cls, C):
        """Support structural typing for evaluator-like classes.

        Args:
            C: Candidate class to check for an ``evaluate`` implementation.

        Returns:
            bool | NotImplemented: ``True`` when class satisfies the contract,
            otherwise ``NotImplemented``.
        """
        if cls is EvaluatorContract and any("evaluate" in B.__dict__ for B in C.__mro__):
            return True
        return NotImplemented

    @abstractmethod
    def evaluate(self, learner: LearnerContract, episodes: Iterable[Episode]) -> Metrics:
        """Evaluate a learner over a sequence of episodes.

        Args:
            learner: Learner to evaluate.
            episodes: Episodes to score.

        Returns:
            Metrics: Aggregate evaluation outcomes.
        """
        raise NotImplementedError
