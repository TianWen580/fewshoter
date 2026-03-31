import importlib
import unittest


torch = importlib.import_module("torch")
episode_module = importlib.import_module("fewshoter.core.episode")
episodic_module = importlib.import_module("fewshoter.evaluation.episodic")

Episode = episode_module.Episode
EpisodicEvaluator = episodic_module.EpisodicEvaluator


def _episode(seed: int, support_items=None, query_items=None, class_names=None):
    names = class_names if class_names is not None else ["cat", "dog"]
    return Episode(
        support_embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        support_labels=torch.tensor([0, 1], dtype=torch.long),
        query_embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        query_labels=torch.tensor([0, 1], dtype=torch.long),
        class_names=names,
        n_way=2,
        k_shot=1,
        num_queries=2,
        seed=seed,
        support_items=support_items,
        query_items=query_items,
    )


class SeedAwareLearner:
    def predict_episode(self, episode: Episode):
        labels = torch.as_tensor(episode.query_labels, dtype=torch.long)
        if int(episode.seed or 0) % 2 == 0:
            return labels
        return 1 - labels


class PerfectLearner:
    def predict_episode(self, episode: Episode):
        return torch.as_tensor(episode.query_labels, dtype=torch.long)


class AlwaysZeroLearner:
    def predict_episode(self, episode: Episode):
        labels = torch.as_tensor(episode.query_labels, dtype=torch.long)
        return torch.zeros_like(labels)


def test_episodic_evaluator_returns_mean_accuracy_and_confidence_interval():
    evaluator = EpisodicEvaluator(confidence_level=0.95)
    metrics = evaluator.evaluate(SeedAwareLearner(), [_episode(seed=0), _episode(seed=1)])

    assert metrics.accuracy == 0.5
    assert metrics.confidence_interval > 0.0
    assert set(metrics.per_class_accuracy.keys()) == {"cat", "dog"}


def test_episodic_evaluator_compare_supports_legacy_vs_prototypical():
    evaluator = EpisodicEvaluator()
    comparison = evaluator.compare(
        legacy_learner=AlwaysZeroLearner(),
        prototypical_learner=PerfectLearner(),
        episodes=[_episode(seed=0), _episode(seed=2)],
        split="base",
        allowed_classes=["cat", "dog"],
    )

    assert comparison.legacy.accuracy == 0.5
    assert comparison.prototypical.accuracy == 1.0
    assert comparison.accuracy_delta == 0.5


def test_episodic_evaluator_rejects_support_query_leakage():
    evaluator = EpisodicEvaluator()
    leaked = _episode(seed=0, support_items=["x", "y"], query_items=["x", "z"])

    with unittest.TestCase().assertRaisesRegex(
        ValueError, "support/query leakage detected in episode items"
    ):
        evaluator.evaluate(PerfectLearner(), [leaked])


def test_episodic_evaluator_rejects_invalid_split_name():
    evaluator = EpisodicEvaluator()

    with unittest.TestCase().assertRaisesRegex(
        ValueError, "split must be one of: None, 'base', 'novel'"
    ):
        evaluator.evaluate(PerfectLearner(), [_episode(seed=0)], split="train")


def test_episodic_evaluator_rejects_classes_outside_allowed_split():
    evaluator = EpisodicEvaluator()
    episode = _episode(seed=0, class_names=["cat", "otter"])

    with unittest.TestCase().assertRaisesRegex(
        ValueError, "episode includes classes outside base split"
    ):
        evaluator.evaluate(
            PerfectLearner(),
            [episode],
            split="base",
            allowed_classes=["cat", "dog"],
        )
