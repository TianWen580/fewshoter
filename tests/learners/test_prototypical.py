import importlib
import unittest


torch = importlib.import_module("torch")
episode_module = importlib.import_module("fewshoter.core.episode")
learner_module = importlib.import_module("fewshoter.learners.prototypical")

Episode = episode_module.Episode
PrototypicalLearner = learner_module.PrototypicalLearner


def _build_episode() -> Episode:
    return Episode(
        support_embeddings=torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]),
        support_labels=torch.tensor([0, 0, 1, 1]),
        query_embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]),
        query_labels=torch.tensor([0, 1, 0]),
        class_names=["a", "b"],
        n_way=2,
        k_shot=2,
        num_queries=3,
    )


def test_prototypical_predicts_expected_classes_from_episode_embeddings():
    episode = _build_episode()
    learner = PrototypicalLearner()

    learner.fit([episode])
    predictions = learner.predict(episode.query_embeddings)

    assert predictions.tolist() == [0, 1, 0]


def test_prototypical_predict_from_tensors_is_deterministic():
    episode = _build_episode()
    learner = PrototypicalLearner()

    first = learner.predict_from_tensors(
        support_embeddings=episode.support_embeddings,
        support_labels=episode.support_labels,
        query_embeddings=episode.query_embeddings,
    )
    second = learner.predict_from_tensors(
        support_embeddings=episode.support_embeddings,
        support_labels=episode.support_labels,
        query_embeddings=episode.query_embeddings,
    )

    assert torch.equal(first, second)
    assert first.tolist() == [0, 1, 0]


def test_dimension_mismatch_between_support_and_query_embeddings_raises_error():
    learner = PrototypicalLearner()
    support_embeddings = torch.randn(4, 6)
    support_labels = torch.tensor([0, 0, 1, 1])
    query_embeddings = torch.randn(2, 8)

    with unittest.TestCase().assertRaisesRegex(
        ValueError, "support and query embeddings must share embedding dimension"
    ):
        learner.predict_from_tensors(
            support_embeddings=support_embeddings,
            support_labels=support_labels,
            query_embeddings=query_embeddings,
        )
