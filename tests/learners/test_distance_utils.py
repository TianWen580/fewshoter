import importlib
import unittest


torch = importlib.import_module("torch")
utils = importlib.import_module("fewshoter.learners.utils")


def test_l2_normalize_returns_unit_length_vectors():
    embeddings = torch.tensor([[3.0, 4.0], [0.0, 5.0]])

    normalized = utils.l2_normalize(embeddings)
    norms = normalized.norm(dim=1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_squared_euclidean_distance_matches_manual_computation():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    distances = utils.squared_euclidean_distance(queries, prototypes)

    expected = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    assert torch.allclose(distances, expected)


def test_validate_episode_tensors_rejects_dimension_mismatch():
    support_embeddings = torch.randn(4, 8)
    support_labels = torch.tensor([0, 0, 1, 1])
    query_embeddings = torch.randn(3, 4)

    with unittest.TestCase().assertRaisesRegex(
        ValueError, "support and query embeddings must share embedding dimension"
    ):
        utils.validate_episode_tensors(
            support_embeddings=support_embeddings,
            support_labels=support_labels,
            query_embeddings=query_embeddings,
        )
