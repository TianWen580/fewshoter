import importlib
import unittest

torch = importlib.import_module("torch")
episode_module = importlib.import_module("fewshoter.core.episode")
Episode = episode_module.Episode
EpisodeConfig = episode_module.EpisodeConfig


def test_episode_dataclass_roundtrip():
    episode = Episode(
        support_embeddings=torch.randn(6, 8),
        support_labels=torch.tensor([0, 0, 1, 1, 2, 2]),
        query_embeddings=torch.randn(9, 8),
        query_labels=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        class_names=["a", "b", "c"],
        n_way=3,
        k_shot=2,
    )

    assert episode.support_embeddings.shape == (6, 8)
    assert episode.query_labels.shape[0] == 9
    assert episode.class_names == ["a", "b", "c"]
    assert episode.n_way == 3
    assert episode.k_shot == 2


def test_episode_config_defaults_and_override():
    default_cfg = EpisodeConfig()
    custom_cfg = EpisodeConfig(n_way=7, k_shot=3, num_queries=4, seed=123)

    assert default_cfg.n_way == 5
    assert default_cfg.k_shot == 1
    assert custom_cfg.n_way == 7
    assert custom_cfg.k_shot == 3
    assert custom_cfg.num_queries == 4
    assert custom_cfg.seed == 123


def test_episode_contract_tracks_items_and_validates():
    episode = Episode(
        support_items=["cat_1", "cat_2"],
        support_labels=torch.tensor([0, 0]),
        query_items=["cat_3"],
        query_labels=torch.tensor([0]),
        class_names=["cat"],
        n_way=1,
        k_shot=2,
        num_queries=1,
        seed=7,
    )

    assert episode.support_items == ["cat_1", "cat_2"]
    assert episode.query_items == ["cat_3"]
    assert episode.num_queries == 1
    assert episode.seed == 7
    assert episode.validate() is episode


def test_episode_contract_rejects_mismatched_lengths():
    episode = Episode(
        support_items=["cat_1"],
        support_labels=torch.tensor([0, 0]),
        query_items=["cat_2"],
        query_labels=torch.tensor([0]),
        class_names=["cat"],
        n_way=1,
        k_shot=1,
        num_queries=1,
        seed=7,
    )

    with unittest.TestCase().assertRaisesRegex(
        ValueError, "support items and labels must have the same length"
    ):
        episode.validate()
