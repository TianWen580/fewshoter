import importlib
import unittest

samplers = importlib.import_module("fewshoter.data.samplers")
EpisodeSampler = samplers.EpisodeSampler


def _build_sampler():
    return EpisodeSampler(
        {
            "cat": ["cat_1", "cat_2", "cat_3", "cat_4", "cat_5", "cat_6"],
            "dog": ["dog_1", "dog_2", "dog_3", "dog_4", "dog_5", "dog_6"],
            "fox": ["fox_1", "fox_2", "fox_3", "fox_4", "fox_5", "fox_6"],
            "owl": ["owl_1", "owl_2", "owl_3", "owl_4", "owl_5", "owl_6"],
        },
        base_classes=["cat", "dog"],
        novel_classes=["fox", "owl"],
    )


def test_episode_sampler_is_deterministic_and_split_aware():
    sampler = _build_sampler()

    episode_a = sampler.sample_episode(split="base", n_way=2, k_shot=2, num_queries=1, seed=11)
    episode_b = sampler.sample_episode(split="base", n_way=2, k_shot=2, num_queries=1, seed=11)

    assert episode_a.class_names == episode_b.class_names
    assert episode_a.support_items == episode_b.support_items
    assert episode_a.query_items == episode_b.query_items
    assert set(episode_a.class_names).issubset({"cat", "dog"})
    assert set(episode_a.support_items).isdisjoint(set(episode_a.query_items))


def test_episode_sampler_rejects_too_many_classes():
    sampler = _build_sampler()

    with unittest.TestCase().assertRaisesRegex(
        ValueError, "requested 3-way episode but only 2 classes are available"
    ):
        sampler.sample_episode(split="base", n_way=3, k_shot=1, num_queries=1, seed=2)


def test_episode_sampler_rejects_insufficient_shots():
    sampler = EpisodeSampler(
        {"cat": ["cat_1", "cat_2", "cat_3", "cat_4"]},
        base_classes=["cat"],
    )

    with unittest.TestCase().assertRaisesRegex(
        ValueError, "Class 'cat' has 4 items, needs 5 for 2-shot 3-query episodes"
    ):
        sampler.sample_episode(split="base", n_way=1, k_shot=2, num_queries=3, seed=0)
