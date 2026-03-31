import importlib


datasets = importlib.import_module("fewshoter.data.datasets")
samplers = importlib.import_module("fewshoter.data.samplers")

EpisodeDataset = datasets.EpisodeDataset
DatasetSplitter = datasets.DatasetSplitter
EpisodeSampler = samplers.EpisodeSampler


def _class_to_samples():
    return {
        "cat": ["cat_1", "cat_2", "cat_3", "cat_4", "cat_5", "cat_6"],
        "dog": ["dog_1", "dog_2", "dog_3", "dog_4", "dog_5", "dog_6"],
        "fox": ["fox_1", "fox_2", "fox_3", "fox_4", "fox_5", "fox_6"],
        "owl": ["owl_1", "owl_2", "owl_3", "owl_4", "owl_5", "owl_6"],
    }


def test_episode_dataset_is_indexable_and_deterministic():
    sampler = EpisodeSampler(_class_to_samples())
    dataset = EpisodeDataset(
        sampler,
        n_way=2,
        k_shot=2,
        num_queries=1,
        num_episodes=8,
        base_seed=101,
    )

    episode_a = dataset[3]
    episode_b = dataset[3]

    assert len(dataset) == 8
    assert episode_a.class_names == episode_b.class_names
    assert episode_a.support_items == episode_b.support_items
    assert episode_a.query_items == episode_b.query_items


def test_episode_dataset_iterator_respects_num_episodes():
    sampler = EpisodeSampler(_class_to_samples())
    dataset = EpisodeDataset(
        sampler,
        n_way=2,
        k_shot=1,
        num_queries=1,
        num_episodes=3,
        base_seed=0,
    )

    episodes = list(dataset)

    assert len(episodes) == 3
    assert all(len(ep.class_names) == 2 for ep in episodes)


def test_dataset_splitter_builds_working_split_samplers():
    splitter = DatasetSplitter(
        _class_to_samples(), train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=7
    )
    split_map = splitter.split()

    assert set(split_map.keys()) == {"train", "val", "test"}
    assert sum(len(v) for v in split_map.values()) == 4

    split_samplers = splitter.create_episode_samplers()
    assert set(split_samplers.keys()) == {"train", "val", "test"}

    train_classes = split_map["train"]
    if len(train_classes) >= 1:
        train_sampler = split_samplers["train"]
        episode = train_sampler.sample_episode(n_way=1, k_shot=1, num_queries=1, seed=5)
        assert set(episode.class_names).issubset(set(train_classes.keys()))
