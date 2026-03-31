import importlib

Config = importlib.import_module("fewshoter.core.config").Config


def test_legacy_config_dict_still_loads_with_new_sections_present():
    legacy = {
        "model": {"clip_model_name": "ViT-B/16", "cache_backend": "memory"},
        "classification": {"visual_weight": 0.6, "text_weight": 0.4},
        "support_set": {"min_shots_per_class": 1, "max_shots_per_class": 5},
        "random_seed": 7,
    }

    cfg = Config.from_dict(legacy)

    assert cfg.model.clip_model_name == "ViT-B/16"
    assert cfg.classification.visual_weight == 0.6
    assert cfg.classification.text_weight == 0.4
    assert cfg.random_seed == 7
    assert cfg.episode.n_way == 5
    assert cfg.encoder.modality == "image"


def test_new_episode_and_encoder_sections_load_from_dict():
    data = {
        "episode": {"n_way": 8, "k_shot": 2, "num_queries": 3, "seed": 9},
        "encoder": {
            "modality": "image",
            "encoder_name": "clip",
            "embedding_dim": 512,
            "normalize_embeddings": True,
            "output_key": "global",
        },
    }

    cfg = Config.from_dict(data)

    assert cfg.episode.n_way == 8
    assert cfg.episode.k_shot == 2
    assert cfg.episode.num_queries == 3
    assert cfg.encoder.embedding_dim == 512
    assert cfg.encoder.output_key == "global"


def test_to_dict_includes_new_sections():
    cfg = Config()
    data = cfg.to_dict()

    assert "episode" in data
    assert "encoder" in data
    assert data["episode"]["n_way"] == cfg.episode.n_way
    assert data["encoder"]["modality"] == cfg.encoder.modality
