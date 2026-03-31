import importlib
import json


train_cli = importlib.import_module("fewshoter.cli.train")
evaluate_cli = importlib.import_module("fewshoter.cli.evaluate")
episode_module = importlib.import_module("fewshoter.core.episode")
metrics_module = importlib.import_module("fewshoter.evaluation.base")

Episode = episode_module.Episode
Metrics = metrics_module.Metrics


def test_train_cli_keeps_legacy_defaults_with_additive_episodic_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["fewshoter-train", "--support_dir", "dummy_support"],
    )

    args = train_cli.parse_arguments()

    assert args.evaluation_mode == "legacy"
    assert args.num_episodes == 100
    assert args.episode_n_way == 5
    assert args.episode_k_shot == 1
    assert args.episode_num_queries == 5
    assert args.episode_split == "none"
    assert args.max_shots == 5


def test_train_cli_parses_episodic_additive_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "fewshoter-train",
            "--support_dir",
            "dummy_support",
            "--evaluation_mode",
            "episodic",
            "--num_episodes",
            "20",
            "--episode_n_way",
            "3",
            "--episode_k_shot",
            "2",
            "--episode_num_queries",
            "4",
            "--episode_split",
            "novel",
            "--confidence_level",
            "0.9",
        ],
    )

    args = train_cli.parse_arguments()

    assert args.evaluation_mode == "episodic"
    assert args.num_episodes == 20
    assert args.episode_n_way == 3
    assert args.episode_k_shot == 2
    assert args.episode_num_queries == 4
    assert args.episode_split == "novel"
    assert args.confidence_level == 0.9


def test_evaluate_cli_keeps_legacy_defaults_with_additive_episodic_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "fewshoter-evaluate",
            "--prototypes",
            "dummy_prototypes.json",
            "--test_dir",
            "dummy_test",
        ],
    )

    args = evaluate_cli.parse_arguments()

    assert args.evaluation_mode == "legacy"
    assert args.num_episodes == 100
    assert args.episode_n_way == 5
    assert args.episode_k_shot == 1
    assert args.episode_num_queries == 5
    assert args.episode_split == "none"
    assert args.visual_weight == 0.7
    assert args.text_weight == 0.3


def test_evaluate_cli_parses_compare_mode_and_episode_split(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "fewshoter-evaluate",
            "--prototypes",
            "dummy_prototypes.json",
            "--test_dir",
            "dummy_test",
            "--evaluation_mode",
            "compare",
            "--episode_split",
            "base",
            "--num_episodes",
            "50",
            "--confidence_level",
            "0.99",
        ],
    )

    args = evaluate_cli.parse_arguments()

    assert args.evaluation_mode == "compare"
    assert args.episode_split == "base"
    assert args.num_episodes == 50
    assert args.confidence_level == 0.99


def _write_minimal_inputs(tmp_path):
    prototypes = tmp_path / "prototypes.json"
    prototypes.write_text(json.dumps({"config": {}}), encoding="utf-8")

    test_dir = tmp_path / "test"
    (test_dir / "cat").mkdir(parents=True)
    (test_dir / "dog").mkdir(parents=True)
    return prototypes, test_dir


def _patch_runtime_dependencies(monkeypatch):
    class DummyFeatureExtractor:
        def encode(self, paths):
            torch = importlib.import_module("torch")
            return {"global": torch.ones((len(paths), 2), dtype=torch.float32)}

    class DummySupportManager:
        class_names = ["cat", "dog"]

        @classmethod
        def from_prototypes(cls, _prototypes, _feature_extractor):
            return cls()

    class DummyClassifier:
        def __init__(self):
            self.classify_batch_calls = 0

        def classify_batch(self, image_paths, **_kwargs):
            self.classify_batch_calls += 1
            return [{"predicted_class": "cat", "confidence": 0.9} for _ in image_paths]

    classifier = DummyClassifier()
    monkeypatch.setattr(
        evaluate_cli, "create_feature_extractor", lambda _config: DummyFeatureExtractor()
    )
    monkeypatch.setattr(evaluate_cli, "SupportSetManager", DummySupportManager)
    monkeypatch.setattr(evaluate_cli, "create_classifier", lambda *_args, **_kwargs: classifier)
    monkeypatch.setattr(
        evaluate_cli,
        "load_test_dataset",
        lambda _test_dir: (["a.jpg", "b.jpg"], ["cat", "dog"], ["cat", "dog"]),
    )
    return classifier


def test_evaluate_main_legacy_mode_uses_legacy_classifier_path(monkeypatch, tmp_path):
    prototypes, test_dir = _write_minimal_inputs(tmp_path)
    classifier = _patch_runtime_dependencies(monkeypatch)

    episodic_used = {"value": False}

    class FailingEpisodicEvaluator:
        def __init__(self, *args, **kwargs):
            episodic_used["value"] = True
            raise AssertionError("episodic evaluator should not be used in legacy mode")

    monkeypatch.setattr(evaluate_cli, "EpisodicEvaluator", FailingEpisodicEvaluator)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fewshoter-evaluate",
            "--prototypes",
            str(prototypes),
            "--test_dir",
            str(test_dir),
            "--evaluation_mode",
            "legacy",
        ],
    )

    evaluate_cli.main()

    assert classifier.classify_batch_calls == 1
    assert episodic_used["value"] is False


def test_evaluate_main_episodic_mode_uses_episodic_evaluator(monkeypatch, tmp_path):
    prototypes, test_dir = _write_minimal_inputs(tmp_path)
    classifier = _patch_runtime_dependencies(monkeypatch)

    episode = Episode(
        support_embeddings=None,
        support_labels=[0, 1],
        query_embeddings=None,
        query_labels=[0, 1],
        class_names=["cat", "dog"],
        n_way=2,
        k_shot=1,
        num_queries=1,
        support_items=["s1", "s2"],
        query_items=["q1", "q2"],
    )
    monkeypatch.setattr(
        evaluate_cli, "_build_episodic_episodes", lambda *_: ([episode], ["cat", "dog"])
    )
    monkeypatch.setattr(evaluate_cli, "_materialize_episode_embeddings", lambda *_: None)

    calls = {"evaluate": 0, "compare": 0}

    class DummyEpisodicEvaluator:
        def __init__(self, confidence_level):
            self.confidence_level = confidence_level

        def evaluate(self, learner, episodes, **kwargs):
            _ = (learner, episodes, kwargs)
            calls["evaluate"] += 1
            return Metrics(accuracy=0.75, confidence_interval=0.1, per_class_accuracy={"cat": 1.0})

        def compare(self, **kwargs):
            _ = kwargs
            calls["compare"] += 1
            raise AssertionError("compare should not be called in episodic mode")

    monkeypatch.setattr(evaluate_cli, "EpisodicEvaluator", DummyEpisodicEvaluator)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fewshoter-evaluate",
            "--prototypes",
            str(prototypes),
            "--test_dir",
            str(test_dir),
            "--evaluation_mode",
            "episodic",
        ],
    )

    evaluate_cli.main()

    assert calls["evaluate"] == 1
    assert calls["compare"] == 0
    assert classifier.classify_batch_calls == 0


def test_evaluate_main_compare_mode_uses_compare_path(monkeypatch, tmp_path):
    prototypes, test_dir = _write_minimal_inputs(tmp_path)
    classifier = _patch_runtime_dependencies(monkeypatch)

    episode = Episode(
        support_embeddings=None,
        support_labels=[0, 1],
        query_embeddings=None,
        query_labels=[0, 1],
        class_names=["cat", "dog"],
        n_way=2,
        k_shot=1,
        num_queries=1,
        support_items=["s1", "s2"],
        query_items=["q1", "q2"],
    )
    monkeypatch.setattr(
        evaluate_cli, "_build_episodic_episodes", lambda *_: ([episode], ["cat", "dog"])
    )
    monkeypatch.setattr(evaluate_cli, "_materialize_episode_embeddings", lambda *_: None)

    calls = {"evaluate": 0, "compare": 0}

    class DummyComparison:
        def __init__(self):
            self.legacy = Metrics(
                accuracy=0.4, confidence_interval=0.1, per_class_accuracy={"cat": 0.4}
            )
            self.prototypical = Metrics(
                accuracy=0.8, confidence_interval=0.05, per_class_accuracy={"cat": 0.8}
            )
            self.accuracy_delta = 0.4

    class DummyEpisodicEvaluator:
        def __init__(self, confidence_level):
            self.confidence_level = confidence_level

        def evaluate(self, *args, **kwargs):
            _ = (args, kwargs)
            calls["evaluate"] += 1
            raise AssertionError("evaluate should not be called in compare mode")

        def compare(self, **kwargs):
            _ = kwargs
            calls["compare"] += 1
            return DummyComparison()

    monkeypatch.setattr(evaluate_cli, "EpisodicEvaluator", DummyEpisodicEvaluator)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fewshoter-evaluate",
            "--prototypes",
            str(prototypes),
            "--test_dir",
            str(test_dir),
            "--evaluation_mode",
            "compare",
        ],
    )

    evaluate_cli.main()

    assert calls["evaluate"] == 0
    assert calls["compare"] == 1
    assert classifier.classify_batch_calls == 0
