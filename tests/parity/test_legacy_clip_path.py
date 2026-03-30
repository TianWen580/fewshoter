import inspect
from importlib import import_module
from typing import cast


def test_fine_grained_classifier_constructor_signature() -> None:
    classifier_module = import_module("fewshoter.engine.classifier")
    fine_grained_classifier = classifier_module.FineGrainedClassifier
    signature = inspect.signature(fine_grained_classifier.__init__)

    assert "support_manager" in signature.parameters
    assert "config" in signature.parameters
    assert cast(object, signature.parameters["config"].default) is None


def test_support_set_manager_constructor_signature() -> None:
    support_module = import_module("fewshoter.data.support_manager")
    support_set_manager = support_module.SupportSetManager
    signature = inspect.signature(support_set_manager.__init__)

    assert "support_dir" in signature.parameters
    assert "feature_extractor" in signature.parameters
    assert "config" in signature.parameters
    assert cast(object, signature.parameters["support_dir"].default) is None
    assert cast(object, signature.parameters["feature_extractor"].default) is None
    assert cast(object, signature.parameters["config"].default) is None


def test_create_feature_extractor_accepts_config(monkeypatch) -> None:
    feature_extractor_module = import_module("fewshoter.features.feature_extractor")
    config_module = import_module("fewshoter.core.config")
    config_cls = config_module.Config

    captured: dict[str, object] = {}

    class DummyExtractor:
        def __init__(
            self,
            model_name: str,
            device: str,
            cache_features: bool,
            cache_backend: str,
            cache_dir: str,
        ) -> None:
            captured["model_name"] = model_name
            captured["device"] = device
            captured["cache_features"] = cache_features
            captured["cache_backend"] = cache_backend
            captured["cache_dir"] = cache_dir

    monkeypatch.setattr(feature_extractor_module, "MultiScaleFeatureExtractor", DummyExtractor)

    config = config_cls()
    extractor = feature_extractor_module.create_feature_extractor(config)

    assert isinstance(extractor, DummyExtractor)
    assert captured["model_name"] == config.model.clip_model_name


def test_create_classifier_accepts_support_manager_and_config(
    monkeypatch,
) -> None:
    classifier_module = import_module("fewshoter.engine.classifier")
    support_module = import_module("fewshoter.data.support_manager")
    config_module = import_module("fewshoter.core.config")

    support_set_manager_cls = support_module.SupportSetManager
    create_classifier = classifier_module.create_classifier
    config_cls = config_module.Config

    captured: dict[str, object] = {}

    class DummyClassifier:
        def __init__(self, support_manager: object, config: object) -> None:
            captured["support_manager"] = support_manager
            captured["config"] = config

    support_manager = support_set_manager_cls.__new__(support_set_manager_cls)
    config = config_cls()

    monkeypatch.setattr(classifier_module, "FineGrainedClassifier", DummyClassifier)

    classifier = create_classifier(support_manager=support_manager, config=config)

    assert isinstance(classifier, DummyClassifier)
    assert captured["support_manager"] is support_manager
    assert captured["config"] is config
