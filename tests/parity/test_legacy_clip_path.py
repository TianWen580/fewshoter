import inspect
from importlib import import_module
from pathlib import Path
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


def test_legacy_multiscale_extractor_wraps_image_adapter() -> None:
    feature_extractor_module = import_module("fewshoter.features.feature_extractor")
    modalities_image_module = import_module("fewshoter.modalities.image")

    assert issubclass(
        feature_extractor_module.MultiScaleFeatureExtractor,
        modalities_image_module.CLIPImageEncoderAdapter,
    )


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


def _write_test_image(path: Path, seed: int) -> None:
    np = import_module("numpy")
    image_module = import_module("PIL.Image")

    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    image = image_module.fromarray(pixels, mode="RGB")
    image.save(path)


def _load_real_extractors(model_name: str):
    pytest = import_module("pytest")
    feature_extractor_module = import_module("fewshoter.features.feature_extractor")
    modalities_image_module = import_module("fewshoter.modalities.image")

    kwargs = {
        "model_name": model_name,
        "device": "cpu",
        "cache_features": False,
        "cache_backend": "memory",
        "cache_dir": "cache/test_clip_parity",
    }

    try:
        legacy = feature_extractor_module.MultiScaleFeatureExtractor(**kwargs)
        adapter = modalities_image_module.CLIPImageEncoderAdapter(**kwargs)
    except Exception as exc:
        pytest.skip(f"Real CLIP model unavailable for parity test: {exc}")
        raise AssertionError("unreachable after pytest.skip")

    return legacy, adapter


def _collect_labeled_embeddings(manager) -> list[tuple[str, object]]:
    np = import_module("numpy")

    labeled: list[tuple[str, object]] = []
    for class_name in sorted(manager.class_names):
        prototype = manager.prototypes[class_name]["global"]
        labeled.append((class_name, np.asarray(prototype)))
    return labeled


def test_legacy_clip_and_adapter_produce_matching_embeddings_and_labels(tmp_path) -> None:
    torch = import_module("torch")
    config_module = import_module("fewshoter.core.config")
    support_module = import_module("fewshoter.data.support_manager")

    config = config_module.Config()
    config.model.clip_model_name = "ViT-B/32"
    config.model.device = "cpu"
    config.model.cache_features = False
    config.support_set.save_prototypes = False
    config.support_set.save_embeddings = False

    legacy_extractor, adapter = _load_real_extractors(config.model.clip_model_name)

    class_names = ["class_alpha", "class_beta"]
    image_paths: list[Path] = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = tmp_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for shot_idx in range(2):
            img_path = class_dir / f"shot_{shot_idx}.png"
            _write_test_image(img_path, seed=class_idx * 100 + shot_idx)
            image_paths.append(img_path)

    legacy_encoded = legacy_extractor.encode(image_paths)["global"].detach().cpu()
    adapter_encoded = adapter.encode(image_paths)["global"].detach().cpu()

    assert legacy_encoded.shape == adapter_encoded.shape
    assert torch.allclose(legacy_encoded, adapter_encoded, atol=1e-6, rtol=1e-5)

    support_set_manager = support_module.SupportSetManager
    legacy_manager = support_set_manager(
        support_dir=tmp_path,
        feature_extractor=legacy_extractor,
        config=config,
    )
    adapter_manager = support_set_manager(
        support_dir=tmp_path,
        feature_extractor=adapter,
        config=config,
    )

    assert sorted(legacy_manager.class_names) == sorted(adapter_manager.class_names)

    legacy_labeled = _collect_labeled_embeddings(legacy_manager)
    adapter_labeled = _collect_labeled_embeddings(adapter_manager)

    assert [label for label, _ in legacy_labeled] == [label for label, _ in adapter_labeled]
    for (_, legacy_embedding), (_, adapter_embedding) in zip(legacy_labeled, adapter_labeled):
        assert torch.allclose(
            torch.as_tensor(legacy_embedding),
            torch.as_tensor(adapter_embedding),
            atol=1e-6,
            rtol=1e-5,
        )
