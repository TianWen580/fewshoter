import importlib

pytest = importlib.import_module("pytest")


def _build_dummy_openclip_bundle(torch_module, nn_module):
    class DummyAttention(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention_weights = None

        def forward(self, x):
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            self.attention_weights = torch_module.ones(batch_size, 2, seq_len, seq_len)
            return x

    class DummyBlock(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = DummyAttention()

        def forward(self, x):
            return self.attn(x)

    class DummyTransformer(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.resblocks = nn_module.ModuleList([DummyBlock() for _ in range(12)])
            self.width = 4

    class DummyVisual(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = DummyTransformer()
            self.width = 4
            self.positional_embedding = torch_module.zeros(1, 17, 4)

    class DummyModel(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = DummyVisual()

        def encode_image(self, image_tensor):
            batch_size = image_tensor.shape[0]
            tokens = torch_module.ones(batch_size, 17, 4, device=image_tensor.device)
            for block in self.visual.transformer.resblocks:
                tokens = block(tokens)
            return torch_module.ones(batch_size, 4, device=image_tensor.device)

    def preprocess(image):
        _ = image
        return torch_module.zeros(3, 224, 224)

    return DummyModel, preprocess


def test_clip_image_encoder_adapter_emits_legacy_feature_keys(monkeypatch) -> None:
    modalities_image = importlib.import_module("fewshoter.modalities.image")
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")

    dummy_model_cls, preprocess = _build_dummy_openclip_bundle(torch, nn)

    def fake_create_model_and_transforms(arch, pretrained=None, device=None):
        _ = (arch, pretrained, device)
        return dummy_model_cls(), preprocess, preprocess

    monkeypatch.setattr(modalities_image.clip, "list_pretrained", lambda: [("ViT-B-32", "openai")])
    monkeypatch.setattr(
        modalities_image.clip,
        "create_model_and_transforms",
        fake_create_model_and_transforms,
    )

    encoder = modalities_image.CLIPImageEncoderAdapter(
        model_name="ViT-B/32",
        device="cpu",
        cache_features=False,
    )

    features = encoder.extract_features(
        torch.randn(1, 3, 224, 224),
        return_attention=True,
        normalize=False,
    )

    assert "global" in features
    assert "layer_9" in features
    assert any(key.startswith("attn_") for key in features)
    assert hasattr(encoder, "model")
    assert hasattr(encoder, "model_name")
    assert hasattr(encoder, "feature_dim")
    assert hasattr(encoder, "grid_size")
    assert callable(encoder.clear_cache)


def test_clip_image_encoder_adapter_invalid_model_raises_clear_error(monkeypatch) -> None:
    modalities_image = importlib.import_module("fewshoter.modalities.image")

    monkeypatch.setattr(modalities_image.clip, "list_pretrained", lambda: [("ViT-B-32", "openai")])

    with pytest.raises(ValueError, match="Invalid CLIP model configuration"):
        modalities_image.CLIPImageEncoderAdapter(
            model_name="totally-invalid-model",
            device="cpu",
            cache_features=False,
        )
