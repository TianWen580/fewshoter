import importlib
import unittest


torch = importlib.import_module("torch")
nn = importlib.import_module("torch.nn")
lora_module = importlib.import_module("fewshoter.peft.lora")

apply_image_encoder_lora = lora_module.apply_image_encoder_lora
apply_lora_to_model = lora_module.apply_lora_to_model
count_trainable_parameters = lora_module.count_trainable_parameters
select_lora_target_layers = lora_module.select_lora_target_layers
LoRALinear = lora_module.LoRALinear


class _ToyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_proj = nn.Linear(8, 4)
        self.mlp_fc = nn.Linear(8, 6)
        self.head = nn.Linear(8, 3)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _ToyVisual()
        self.text = nn.Linear(8, 5)


class _ToyEncoder:
    def __init__(self):
        self.model = _ToyModel()


def test_target_layer_selection_filters_by_name_tokens():
    model = _ToyVisual()

    selected = select_lora_target_layers(model, target_modules=["attn", "mlp"])

    assert "attn_proj" in selected
    assert "mlp_fc" in selected
    assert "head" not in selected


def test_trainable_parameter_count_matches_selected_lora_layers():
    encoder = _ToyEncoder()
    rank = 2

    report = apply_image_encoder_lora(
        encoder,
        enabled=True,
        rank=rank,
        alpha=8.0,
        target_modules=["attn", "mlp"],
        freeze_non_lora=True,
    )

    assert report["enabled"] is True
    assert sorted(report["targets"]) == ["attn_proj", "mlp_fc"]
    assert report["applied"] == 2

    expected = rank * (8 + 4) + rank * (8 + 6)
    assert count_trainable_parameters(encoder.model.visual) == expected

    assert isinstance(encoder.model.visual.attn_proj, LoRALinear)
    assert isinstance(encoder.model.visual.mlp_fc, LoRALinear)
    assert isinstance(encoder.model.text, nn.Linear)


def test_disabled_lora_path_is_noop():
    encoder = _ToyEncoder()
    before_layer = encoder.model.visual.attn_proj
    before_trainable = count_trainable_parameters(encoder.model.visual)

    report = apply_image_encoder_lora(
        encoder,
        enabled=False,
        rank=4,
        alpha=16.0,
        target_modules=["attn"],
        freeze_non_lora=True,
    )

    assert report == {"enabled": False, "applied": 0, "targets": [], "scope": "image"}
    assert encoder.model.visual.attn_proj is before_layer
    assert count_trainable_parameters(encoder.model.visual) == before_trainable


def test_enabled_lora_fails_when_no_target_layers_match():
    model = _ToyVisual()

    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "No LoRA target layers matched the requested target_modules",
    ):
        apply_lora_to_model(
            model=model,
            enabled=True,
            rank=2,
            alpha=8.0,
            target_modules=["does_not_exist"],
            freeze_non_lora=True,
        )


def test_lora_linear_rejects_invalid_dropout_values():
    base_layer = nn.Linear(8, 4)

    with unittest.TestCase().assertRaisesRegex(ValueError, r"dropout must be in \[0, 1\)"):
        LoRALinear(base_layer=base_layer, rank=2, alpha=8.0, dropout=-0.1)

    with unittest.TestCase().assertRaisesRegex(ValueError, r"dropout must be in \[0, 1\)"):
        LoRALinear(base_layer=base_layer, rank=2, alpha=8.0, dropout=1.0)
