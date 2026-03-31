import importlib
import math
from typing import Any, Sequence


torch = importlib.import_module("torch")
nn = importlib.import_module("torch.nn")


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) linear layer wrapper.

    Implements LoRA parameter-efficient fine-tuning by decomposing weight updates
    into low-rank matrices A and B. The effective weight update is BA^T scaled by
    alpha/rank, applied to the frozen base layer.

    Args:
        base_layer: The nn.Linear layer to wrap with LoRA
        rank: Rank of the low-rank decomposition (controls parameter count)
        alpha: Scaling factor for LoRA updates (typically rank or 2*rank)
        dropout: Dropout probability applied to LoRA path (0.0 to disable)

    Raises:
        TypeError: If base_layer is not nn.Linear
        ValueError: If rank <= 0 or dropout not in [0, 1)

    Attributes:
        base_layer: Frozen base linear layer
        lora_A: Learnable low-rank matrix A (shape: rank x in_features)
        lora_B: Learnable low-rank matrix B (shape: out_features x rank)
        scaling: Scaling factor alpha / rank
    """

    def __init__(self, base_layer: Any, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear only supports nn.Linear base layers")
        if rank <= 0:
            raise ValueError("rank must be a positive integer")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(self.rank, self.base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.base_layer.out_features, self.rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: Any) -> Any:
        """Forward pass combining base layer and LoRA adaptation.

        Computes: output = base_layer(x) + dropout(x) @ A^T @ B^T * scaling

        Args:
            x: Input tensor of shape (batch, in_features)

        Returns:
            Output tensor of shape (batch, out_features)
        """
        base = self.base_layer(x)
        dropped = self.dropout(x)
        lora = torch.matmul(dropped, self.lora_A.t())
        lora = torch.matmul(lora, self.lora_B.t())
        return base + (lora * self.scaling)


def count_trainable_parameters(module: Any) -> int:
    """Count the number of trainable parameters in a module.

    Args:
        module: PyTorch module to count parameters from

    Returns:
        Total count of trainable (requires_grad=True) parameters
    """
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


def _resolve_parent_module(root: Any, module_name: str) -> tuple[Any, str]:
    parts = module_name.split(".")
    if len(parts) == 1:
        return root, parts[0]
    parent = root.get_submodule(".".join(parts[:-1]))
    return parent, parts[-1]


def _resolve_lora_scope(model: Any) -> tuple[Any, str]:
    encoder_model = getattr(model, "model", model)
    image_branch = getattr(encoder_model, "visual", None)
    if image_branch is not None:
        return image_branch, "image"
    return encoder_model, "model"


def select_lora_target_layers(
    model: Any,
    target_modules: Sequence[str] | None = None,
) -> list[str]:
    """Select nn.Linear layers in model matching target module patterns.

    Filters model's named_modules to find Linear layers that match
    specified target module name patterns (e.g., 'attn', 'mlp', 'proj').

    Args:
        model: PyTorch model to search for target layers
        target_modules: Module name substrings to match (None = all Linear layers)

    Returns:
        List of fully qualified layer names matching criteria
    """
    selected: list[str] = []
    for name, layer in model.named_modules():
        if not name:
            continue
        if not isinstance(layer, nn.Linear):
            continue
        if target_modules and not any(token in name for token in target_modules):
            continue
        selected.append(name)
    return selected


def apply_lora_to_model(
    model: Any,
    enabled: bool = False,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Sequence[str] | None = None,
    freeze_non_lora: bool = True,
) -> dict[str, Any]:
    """Apply LoRA adaptation to specified layers in a model.

    Wraps matching Linear layers with LoRALinear and optionally freezes
    all other parameters. Resolves target scope (image branch vs full model).

    Args:
        model: Model to apply LoRA to (typically encoder.model or encoder.model.visual)
        enabled: Whether to actually apply LoRA (False returns no-op result)
        rank: LoRA rank (controls adapter parameter count)
        alpha: LoRA scaling factor
        dropout: Dropout rate for LoRA path
        target_modules: Module name patterns to target (e.g., ['attn', 'mlp'])
        freeze_non_lora: If True, freeze all non-LoRA parameters

    Returns:
        Dict with keys: enabled, applied (count), targets (list), scope ('image' or 'model')

    Raises:
        ValueError: If no layers match target_modules when enabled=True
    """
    target_model, scope = _resolve_lora_scope(model)

    if not enabled:
        return {
            "enabled": False,
            "applied": 0,
            "targets": [],
            "scope": scope,
        }

    if freeze_non_lora:
        for param in target_model.parameters():
            param.requires_grad = False

    targets = select_lora_target_layers(model=target_model, target_modules=target_modules)
    if not targets:
        requested = list(target_modules) if target_modules else []
        raise ValueError(f"No LoRA target layers matched the requested target_modules: {requested}")

    for name in targets:
        parent, attr_name = _resolve_parent_module(target_model, name)
        base_layer = getattr(parent, attr_name)
        setattr(
            parent,
            attr_name,
            LoRALinear(base_layer=base_layer, rank=rank, alpha=alpha, dropout=dropout),
        )

    return {
        "enabled": True,
        "applied": len(targets),
        "targets": targets,
        "scope": scope,
    }


def apply_image_encoder_lora(
    encoder: Any,
    enabled: bool = False,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Sequence[str] | None = None,
    freeze_non_lora: bool = True,
) -> dict[str, Any]:
    """Apply LoRA specifically to an image encoder.

    Convenience wrapper around apply_lora_to_model that forces scope="image"
    for image encoder adapters. Applies LoRA to the visual branch of the encoder.

    Args:
        encoder: Image encoder adapter (e.g., CLIPImageEncoderAdapter)
        enabled: Whether to apply LoRA adaptation
        rank: LoRA rank parameter
        alpha: LoRA scaling factor
        dropout: Dropout rate for LoRA path
        target_modules: Module name patterns to target
        freeze_non_lora: Freeze non-LoRA parameters if True

    Returns:
        Dict with keys: enabled, applied, targets, scope (always 'image')
    """
    result = apply_lora_to_model(
        model=encoder,
        enabled=enabled,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        freeze_non_lora=freeze_non_lora,
    )
    result["scope"] = "image"
    return result


__all__ = [
    "LoRALinear",
    "apply_lora_to_model",
    "apply_image_encoder_lora",
    "count_trainable_parameters",
    "select_lora_target_layers",
]
