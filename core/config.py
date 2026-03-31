"""
Configuration management for CLIP Few-Shot Classification System

This module provides centralized configuration management for all system components,
including model parameters, feature extraction settings, and classification thresholds.
"""

import json
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .episode import EpisodeConfig


@dataclass
class ModelConfig:
    """Configuration for CLIP model and feature extraction"""

    clip_model_name: str = "ViT-B/32"
    device: str = "cuda"
    batch_size: int = 32
    image_size: int = 224
    feature_dim: int = 512
    use_fp16: bool = True
    cache_features: bool = True
    # Feature cache backend: 'memory' | 'disk' | 'mmap'
    cache_backend: str = "mmap"
    # Directory for on-disk feature cache
    cache_dir: str = "cache/clip_features"
    enable_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: ["attn", "mlp", "proj"])


@dataclass
class EncoderConfig:
    """Configuration for encoder/modality selection and embedding parameters.

    This dataclass configures which encoder to use (e.g., CLIP, Perch),
    the target modality (image, audio), and embedding normalization settings.

    Attributes:
        modality: Target modality for encoding ('image', 'audio', etc.)
        encoder_name: Name of the encoder architecture ('clip', 'perch', etc.)
        embedding_dim: Dimensionality of output embeddings
        normalize_embeddings: Whether to L2-normalize embeddings
        output_key: Key for selecting output feature type ('global', 'local', etc.)
        runtime_framework: Framework for runtime execution ('torch', 'onnx', etc.)
        checkpoint: Path to custom checkpoint file (empty for pretrained)
    """

    modality: str = "image"
    encoder_name: str = "clip"
    embedding_dim: int = 512
    normalize_embeddings: bool = True
    output_key: str = "global"
    runtime_framework: str = "torch"
    checkpoint: str = ""


@dataclass
class AudioConfig:
    """Configuration for audio processing and Perch 2.0 integration.

    This dataclass defines audio preprocessing parameters and integration
    settings for the Perch 2.0 bioacoustic encoder (offline placeholder mode).

    Attributes:
        enable_audio: Whether to enable audio modality processing
        sample_rate_hz: Target sample rate for audio resampling
        window_seconds: Duration of each analysis window in seconds
        window_hop_seconds: Hop size between consecutive windows
        channel_count: Number of audio channels (1 for mono)
        preprocessing_policy: Preprocessing pipeline identifier
        provenance_tag: Data source tag for tracking/auditing
        cache_namespace: Namespace for audio embedding cache storage
    """

    enable_audio: bool = False
    sample_rate_hz: int = 32000
    window_seconds: float = 5.0
    window_hop_seconds: float = 5.0
    channel_count: int = 1
    preprocessing_policy: str = "perch2-default"
    provenance_tag: str = "offline-placeholder"
    cache_namespace: str = "audio_embeddings"


@dataclass
class ClassificationConfig:
    """Configuration for classification parameters"""

    visual_weight: float = 0.7
    text_weight: float = 0.3
    confidence_threshold: float = 0.5
    max_text_probes: int = 5
    use_feature_alignment: bool = True
    alignment_strength: float = 0.1
    # Text calibration toggle
    enable_text_calibration: bool = True
    # Local patch enhancement
    local_patch_enhance: bool = True
    local_patch_weight: float = 0.3
    local_patch_topk: int = 5
    # Post-hoc logit calibration
    global_logit_temperature: Optional[float] = None
    enable_logit_bias: bool = True
    # Neighborhood Subspace Suppression (NSS)
    enable_nss: bool = False
    nss_num_neighbors: int = 3
    nss_proj_weight: float = 0.5
    nss_bias_weight: float = 0.05
    # Optional path to a Python file that defines a description dict for classes
    # If provided, and a class name exists in the dict, the description text will
    # be used as the text feature source instead of the class name/templates.
    desc_text_path: Optional[str] = None

    # Stage 1: Discriminative mining and subcenters
    enable_discriminative_mining: bool = True
    discriminative_token_topk_ratio: float = 0.15
    num_subcenters: int = 3
    discriminative_global_weight: float = 0.6
    discriminative_part_weight: float = 0.4

    # Stage 4: Impostor-aware reranking and calibration
    enable_impostor_rerank: bool = True
    impostor_rerank_strength: float = 0.5
    enable_per_class_temperature: bool = False
    rejection_enable: bool = False
    rejection_margin_threshold: float = 0.05

    # Score normalization across classes per query: 'none' | 'zscore' | 'mean_center'
    score_normalization: str = "none"

    # Open-set / unknown detection
    unknown_enable: bool = True
    unknown_strategy: str = "zscore"  # 'absolute' | 'margin' | 'softmax_p' | 'zscore'
    unknown_label: str = "UNKNOWN"
    unknown_abs_threshold: float = 0.25  # for 'absolute' strategy
    unknown_margin_threshold: float = (
        0.05  # for 'margin' strategy (fallback to rejection_margin_threshold)
    )
    unknown_pmax_threshold: float = 0.6  # for 'softmax_p' strategy
    unknown_z_threshold: float = 1.0  # for 'zscore' strategy
    softmax_temperature: float = 10.0  # temperature for softmax-based confidence

    # Dimensionality reduction (sklearn-driven). Default off.
    enable_dim_reduction: bool = False
    dim_reduction_method: str = "pca"  # currently supports: 'pca'
    dim_reduction_n_components: Optional[float] = (
        0.95  # float in (0,1] as variance ratio or int dims if set via config
    )
    dim_reduction_whiten: bool = False
    dim_reduction_random_state: Optional[int] = 42

    # SVM-based classifier settings (optional)
    use_svm: bool = False
    svm_kernel: str = "rbf"  # 'linear' | 'rbf' | 'poly' | 'sigmoid'
    svm_C: float = 1.0
    svm_gamma: Optional[str] = "scale"  # 'scale' | 'auto' | float
    svm_probability: bool = True
    svm_feature_layer: str = "global"
    svm_model_path: Optional[str] = None
    svm_infer_mode: str = "svm_only"  # 'svm_only' | 'hybrid'
    svm_hybrid_weight: float = 0.5  # weight of SVM when hybrid
    svm_use_class_weight: bool = True

    # Neural Network (MLP) classifier settings (optional)
    use_nn: bool = False
    nn_hidden_sizes: Optional[list[int]] = None  # e.g., [512] or [512, 256]
    nn_activation: str = "relu"  # 'relu' | 'gelu' | 'tanh'
    nn_dropout: float = 0.1
    nn_epochs: int = 20
    nn_lr: float = 1e-3
    nn_weight_decay: float = 1e-4
    nn_batch_size: int = 64
    nn_early_stopping_patience: int = 5
    nn_class_weight_balanced: bool = True
    nn_feature_layer: str = "global"
    nn_model_path: Optional[str] = None
    nn_infer_mode: str = "nn_only"  # 'nn_only' | 'hybrid'
    nn_hybrid_weight: float = 0.5  # weight of NN when hybrid

    # Masked-Global pooling (foreground-weighted global similarity)
    enable_masked_global: bool = False
    masked_global_weight: float = 0.5
    masked_global_layer: str = "layer_9"
    masked_global_source: str = "discriminative_mask"  # 'discriminative_mask' | 'attention'

    # Global subcenters on global features
    enable_global_subcenters: bool = False
    global_subcenters_k: int = 3
    global_subcenters_iters: int = 2
    global_subcenters_fusion: bool = True
    global_subcenters_weight: float = 0.5

    # Inference-time TTA
    tta_enable: bool = False
    tta_hflip: bool = True
    tta_num_scales: int = 1
    tta_scale_min: float = 0.9
    tta_scale_max: float = 1.1


@dataclass
class AdapterConfig:
    """Configuration for adapter-based few-shot enhancements (e.g., Tip-Adapter)"""

    use_tip_adapter: bool = True
    alpha: float = 0.5  # fusion weight for adapter vs text when adapter path is used
    beta: float = 50.0  # temperature for similarity -> weights in adapter

    # Optional small head (ArcFace/CosFace-like). Disabled by default.
    use_small_head: bool = False
    train_small_head: bool = False
    head_type: str = "arcface"  # or "cosface", "cosine"
    margin: float = 0.5
    scale: float = 30.0
    subcenters: int = 1
    train_epochs: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class SupportSetConfig:
    """Configuration for support set management"""

    min_shots_per_class: int = 1
    max_shots_per_class: int = 5
    prototype_update_rate: float = 0.1
    attribute_threshold: float = 0.5
    save_prototypes: bool = True
    prototype_cache_dir: str = "cache/prototypes"
    # Prototype refinement (EMA)
    enable_prototype_ema: bool = True
    prototype_ema_momentum: float = 0.05  # EMA momentum when updating with confident queries
    prototype_ema_min_conf: float = 0.8  # only update when confidence above threshold

    # Embedding cache settings
    save_embeddings: bool = True
    embedding_cache_filename: str = "embed.json"
    # Fast init: whether to load embed.json at init when using cached prototypes.
    # Default False to prioritize quick startup; will be loaded on-demand if needed.
    load_embeddings_on_init: bool = False

    # Support-set augmentation to enrich embeddings for prototypes/SVM/NN
    enable_support_augmentation: bool = False
    support_aug_hflip: bool = True
    support_aug_multiplier: int = 1


@dataclass
class VisualizationConfig:
    """Configuration for visualization and debugging"""

    save_attention_maps: bool = False
    save_saliency_maps: bool = False
    output_dir: str = "outputs"
    heatmap_alpha: float = 0.3
    colormap: str = "jet"
    # Calibration visualization
    save_calibration_curves: bool = False


@dataclass
class EvaluationConfig:
    """Configuration for episodic evaluation protocol.

    This dataclass controls the few-shot evaluation methodology,
    including number of episodes and confidence interval settings.

    Attributes:
        mode: Evaluation mode ('legacy', 'episodic', or 'compare')
        episodes: Number of episodes to run for episodic evaluation
        confidence_level: Confidence level for statistical intervals (0-1)
        split: Dataset split to evaluate ('base', 'novel', or None for all)
    """

    mode: str = "legacy"
    episodes: int = 100
    confidence_level: float = 0.95
    split: Optional[str] = None


@dataclass
class Config:
    """Main configuration class combining all sub-configurations"""

    model: ModelConfig = field(default_factory=ModelConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    support_set: SupportSetConfig = field(default_factory=SupportSetConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)

    # System settings
    random_seed: int = 42
    num_workers: int = 4
    log_level: str = "INFO"

    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from YAML or JSON file"""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            if config_file.suffix.lower() == ".yaml" or config_file.suffix.lower() == ".yml":
                data = yaml.safe_load(f)
            elif config_file.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file.suffix}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary"""
        config = cls()

        # Update model config
        if "model" in data:
            for key, value in data["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)

        if "encoder" in data:
            for key, value in data["encoder"].items():
                if hasattr(config.encoder, key):
                    setattr(config.encoder, key, value)

        if "audio" in data:
            for key, value in data["audio"].items():
                if hasattr(config.audio, key):
                    setattr(config.audio, key, value)

        if "episode" in data:
            for key, value in data["episode"].items():
                if hasattr(config.episode, key):
                    setattr(config.episode, key, value)

        # Update classification config
        if "classification" in data:
            for key, value in data["classification"].items():
                if hasattr(config.classification, key):
                    setattr(config.classification, key, value)

        if "evaluation" in data:
            for key, value in data["evaluation"].items():
                if hasattr(config.evaluation, key):
                    setattr(config.evaluation, key, value)

        # Update adapter config
        if "adapter" in data:
            for key, value in data["adapter"].items():
                if hasattr(config.adapter, key):
                    setattr(config.adapter, key, value)

        # Update support set config
        if "support_set" in data:
            for key, value in data["support_set"].items():
                if hasattr(config.support_set, key):
                    setattr(config.support_set, key, value)

        # Update visualization config
        if "visualization" in data:
            for key, value in data["visualization"].items():
                if hasattr(config.visualization, key):
                    setattr(config.visualization, key, value)

        # Update system settings
        for key in ["random_seed", "num_workers", "log_level"]:
            if key in data:
                setattr(config, key, data[key])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model": asdict(self.model),
            "encoder": asdict(self.encoder),
            "audio": asdict(self.audio),
            "episode": asdict(self.episode),
            "classification": asdict(self.classification),
            "evaluation": asdict(self.evaluation),
            "adapter": asdict(self.adapter),
            "support_set": asdict(self.support_set),
            "visualization": asdict(self.visualization),
            "random_seed": self.random_seed,
            "num_workers": self.num_workers,
            "log_level": self.log_level,
        }

    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a YAML or JSON file.

        Args:
            config_path: Output file path as string or ``Path``.

        Raises:
            ValueError: If file extension is unsupported.
        """
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        with open(config_file, "w", encoding="utf-8") as f:
            if config_file.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif config_file.suffix.lower() == ".json":
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config file format: {config_file.suffix}")

    def validate(self):
        """Validate configuration parameters"""
        # Validate model config
        # Accept any model string; allow backward-compatible common names
        if (
            not isinstance(self.model.clip_model_name, str)
            or len(self.model.clip_model_name.strip()) == 0
        ):
            raise ValueError(f"Invalid CLIP model name: {self.model.clip_model_name}")

        if self.encoder.embedding_dim < 1:
            raise ValueError("encoder.embedding_dim must be at least 1")

        if self.audio.sample_rate_hz < 1:
            raise ValueError("audio.sample_rate_hz must be at least 1")
        if self.audio.window_seconds <= 0:
            raise ValueError("audio.window_seconds must be positive")
        if self.audio.window_hop_seconds <= 0:
            raise ValueError("audio.window_hop_seconds must be positive")
        if self.audio.channel_count != 1:
            raise ValueError("audio.channel_count must be 1 for current audio placeholder")

        if self.model.device not in ["cuda", "cpu"]:
            raise ValueError(f"Unsupported device: {self.model.device}")

        if self.model.lora_rank < 1:
            raise ValueError("model.lora_rank must be at least 1")

        if self.model.lora_alpha <= 0:
            raise ValueError("model.lora_alpha must be > 0")

        if not 0 <= self.model.lora_dropout < 1:
            raise ValueError("model.lora_dropout must be in [0, 1)")

        # Validate classification config
        if not 0 <= self.classification.visual_weight <= 1:
            raise ValueError("visual_weight must be between 0 and 1")

        if not 0 <= self.classification.text_weight <= 1:
            raise ValueError("text_weight must be between 0 and 1")

        if abs(self.classification.visual_weight + self.classification.text_weight - 1.0) > 1e-6:
            raise ValueError("visual_weight + text_weight must equal 1.0")

        if self.evaluation.mode not in ["legacy", "episodic", "compare"]:
            raise ValueError("evaluation.mode must be one of: legacy, episodic, compare")

        if self.evaluation.episodes < 1:
            raise ValueError("evaluation.episodes must be at least 1")

        if not 0 < self.evaluation.confidence_level < 1:
            raise ValueError("evaluation.confidence_level must be between 0 and 1")

        if self.evaluation.split not in [None, "base", "novel"]:
            raise ValueError("evaluation.split must be one of: None, 'base', 'novel'")

        # Validate support set config
        if self.support_set.min_shots_per_class < 1:
            raise ValueError("min_shots_per_class must be at least 1")

        if self.support_set.max_shots_per_class < self.support_set.min_shots_per_class:
            raise ValueError("max_shots_per_class must be >= min_shots_per_class")


# Default configuration instance
default_config = Config()


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration instance"""
    if config_path is None:
        return default_config
    else:
        return Config.from_file(config_path)


def create_default_config_file(output_path: str = "config/default.yaml"):
    """Create a default configuration file"""
    config = Config()
    config.save(output_path)
    print(f"Default configuration saved to: {output_path}")


if __name__ == "__main__":
    # Create default configuration file for reference
    create_default_config_file()
