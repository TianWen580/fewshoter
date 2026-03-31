#!/usr/bin/env python3
"""
Training Script for CLIP Few-Shot Classification System

This script processes the support set to generate prototypes and prepares
the system for inference. Since this is a zero-training approach, "training"
refers to support set processing and prototype generation.

Usage:
    python train.py --support_dir data/support_set --config config/default.yaml
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

torch = importlib.import_module("torch")

from ..core.config import Config, get_config
from ..features.feature_extractor import create_feature_extractor
from ..data.support_manager import create_support_manager
from ..core.utils import setup_logging, set_random_seed, ensure_dir


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process support set for CLIP few-shot classification"
    )

    # Required arguments
    parser.add_argument(
        "--support_dir",
        type=str,
        required=True,
        help="Directory containing support set organized by class",
    )

    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (YAML or JSON)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for prototypes and logs",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        help="CLIP/OpenCLIP model name (e.g., ViT-B/32, ViT-L/14@336px, ViT-H-14-quickgelu)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation",
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for feature extraction"
    )

    parser.add_argument(
        "--min_shots", type=int, default=1, help="Minimum number of shots per class"
    )

    parser.add_argument(
        "--max_shots", type=int, default=5, help="Maximum number of shots per class"
    )

    parser.add_argument(
        "--evaluation_mode",
        type=str,
        default="legacy",
        choices=["legacy", "episodic", "compare"],
        help="Evaluation pipeline mode for downstream compatibility wiring",
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodic tasks to evaluate when episodic mode is used",
    )

    parser.add_argument(
        "--episode_n_way",
        type=int,
        default=5,
        help="Number of classes per episodic task (N-way)",
    )

    parser.add_argument(
        "--episode_k_shot",
        type=int,
        default=1,
        help="Number of support examples per class in episodic tasks (K-shot)",
    )

    parser.add_argument(
        "--episode_num_queries",
        type=int,
        default=5,
        help="Number of query examples per class in episodic tasks",
    )

    parser.add_argument(
        "--episode_split",
        type=str,
        default="none",
        choices=["none", "base", "novel"],
        help="Optional split selector for episodic tasks",
    )

    parser.add_argument(
        "--confidence_level",
        type=float,
        default=0.95,
        help="Confidence level for episodic confidence intervals",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate support set quality after processing",
    )

    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """Setup and return the appropriate device"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            print("CUDA not available, using CPU")
    else:
        device = device_arg
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device = "cpu"

    return device


def validate_support_directory(support_dir: Path) -> bool:
    """Validate support directory structure"""
    if not support_dir.exists():
        print(f"Error: Support directory does not exist: {support_dir}")
        return False

    if not support_dir.is_dir():
        print(f"Error: Support path is not a directory: {support_dir}")
        return False

    # Check for class subdirectories
    class_dirs = [d for d in support_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"Error: No class subdirectories found in: {support_dir}")
        return False

    print(f"Found {len(class_dirs)} class directories in support set")

    # Check each class directory for images
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    for class_dir in class_dirs:
        image_files = [f for f in class_dir.iterdir() if f.suffix.lower() in valid_extensions]
        if not image_files:
            print(f"Warning: No images found in class directory: {class_dir.name}")
        else:
            print(f"  {class_dir.name}: {len(image_files)} images")

    return True


def main() -> None:
    """Main training function"""
    args = parse_arguments()

    # Setup paths
    support_dir = Path(args.support_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Setup logging
    log_file = output_dir / "training.log"
    logger = setup_logging(args.log_level, str(log_file))

    logger.info("Starting CLIP few-shot classification training")
    logger.info(f"Arguments: {vars(args)}")

    # Validate support directory
    if not validate_support_directory(support_dir):
        sys.exit(1)

    # Setup device
    device = setup_device(args.device)

    # Set random seed
    set_random_seed(args.seed)

    # Load or create configuration
    if args.config:
        config = get_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        config = Config()
        logger.info("Using default configuration")

    # Override config with command line arguments
    config.model.clip_model_name = args.model
    config.model.device = device
    config.model.batch_size = args.batch_size
    config.support_set.min_shots_per_class = args.min_shots
    config.support_set.max_shots_per_class = args.max_shots
    config.evaluation.mode = args.evaluation_mode
    config.evaluation.episodes = args.num_episodes
    config.evaluation.confidence_level = args.confidence_level
    config.evaluation.split = None if args.episode_split == "none" else args.episode_split
    config.episode.n_way = args.episode_n_way
    config.episode.k_shot = args.episode_k_shot
    config.episode.num_queries = args.episode_num_queries
    config.log_level = args.log_level
    config.random_seed = args.seed

    # Validate configuration
    try:
        config.validate()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)

    # Save configuration
    config_output_path = output_dir / "config.yaml"
    config.save(config_output_path)
    logger.info(f"Configuration saved to: {config_output_path}")

    try:
        # Initialize feature extractor
        logger.info(f"Initializing feature extractor with model: {config.model.clip_model_name}")
        feature_extractor = create_feature_extractor(config)

        # Initialize support set manager
        logger.info("Loading and processing support set...")
        support_manager = create_support_manager(
            support_dir=support_dir, feature_extractor=feature_extractor, config=config
        )

        # Validate support set if requested
        if args.validate:
            logger.info("Validating support set quality...")
            validation_results = support_manager.validate_support_set()

            logger.info(f"Validation results:")
            logger.info(f"  Total classes: {validation_results['total_classes']}")
            logger.info(f"  Warnings: {len(validation_results['warnings'])}")
            logger.info(f"  Errors: {len(validation_results['errors'])}")

            if validation_results["warnings"]:
                logger.warning("Validation warnings:")
                for warning in validation_results["warnings"]:
                    logger.warning(f"  - {warning}")

            if validation_results["errors"]:
                logger.error("Validation errors:")
                for error in validation_results["errors"]:
                    logger.error(f"  - {error}")

                if validation_results["errors"]:
                    logger.error("Support set validation failed")
                    sys.exit(1)

            # Save validation results
            validation_output = output_dir / "validation_results.json"
            import json

            with open(validation_output, "w") as f:
                json.dump(validation_results, f, indent=2)
            logger.info(f"Validation results saved to: {validation_output}")

        # Save prototypes
        prototypes_path = output_dir / "prototypes.json"
        support_manager.save_prototypes(prototypes_path)

        # Optionally train and save SVM classifier from support embeddings
        try:
            if getattr(config.classification, "use_svm", False):
                from ..engine.svm_classifier import (
                    SVMConfig as _SVMConfig,
                    create_svm_classifier as _create_svm,
                )
                from ..features.attribute_generator import (
                    AttributeGenerator as _AttrGen,
                )

                # Precompute class text features
                _desc = None
                try:
                    dp = getattr(config.classification, "desc_text_path", None)
                    if dp:
                        p = Path(str(dp))
                        if p.exists():
                            import json as _json

                            with open(p, "r", encoding="utf-8") as _f:
                                _desc = _json.load(_f)
                except Exception:
                    _desc = None
                _attr = _AttrGen(
                    clip_model=feature_extractor.model,
                    device=str(config.model.device),
                    max_probes_per_class=config.classification.max_text_probes,
                )
                _text_feats = {}
                for _cn in support_manager.class_names:
                    try:
                        if (
                            _desc
                            and _cn in _desc
                            and isinstance(_desc[_cn], str)
                            and _desc[_cn].strip()
                        ):
                            _probes = [_desc[_cn].strip()]
                        else:
                            _probes = _attr.generate_text_probes(
                                _cn, support_manager.get_attributes(_cn)
                            )
                        _tf = _attr.encode_text_probes(_probes).mean(dim=0)
                        _tf = _tf / (_tf.norm() + 1e-8)
                        _text_feats[_cn] = _tf
                    except Exception:
                        continue

                svm_cfg = _SVMConfig(
                    kernel=getattr(config.classification, "svm_kernel", "rbf"),
                    C=float(getattr(config.classification, "svm_C", 1.0)),
                    gamma=getattr(config.classification, "svm_gamma", "scale"),
                    probability=bool(getattr(config.classification, "svm_probability", True)),
                    feature_layer=getattr(config.classification, "svm_feature_layer", "global"),
                    use_class_weight=bool(
                        getattr(config.classification, "svm_use_class_weight", True)
                    ),
                )
                svm = _create_svm(support_manager.class_names, svm_cfg)
                svm.fit_from_support_manager(
                    support_manager,
                    svm_cfg.feature_layer,
                    text_features=_text_feats,
                    use_text_sim_features=bool(
                        getattr(config.classification, "use_text_sim_features", False)
                    ),
                    beta=float(getattr(config.classification, "text_feat_beta", 0.5)),
                )
                svm_path = getattr(config.classification, "svm_model_path", None)
                if not svm_path:
                    svm_path = str(output_dir / "svm.pkl")
                svm.save(svm_path)
                logger.info(f"SVM model trained and saved to: {svm_path}")
        except Exception as e:
            logger.warning(f"SVM training skipped due to error: {e}")

        # Optionally train and save NN classifier from support embeddings
        try:
            if getattr(config.classification, "use_nn", False):
                from ..engine.nn_classifier import (
                    NNConfig as _NNConfig,
                    create_nn_classifier as _create_nn,
                )
                from ..features.attribute_generator import (
                    AttributeGenerator as _AttrGen,
                )

                # Precompute class text features (reuse if already computed above)
                _desc = None
                try:
                    dp = getattr(config.classification, "desc_text_path", None)
                    if dp:
                        p = Path(str(dp))
                        if p.exists():
                            import json as _json

                            with open(p, "r", encoding="utf-8") as _f:
                                _desc = _json.load(_f)
                except Exception:
                    _desc = None
                _attr = _AttrGen(
                    clip_model=feature_extractor.model,
                    device=str(config.model.device),
                    max_probes_per_class=config.classification.max_text_probes,
                )
                _text_feats = {}
                for _cn in support_manager.class_names:
                    try:
                        if (
                            _desc
                            and _cn in _desc
                            and isinstance(_desc[_cn], str)
                            and _desc[_cn].strip()
                        ):
                            _probes = [_desc[_cn].strip()]
                        else:
                            _probes = _attr.generate_text_probes(
                                _cn, support_manager.get_attributes(_cn)
                            )
                        _tf = _attr.encode_text_probes(_probes).mean(dim=0)
                        _tf = _tf / (_tf.norm() + 1e-8)
                        _text_feats[_cn] = _tf
                    except Exception:
                        continue

                nn_cfg = _NNConfig(
                    hidden_sizes=list(getattr(config.classification, "nn_hidden_sizes", [512])),
                    activation=getattr(config.classification, "nn_activation", "relu"),
                    dropout=float(getattr(config.classification, "nn_dropout", 0.1)),
                    epochs=int(getattr(config.classification, "nn_epochs", 20)),
                    lr=float(getattr(config.classification, "nn_lr", 1e-3)),
                    weight_decay=float(getattr(config.classification, "nn_weight_decay", 1e-4)),
                    batch_size=int(getattr(config.classification, "nn_batch_size", 64)),
                    early_stopping_patience=int(
                        getattr(config.classification, "nn_early_stopping_patience", 5)
                    ),
                    class_weight_balanced=bool(
                        getattr(config.classification, "nn_class_weight_balanced", True)
                    ),
                    feature_layer=getattr(config.classification, "nn_feature_layer", "global"),
                    model_path=getattr(config.classification, "nn_model_path", None),
                    device=getattr(config.model, "device", "cpu"),
                )
                nn = _create_nn(support_manager.class_names, nn_cfg)
                nn.fit_from_support_manager(
                    support_manager,
                    nn_cfg.feature_layer,
                    text_features=_text_feats,
                    use_text_sim_features=bool(
                        getattr(config.classification, "use_text_sim_features", False)
                    ),
                    beta=float(getattr(config.classification, "text_feat_beta", 0.5)),
                )
                nn_path = getattr(config.classification, "nn_model_path", None)
                if not nn_path:
                    nn_path = str(output_dir / "nn_head.pt")
                nn.save(nn_path)
                logger.info(f"NN model trained and saved to: {nn_path}")
        except Exception as e:
            logger.warning(f"NN training skipped due to error: {e}")

        # Print summary
        logger.info("Training completed successfully!")
        logger.info(f"Summary:")
        logger.info(f"  Classes processed: {len(support_manager.class_names)}")
        logger.info(
            f"  Total support images: {sum(len(imgs) for imgs in support_manager.support_images.values())}"
        )
        logger.info(f"  Prototypes saved to: {prototypes_path}")
        logger.info(f"  Configuration saved to: {config_output_path}")

        # Print class statistics
        logger.info("Class statistics:")
        for class_name, stats in support_manager.class_statistics.items():
            logger.info(
                f"  {class_name}: {stats['num_images']} images, "
                f"{stats['num_attributes']} attributes, "
                f"{len(stats['prototype_scales'])} prototype scales"
            )

        print("\nTraining completed successfully!")
        print(f"Prototypes saved to: {prototypes_path}")
        print(f"Ready for inference. Use: python inference.py --prototypes {prototypes_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
