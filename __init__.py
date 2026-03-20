"""
Fewshoter: CLIP Few-Shot Fine-Grained Classification Toolkit

This package implements a zero-training few-shot learning system for fine-grained
bird and animal classification using CLIP (Contrastive Language-Image Pretraining).

The system combines visual prototypes from support examples with text-based semantic
information to achieve robust classification without requiring model fine-tuning.

Key Features:
- Multi-scale feature extraction from CLIP models
- Attention-guided spatial feature alignment
- Discriminative text probe generation
- Hybrid visual-textual similarity computation
- Support for various CLIP model architectures

Modules:
- feature_extractor: Multi-scale feature extraction from images
- support_manager: Management of few-shot support sets and prototypes
- feature_aligner: Spatial alignment of features using attention maps
- attribute_generator: Generation of discriminative text descriptions
- classifier: Main classification engine combining all components
- utils: Utility functions and helper classes
- config: Configuration management
- visualization: Tools for visualizing attention maps and results

Usage:
    from fewshoter import FineGrainedClassifier, SupportSetManager

    # Initialize system
    support_manager = SupportSetManager('path/to/support_set')
    classifier = FineGrainedClassifier(support_manager)

    # Classify image
    result, confidence = classifier.classify('path/to/query_image.jpg')

Author: Based on research paper "利用CLIP实现少样本鸟兽分类"
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "CLIP Few-Shot Classification Team"

# Import main classes for easy access
from .engine.classifier import FineGrainedClassifier, create_classifier
from .data.support_manager import (
    SupportSetManager,
    create_support_manager,
    create_minimal_support_manager_from_models,
)
from .features.feature_extractor import (
    MultiScaleFeatureExtractor,
    create_feature_extractor,
)
from .features.attribute_generator import AttributeGenerator
from .features.feature_aligner import FeatureAligner
from .core.config import Config
from .core.utils import setup_logging, load_image, save_results
from .engine.svm_classifier import SVMClassifier, SVMConfig, create_svm_classifier
from .engine.nn_classifier import NNClassifier, NNConfig, create_nn_classifier

__all__ = [
    "FineGrainedClassifier",
    "SupportSetManager",
    "MultiScaleFeatureExtractor",
    "AttributeGenerator",
    "FeatureAligner",
    "Config",
    "setup_logging",
    "load_image",
    "save_results",
    "create_classifier",
    "create_support_manager",
    "create_minimal_support_manager_from_models",
    "create_feature_extractor",
    "SVMClassifier",
    "SVMConfig",
    "create_svm_classifier",
    "NNClassifier",
    "NNConfig",
    "create_nn_classifier",
]
