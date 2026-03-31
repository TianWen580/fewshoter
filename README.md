# Fewshoter

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/TianWen580/fewshoter/actions/workflows/ci.yml/badge.svg)](https://github.com/TianWen580/fewshoter/actions/workflows/ci.yml)

Fewshoter is a CLIP-based few-shot image classification toolkit focused on fine-grained recognition. It avoids full model fine-tuning and instead learns from a small support set with feature extraction, prototype construction, optional alignment, and hybrid visual-text reasoning.

## Why Fewshoter

- Fast adaptation with small support sets
- Zero-training and lightweight adaptation modes
- Pluggable heads: prototype, SVM, MLP
- Open-set handling and calibration tools
- Practical CLI and API workflow for production usage

## Visual Overview

```text
            +----------------------+
            |    Support Images    |
            +----------+-----------+
                       |
                       v
      +----------------+----------------+
      | Multi-scale Feature Extraction   |
      +----------------+----------------+
                       |
         +-------------+-------------+
         |                           |
         v                           v
+----------------------+   +----------------------+
|  Visual Prototypes   |   |   Text Probes / NLP  |
+----------+-----------+   +----------+-----------+
           |                          |
           +------------+-------------+
                        v
             +----------+----------+
             | Hybrid Classifier   |
             +----------+----------+
                        |
                        v
             +----------+----------+
             | Prediction + Scores |
             +---------------------+
```

## Product Architecture

```text
fewshoter/
├── cli/              # user-facing entrypoints: train/inference/evaluate/api_server
├── core/             # config, episode contracts, and shared utilities
├── modalities/       # encoder adapters (CLIP image, Perch audio placeholder)
├── learners/         # few-shot learners (prototypical, base contracts)
├── evaluation/       # evaluators (episodic, legacy, compare mode)
├── data/             # datasets, samplers, and support-set management
├── features/         # extraction/alignment/attribute generation/dim reduction
├── peft/             # parameter-efficient fine-tuning (LoRA for image encoders)
├── perch_integration/# audio pipeline contracts and Perch 2.0 placeholder
├── engine/           # classification engines and optimization
├── configs/          # task-specific YAML configs
└── descriptions/     # class-level description dictionaries
```

## Quick Start

### 1) Install

```bash
pip install -e .
```

### 2) Prepare support set

```text
support_set/
├── class_a/
│   ├── 1.jpg
│   └── 2.jpg
└── class_b/
    ├── 1.jpg
    └── 2.jpg
```

### 3) Build prototypes

```bash
fewshoter-train \
  --support_dir support_set \
  --output_dir outputs \
  --model ViT-B/32
```

### 4) Run inference

```bash
fewshoter-inference \
  --prototypes outputs/prototypes.json \
  --image query.jpg
```

### 5) Evaluate

Legacy mode (existing classifier):
```bash
fewshoter-evaluate \
  --prototypes outputs/prototypes.json \
  --test_dir test_set
```

Episodic few-shot benchmarking:
```bash
fewshoter-evaluate \
  --mode episodic \
  --episodes 600 \
  --test_dir test_set \
  --output_json results/episodic_5way5shot.json
```

Compare legacy vs episodic:
```bash
fewshoter-evaluate \
  --mode compare \
  --prototypes outputs/prototypes.json \
  --test_dir test_set \
  --output_json results/comparison.json
```

## Architecture Highlights

### Modular Design

The codebase follows a modular, contract-based architecture:

- **Modalities**: Clean encoder interface (`ModalityEncoder`) with implementations for CLIP (image) and Perch 2.0 placeholder (audio)
- **Learners**: Abstract base (`BaseLearner`) with prototypical networks implementation; easy to extend for MAML, matching networks, etc.
- **Evaluation**: Multiple evaluation modes (legacy, episodic, compare) with support for N-way K-shot benchmarking and confidence intervals
- **Data**: Episode-based dataset sampling with deterministic splits for reproducible few-shot experiments
- **PEFT**: LoRA support scoped to image encoder branches, leaving text encoders frozen

### Contracts and Type Safety

Key runtime-checkable contracts ensure component interoperability:

```python
from fewshoter.modalities.base import ModalityEncoder
from fewshoter.learners.base import BaseLearner
from fewshoter.evaluation.base import BaseEvaluator

# All contracts are Protocol classes for structural typing
# Implementations are verified at runtime via validate_instance()
```

### Backward Compatibility

The modular architecture coexists with the legacy pipeline:

- Existing scripts continue to work without changes
- Legacy CLIP path is preserved behind the image modality adapter
- Config files support both old and new configuration sections
- CLI maintains all existing flags while adding new episodic options

## API and Python Usage

### Basic Classification (Legacy Path)

```python
from fewshoter import Config, create_feature_extractor, SupportSetManager, create_classifier

config = Config()
feature_extractor = create_feature_extractor(config)
support_manager = SupportSetManager.from_prototypes("outputs/prototypes.json", feature_extractor)
classifier = create_classifier(support_manager, config)

label, details = classifier.classify("query.jpg", return_scores=True)
print(label, details["confidence"])
```

### Episodic Evaluation (Few-Shot Benchmarking)

```python
from fewshoter.core.config import Config
from fewshoter.evaluation.episodic import EpisodicEvaluator
from fewshoter.learners.prototypical import PrototypicalLearner

config = Config()
config.evaluation.mode = "episodic"
config.evaluation.episodes = 600

learner = PrototypicalLearner(distance="euclidean")
evaluator = EpisodicEvaluator(
    learner=learner,
    encoder=feature_extractor,
    n_way=5,
    k_shot=5,
    n_query=15
)

results = evaluator.evaluate(dataset, split="test")
print(f"Accuracy: {results['mean_accuracy']:.3f} ± {results['ci_half_width']:.3f}")
```

### LoRA Fine-Tuning for Image Encoder

```python
from fewshoter.peft.lora import apply_image_encoder_lora

# Apply LoRA to image encoder's visual branch
result = apply_image_encoder_lora(
    encoder=feature_extractor,
    enabled=True,
    rank=8,
    alpha=16.0,
    target_modules=["attn", "mlp"],
    freeze_non_lora=True
)
print(f"Applied LoRA to {result['applied']} layers in {result['scope']} scope")
```

## Core Capabilities

- **Multi-modal encoders**: CLIP for images, Perch 2.0-ready placeholder for audio
- **Modular learners**: Prototypical networks with Euclidean distance, extensible base classes
- **Episodic evaluation**: N-way K-shot benchmarking with confidence intervals
- **LoRA support**: Parameter-efficient fine-tuning for image encoders (image branch only)
- **Multi-scale CLIP feature extraction**: Intermediate layer hooks for spatial features
- **Attention-guided feature alignment**: Discriminative token mining and subcenters
- **Attribute-driven text probe generation**: Enhanced class descriptions
- **Optional SVM and MLP classifier heads**: Alternative to prototype-based classification
- **Dimensionality reduction and cache acceleration**: PCA and mmap/disk caching
- **Open-set unknown rejection**: Z-score, margin, and softmax-based strategies

## Security and Privacy

- No credentials are required by default
- Keep data paths/configs local to your own environment
- Do not commit private datasets, model artifacts, or generated outputs
- Review `SECURITY.md` before deploying public services

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Roadmap

### Completed ✓

- Modular encoder/learner/evaluator architecture with contracts
- Episodic evaluation with N-way K-shot benchmarking
- Prototypical learner with Euclidean distance
- LoRA support for image encoder fine-tuning
- Perch 2.0 integration placeholder (offline mode)
- Backward-compatible CLI with new episodic flags

### In Progress

- ONNX/TensorRT export path
- Benchmark suite with standard datasets (miniImageNet, tieredImageNet, CUB)

### Future

- Live Perch 2.0 runtime integration (when available)
- Additional learners: MAML, Matching Networks, Relation Networks
- Multi-modal fusion (image + audio joint embeddings)
- Lightweight web demo
- Model zoo with pretrained few-shot checkpoints

## License

MIT License. See [LICENSE](LICENSE).

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening PRs.
