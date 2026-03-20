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
├── cli/        # user-facing entrypoints: train/inference/evaluate/api_server
├── core/       # config + shared utilities
├── features/   # extraction/alignment/attribute generation/dim reduction
├── data/       # support-set and dataset management
├── engine/     # classification engines and optimization
├── configs/    # task-specific YAML configs
└── descriptions/ # class-level description dictionaries
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

```bash
fewshoter-evaluate \
  --prototypes outputs/prototypes.json \
  --test_dir test_set
```

## API and Python Usage

```python
from fewshoter import Config, create_feature_extractor, SupportSetManager, create_classifier

config = Config()
feature_extractor = create_feature_extractor(config)
support_manager = SupportSetManager.from_prototypes("outputs/prototypes.json", feature_extractor)
classifier = create_classifier(support_manager, config)

label, details = classifier.classify("query.jpg", return_scores=True)
print(label, details["confidence"])
```

## Core Capabilities

- Multi-scale CLIP feature extraction
- Attention-guided feature alignment
- Attribute-driven text probe generation
- Optional SVM and MLP classifier heads
- Dimensionality reduction and cache acceleration
- Open-set unknown rejection strategies

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

- Better model zoo support
- ONNX/TensorRT export path
- Lightweight web demo
- Benchmark suite with standard datasets

## License

MIT License. See [LICENSE](LICENSE).

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening PRs.
