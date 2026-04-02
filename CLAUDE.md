# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fewshoter is a CLIP-based few-shot image classification toolkit for fine-grained recognition. It uses visual prototypes and text probes for zero-training classification, with optional SVM/MLP heads and LoRA fine-tuning support.

**Conda environment**: `torch`

## Development Commands

```bash
# Install (editable mode)
conda activate torch
pip install -e .

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Run specific test file
python -m pytest tests/core/test_config_compatibility.py

# Lint
ruff check .

# Format
ruff format .
```

## CLI Entry Points

After installation, these commands are available:

- `fewshoter-train` — Build prototypes from support set
- `fewshoter-inference` — Classify images using prototypes
- `fewshoter-evaluate` — Evaluate classifier (legacy/episodic modes)
- `fewshoter-api-server` — Flask API server

Example workflow:
```bash
fewshoter-train --support_dir support_set --output_dir outputs --model ViT-B/32
fewshoter-inference --prototypes outputs/prototypes.json --image query.jpg
fewshoter-evaluate --mode episodic --episodes 600 --test_dir test_set
```

## Architecture

### Module Structure

```
fewshoter/
├── cli/              # Entry points (train, inference, evaluate, api_server)
├── core/             # Config, Episode, utilities
├── modalities/       # Encoder contracts (CLIP image, Perch audio placeholder)
├── learners/         # Few-shot learners (PrototypicalLearner, BaseLearner contract)
├── evaluation/       # EpisodicEvaluator, legacy evaluator, Metrics
├── data/             # SupportSetManager, datasets, samplers
├── features/         # Feature extraction, alignment, attribute generation
├── peft/             # LoRA for image encoder fine-tuning
├── engine/           # FineGrainedClassifier, SVM/NN classifiers
└── configs/          # YAML configuration files
```

### Key Contracts

The codebase uses Protocol-style contracts for component interoperability:

- **`EncoderContract`** (modalities/base.py): Requires `encode()` method, `embedding_dim`, `modality`
- **`LearnerContract`** (learners/base.py): Requires `fit()`, `predict()`, `from_encoder()`
- **`Episode`** (core/episode.py): Container for support/query embeddings and labels

### Classification Pipeline

1. **Feature Extraction**: `MultiScaleFeatureExtractor` extracts global + intermediate layer features from CLIP
2. **Support Set Management**: `SupportSetManager` builds prototypes per class from support images
3. **Classification**: `FineGrainedClassifier` combines visual similarity + text similarity + optional adapter/SVM/NN scores

### Evaluation Modes

- **legacy**: Use existing classifier on test set
- **episodic**: N-way K-shot benchmarking with confidence intervals
- **compare**: Side-by-side comparison of legacy vs prototypical learner

## Configuration

Config is centralized in `core/config.py` using dataclasses. Load from YAML:

```python
from fewshoter import Config
config = Config.from_file("configs/default.yaml")
```

Key configuration sections:
- `model`: CLIP model name, device, batch size, LoRA settings
- `encoder`: Modality, embedding dimension, normalization
- `classification`: Visual/text weights, thresholds, SVM/NN settings, calibration
- `evaluation`: Mode (legacy/episodic/compare), episodes, confidence level
- `episode`: N-way, K-shot, num_queries for episodic evaluation
- `support_set`: Min/max shots, prototype EMA, augmentation

## Important Patterns

### Support Set Directory Structure

```
support_set/
├── class_a/
│   ├── 1.jpg
│   └── 2.jpg
└── class_b/
    └── 1.jpg
```

### Custom Class Descriptions

Create a Python file with `desc_dict` mapping class names to descriptions:
```python
desc_dict = {
    "species_a": "Small yellow bird with black cap...",
}
```

Reference in config: `classification.desc_text_path: "descriptions/my_project.py"`

### BioCLIP Models

For biology/biodiversity applications, use BioCLIP models:
```yaml
model:
  clip_model_name: "BioCLIP-2"
```

## Testing

Tests are in `tests/` organized by module:
- `tests/core/` — Config, episode, architecture contracts
- `tests/learners/` — Prototypical learner, distance utilities
- `tests/evaluation/` — Episodic evaluator
- `tests/cli/` — CLI smoke tests
- `tests/modalities/` — CLIP encoder, audio contracts

Run full suite: `python -m pytest`