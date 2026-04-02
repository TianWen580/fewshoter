# Fewshoter

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/TianWen580/fewshoter/actions/workflows/ci.yml/badge.svg)](https://github.com/TianWen580/fewshoter/actions/workflows/ci.yml)

**Few-shot image classification without the training headache.**

Classify images with just 2-5 examples per class. No model fine-tuning. No GPU required for inference. Built on CLIP with hybrid visual-text reasoning for fine-grained recognition.

---

## The Problem

You have a classification task with:
- **Few examples** — Maybe 2-5 images per class
- **Fine-grained categories** — Bird species, plant varieties, product defects
- **No time/resources for training** — Full model fine-tuning takes hours/days

Traditional deep learning requires hundreds of samples and lengthy training. Fewshoter gives you **instant classification** from a handful of examples.

## Why Fewshoter

| Feature | Fewshoter | Traditional Deep Learning |
|---------|-----------|---------------------------|
| Samples needed | **2-5 per class** | 100+ per class |
| Training time | **Seconds (prototype build)** | Hours to days |
| GPU required | **No (inference)** | Yes |
| Fine-grained support | **Yes (multi-scale + text)** | Needs custom training |
| Open-set handling | **Built-in** | Requires extra work |
| Adaptation | **Add new classes instantly** | Retrain from scratch |

**Key capabilities:**

- **Zero-training mode** — Build prototypes, classify immediately
- **Hybrid reasoning** — Visual similarity + text probes for better accuracy
- **Multiple heads** — Prototype, SVM, MLP — pick what works best
- **Calibrated confidence** — Know when predictions are uncertain
- **Production-ready** — CLI + API server + episodic benchmarking

## Quick Demo

### Option 1: Web Interface (Recommended for First Try)

```bash
# Install with web demo support
pip install fewshoter[web]

# Launch interactive demo
fewshoter-web-demo

# Or with public share link
fewshoter-web-demo --share
```

Open http://localhost:7860 and:
1. Select a CLIP model and click **Initialize**
2. Upload 2-5 images per class, name each class
3. Click **Build Prototypes**
4. Upload a query image and get predictions!

### Option 2: Command Line

```bash
# 1. Install
pip install fewshoter

# 2. Organize your support set (2-5 images per class)
support_set/
├── robin/
│   ├── img1.jpg
│   └── img2.jpg
└── sparrow/
    ├── img1.jpg
    └── img2.jpg

# 3. Build prototypes (takes ~2 seconds)
fewshoter-train --support_dir support_set --output_dir outputs

# 4. Classify a new image
fewshoter-inference --prototypes outputs/prototypes.json --image query.jpg

# Output:
# Class: robin
# Confidence: 0.87
# Scores: {robin: 0.87, sparrow: 0.12, unknown: 0.01}
```

**That's it.** No training loops. No hyperparameter tuning. No GPU needed for inference.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        SUPPORT IMAGES                           │
│                    (2-5 samples per class)                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              MULTI-SCALE FEATURE EXTRACTION                     │
│         CLIP visual encoder + intermediate layer hooks          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│   VISUAL PROTOTYPES   │   │     TEXT PROBES       │
│  (class embeddings)   │   │  (class descriptions) │
└─────────────┬─────────┘   └─────────────┬─────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID CLASSIFIER                            │
│         Visual similarity + Text matching + SVM/MLP             │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                PREDICTION + CONFIDENCE                          │
│              (with open-set rejection)                          │
└─────────────────────────────────────────────────────────────────┘
```

**Why hybrid?** Visual prototypes capture appearance. Text probes capture semantics. Together, they handle fine-grained distinctions better than either alone.

## Target Users

- **Biologists/Ecologists** — Species identification from few field photos
- **Quality Control** — Defect classification with limited samples
- **E-commerce** — Product categorization with new SKUs
- **Medical Imaging** — Rare condition classification
- **Researchers** — Few-shot learning benchmarking

## Features in Depth

### Multi-Modal Encoders

- **CLIP (image)** — ViT-B/32, ViT-L/14, BioCLIP for biology tasks
- **Perch 2.0 (audio)** — Placeholder for bird sound classification

### Learners

- **Prototypical Networks** — Distance-based classification
- **Extensible base** — Add MAML, Matching Networks, Relation Networks

### Evaluation Modes

```bash
# Episodic benchmarking (N-way K-shot)
fewshoter-evaluate --mode episodic --episodes 600 --n_way 5 --k_shot 5

# Results with confidence intervals:
# Accuracy: 0.823 ± 0.015 (95% CI)
```

### LoRA Fine-Tuning (Optional)

When you have more data and want better performance:

```python
from fewshoter.peft.lora import apply_image_encoder_lora

# Apply LoRA to visual branch only
apply_image_encoder_lora(encoder, rank=8, alpha=16.0)
```

### Open-Set Rejection

Not everything fits your classes. Fewshoter detects unknown inputs:

```python
# Automatic unknown detection via:
# - Z-score thresholding
# - Margin-based rejection
# - Softmax confidence
```

## Architecture

```
fewshoter/
├── cli/              # train, inference, evaluate, api-server
├── core/             # Config, Episode, utilities
├── modalities/       # CLIP (image), Perch (audio placeholder)
├── learners/         # Prototypical + extensible base
├── evaluation/       # Episodic + legacy benchmarking
├── data/             # Support set management, datasets
├── features/         # Multi-scale extraction, alignment
├── peft/             # LoRA fine-tuning
├── engine/           # Classifier implementations
└── configs/          # YAML task configs
```

**Design principles:**

- Contract-based modularity (Protocol classes)
- Backward compatibility (legacy path preserved)
- Type-safe with runtime validation

## Installation

```bash
# From PyPI (coming soon)
pip install fewshoter

# From source
git clone https://github.com/TianWen580/fewshoter.git
cd fewshoter
pip install -e .
```

## Documentation

- [Support Set Guide](docs/SUPPORT_SET_GUIDE.md) — How to organize your examples
- [Examples](examples/README.md) — BioCLIP, custom descriptions, LoRA
- [Contributing](CONTRIBUTING.md) — Development setup

## Benchmarks

Coming soon: Standard few-shot benchmarks (miniImageNet, tieredImageNet, CUB-200).

## Roadmap

**Done ✓**
- Modular architecture with contracts
- Episodic evaluation + prototypical learner
- LoRA for image encoder
- BioCLIP integration
- Web demo (Gradio)

**Next**
- ONNX/TensorRT export
- Standard benchmark suite

**Future**
- MAML, Matching Networks, Relation Networks
- Multi-modal fusion (image + audio)
- Model zoo

## License

MIT — Free for personal and commercial use.

## Contributing

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

**Star this repo if it helps your project!**