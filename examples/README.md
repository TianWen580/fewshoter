# Fewshoter Examples

This directory contains example scripts demonstrating how to use Fewshoter for few-shot image classification.

## Available Examples

### 1. Basic Classification (`basic_classification.py`)

Simple example showing single-image classification with a support set.

```bash
python basic_classification.py \
    --support_dir ./my_support_set \
    --query_image ./query.jpg \
    --model ViT-B/32
```

**Features:**
- Load support set from directory
- Classify single image
- Display top-k predictions with confidence scores

### 2. BioCLIP Classification (`bioclip_classification.py`)

Biology-optimized example using BioCLIP models for biodiversity classification.

```bash
python bioclip_classification.py \
    --support_dir ./insect_species \
    --query_image ./unknown_insect.jpg \
    --model BioCLIP-2 \
    --enable_nss
```

**Features:**
- BioCLIP model variants (BioCLIP, BioCLIP-2, BioCLIP-2.5)
- Optimized for biological species classification
- NSS (Neighborhood Subspace Suppression) for fine-grained discrimination
- Custom class descriptions support
- Attribute visualization

### 3. Batch Classification (`batch_classification.py`)

Process multiple images efficiently and save results.

```bash
python batch_classification.py \
    --prototypes ./outputs/prototypes.json \
    --query_dir ./test_images \
    --output results.json \
    --model BioCLIP-2
```

**Features:**
- Batch process entire directory
- Save results to JSON
- Compute statistics (accuracy, confidence distribution)
- Progress bar with tqdm

## Model Options

| Model | Architecture | Best For |
|-------|-------------|----------|
| `ViT-B/32` | ViT-Base, 32px patches | Fast inference, general purpose |
| `ViT-B/16` | ViT-Base, 16px patches | Better detail than B/32 |
| `ViT-L/14` | ViT-Large, 14px patches | High quality, slower |
| `BioCLIP` | ViT-B/16 | Biology/biodiversity (TreeOfLife-10M) |
| `BioCLIP-2` | ViT-L/14 | Biology/biodiversity (TreeOfLife-200M) |
| `BioCLIP-2.5` | ViT-H/14 | Best biology model (latest) |

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -e ..
   ```

2. **Create a support set:**
   ```bash
   mkdir -p my_support/{class_a,class_b,class_c}
   # Copy your example images
   ```

3. **Run an example:**
   ```bash
   python basic_classification.py \
       --support_dir my_support \
       --query_image test.jpg \
       --model BioCLIP-2
   ```

## Sample Data

For testing, you can use any image dataset. Recommended sources:

- **Biology:** [iNaturalist](https://www.inaturalist.org/)
- **General:** [ImageNet](http://www.image-net.org/) (sample subsets)
- **Camera Traps:** [Snapshot Serengeti](https://snapshotserengeti.org/)

## Support Set Structure

```
support_set/
├── species_a/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── img_003.jpg
├── species_b/
│   ├── img_001.jpg
│   └── img_002.jpg
└── species_c/
    └── img_001.jpg
```

See [Support Set Building Guide](../docs/SUPPORT_SET_GUIDE.md) for detailed instructions.

## Customization

### Using Custom Descriptions

Create a descriptions file for better text-based classification:

```python
# descriptions.py
classes = ["tiger", "lion", "leopard"]

descriptions = {
    "tiger": "Large orange cat with black stripes, white underbelly",
    "lion": "Large tan cat with mane (male), social predator",
    "leopard": "Spotted cat with rosette patterns, excellent climber",
}
```

Use with:
```bash
python bioclip_classification.py \
    --support_dir ./big_cats \
    --query_image ./unknown.jpg \
    --descriptions descriptions.py
```

### Configuration

Modify the Config object in the scripts:

```python
config = Config()
config.model.clip_model_name = "BioCLIP-2"
config.model.device = "cuda"
config.classification.confidence_threshold = 0.7
config.classification.enable_nss = True
```

## Output Format

### Batch Results JSON

```json
{
  "config": {
    "model": "BioCLIP-2",
    "confidence_threshold": 0.5,
    "num_classes": 10
  },
  "statistics": {
    "total": 100,
    "high_confidence": 85,
    "accuracy": 0.92
  },
  "results": [
    {
      "image": "test/img_001.jpg",
      "predicted": "tiger",
      "confidence": 0.95,
      "ground_truth": "tiger",
      "correct": true,
      "top3": [
        {"class": "tiger", "score": 0.95},
        {"class": "lion", "score": 0.03},
        {"class": "leopard", "score": 0.02}
      ]
    }
  ]
}
```

## Troubleshooting

### CUDA Out of Memory
```python
config.model.device = "cpu"  # Use CPU
# or
config.model.batch_size = 8  # Reduce batch size
```

### Slow Performance
```python
config.model.cache_features = True
config.model.cache_backend = "mmap"
```

### Low Accuracy
- Add more diverse support images
- Use BioCLIP for biology applications
- Enable NSS for fine-grained classes
- Add custom descriptions

## Next Steps

- Read the [main documentation](../README.md)
- Check [configuration options](../configs/)
- Explore [class descriptions](../descriptions/)
