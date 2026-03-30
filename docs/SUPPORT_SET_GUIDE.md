# Support Set Building Guide

This guide explains how to build effective support sets for few-shot classification with Fewshoter, especially for biology and biodiversity applications.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Image Requirements](#image-requirements)
- [Building Support Sets for Biology](#building-support-sets-for-biology)
  - [Species Classification](#species-classification)
  - [Camera Trap Images](#camera-trap-images)
  - [Insect Classification](#insect-classification)
- [Best Practices](#best-practices)
- [Advanced: Custom Descriptions](#advanced-custom-descriptions)
- [Troubleshooting](#troubleshooting)

## Overview

A **support set** is a small collection of labeled example images used to teach the model about your classes. Unlike traditional deep learning that requires thousands of images per class, few-shot learning works with just 1-5 examples per class.

```
support_set/
├── class_a/           # Directory name = class label
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
├── class_b/
│   ├── 1.jpg
│   └── 2.jpg
└── class_c/
    └── 1.jpg
```

## Directory Structure

### Basic Structure

```
my_support_set/
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

### Naming Conventions

- **Class names**: Use descriptive, unique names without special characters
  - ✅ Good: `tiger`, `leopard_cat`, `bengal_tiger`
  - ❌ Bad: `tiger!`, `cat #1`, `unknown`

- **Image files**: Any common image format works (jpg, png, jpeg, webp)
  - Names don't matter, only the folder structure

### Example: Bird Species

```
bird_support_set/
├── american_goldfinch/
│   ├── goldfinch_01.jpg
│   ├── goldfinch_02.jpg
│   ├── goldfinch_03.jpg
│   └── goldfinch_04.jpg
├── blue_jay/
│   ├── bluejay_sight_001.png
│   └── bluejay_sight_002.png
├── northern_cardinal/
│   ├── cardinal_nest_01.jpg
│   ├── cardinal_nest_02.jpg
│   ├── cardinal_nest_03.jpg
│   └── cardinal_nest_04.jpg
└── tufted_titmouse/
    └── titmouse_01.jpg
```

## Image Requirements

### Minimum Requirements

| Parameter | Recommended | Minimum |
|-----------|-------------|---------|
| Images per class | 3-5 | 1 |
| Image resolution | 224x224+ | 64x64 |
| Total classes | 2-100 | 2 |
| Image quality | Clear, in focus | Readable |

### Image Guidelines

1. **Variety is key**: Include different angles, lighting, backgrounds
2. **Quality over quantity**: 3 good images > 10 poor images
3. **Consistent labeling**: All images in a folder should be the same class
4. **No duplicates**: Avoid exact copies (minor augmentations OK)

### Image Examples

**Good support images:**
- Clear view of the subject
- Multiple angles (front, side, back)
- Different lighting conditions
- Various backgrounds

**Poor support images:**
- Blurry or out of focus
- Subject too small in frame
- Obstructed view
- All images identical

## Building Support Sets for Biology

### Species Classification

For biological species classification, consider:

1. **Diagnostic features**: Ensure key identifying features are visible
   - Birds: Beak shape, plumage patterns, size
   - Mammals: Coat pattern, body shape, distinctive marks
   - Insects: Wing patterns, body segments, coloration

2. **Life stages**: Include different life stages if relevant
   ```
   butterfly_species/
   ├── adult_dorsal/      # Top view of adult
   ├── adult_ventral/     # Bottom view of adult
   └── larva/             # Caterpillar stage
   ```

3. **Sexual dimorphism**: If males/females look different
   ```
   peacock/
   ├── male_display/      # Male with full tail
   └── female/            # Female, more subdued colors
   ```

### Camera Trap Images

Camera trap images have unique challenges:

```
camera_trap_support/
├── white_tailed_deer/
│   ├── day_clear_01.jpg      # Daytime, good lighting
│   ├── day_clear_02.jpg
│   ├── night_ir_01.jpg       # Night infrared
│   ├── night_ir_02.jpg
│   └── motion_blur_01.jpg    # Motion blur example
├── black_bear/
│   ├── adult_day_01.jpg
│   ├── cub_day_01.jpg
│   └── night_ir_01.jpg
└── coyote/
    ├── day_01.jpg
    ├── day_02.jpg
    └── night_01.jpg
```

**Camera trap tips:**
- Include both day and night examples
- Include motion blur examples
- Show different distances (close, medium, far)
- Include various angles animals approach from

### Insect Classification

For insect classification, consider mounting/pose:

```
insect_support_set/
├── monarch_butterfly/
│   ├── dorsal_view.jpg       # Wings spread, top view
│   ├── ventral_view.jpg      # Wings spread, bottom view
│   └── lateral_view.jpg      # Side view
├── honey_bee/
│   ├── dorsal_01.jpg
│   ├── lateral_01.jpg
│   └── close_wings.jpg       # Wings folded
└── dragonfly_common/
    ├── spread_wings.jpg
    └── perched.jpg
```

## Best Practices

### 1. Start Small, Expand Gradually

```bash
# Start with 3-5 classes, 3-5 images each
mkdir -p support_set/{class_a,class_b,class_c}
# Add images...
# Test performance
# Add more classes/images based on results
```

### 2. Balance Your Classes

Try to have similar numbers of images per class:

```
# Good - balanced
class_a: 5 images
class_b: 4 images
class_c: 5 images

# Acceptable - slightly imbalanced
class_a: 5 images
class_b: 3 images
class_c: 4 images

# Poor - highly imbalanced
class_a: 20 images
class_b: 2 images
class_c: 1 image
```

### 3. Use Augmentation (Optional)

The system supports automatic augmentation. Enable in config:

```yaml
support_set:
  enable_support_augmentation: true
  support_aug_hflip: true
  support_aug_multiplier: 1
```

### 4. Validate Your Support Set

```python
from fewshoter import SupportSetManager, Config

config = Config()
manager = SupportSetManager(
    support_dir="path/to/support_set",
    config=config
)

# Validate and get statistics
validation = manager.validate_support_set()
print(f"Total classes: {validation['total_classes']}")
print(f"Warnings: {validation['warnings']}")
print(f"Errors: {validation['errors']}")
```

## Advanced: Custom Descriptions

For biology applications, provide class descriptions for better text-based classification:

### 1. Create a Descriptions File

```python
# descriptions/my_biology_project.py
classes = [
    "american_goldfinch",
    "blue_jay",
    "northern_cardinal",
]

descriptions = {
    "american_goldfinch": (
        "Small bright yellow finch with black cap and wings, "
        "white wing bars, and a conical orange bill. "
        "Male breeding plumage is vibrant yellow."
    ),
    "blue_jay": (
        "Large crested songbird with blue upperparts, "
        "white face and underparts, black necklace, "
        "and distinctive blue crest on head."
    ),
    "northern_cardinal": (
        "Medium-sized songbird with prominent crest. "
        "Male is bright red with black face mask. "
        "Female is tan-brown with red accents. "
        "Both have thick orange-red conical bill."
    ),
}
```

### 2. Reference in Config

```yaml
classification:
  desc_text_path: "descriptions/my_biology_project.py"
```

### 3. Use BioCLIP Models

For biology applications, use BioCLIP models which are pre-trained on biodiversity data:

```yaml
model:
  clip_model_name: "BioCLIP-2"  # or "BioCLIP", "BioCLIP-2.5"
```

## Troubleshooting

### Issue: Low Classification Accuracy

**Causes & Solutions:**

1. **Insufficient support images**
   - Solution: Add more variety (different angles, lighting)

2. **Poor image quality**
   - Solution: Replace blurry/dark images

3. **Similar classes hard to distinguish**
   - Solution: Add more images showing diagnostic features
   - Solution: Use custom descriptions highlighting differences
   - Solution: Consider if classes should be merged

4. **Wrong model for domain**
   - Solution: Use BioCLIP for biology/biodiversity
   - Solution: Try different CLIP variants (ViT-B vs ViT-L)

### Issue: "Class has only X images" Warning

This means a class has fewer than `min_shots_per_class`:

```yaml
support_set:
  min_shots_per_class: 1  # Lower the minimum (default: 1)
```

### Issue: Out of Memory

```yaml
model:
  batch_size: 16      # Reduce batch size
  cache_backend: "disk"  # Use disk instead of memory cache
```

### Issue: Slow Processing

```yaml
model:
  device: "cuda"      # Use GPU if available
  cache_features: true  # Enable feature caching
  cache_backend: "mmap"  # Use memory-mapped cache
```

## Quick Start Example

```bash
# 1. Create support set structure
mkdir -p my_birds/{robin,cardinal,blue_jay}

# 2. Add images (copy your images)
cp robin*.jpg my_birds/robin/
cp cardinal*.jpg my_birds/cardinal/
cp bluejay*.jpg my_birds/blue_jay/

# 3. Build prototypes
fewshoter-train \
  --support_dir my_birds \
  --output_dir outputs \
  --model BioCLIP-2 \
  --config configs/bird_classification.yaml

# 4. Run inference
fewshoter-inference \
  --prototypes outputs/prototypes.json \
  --image test_image.jpg
```

## Resources

- [BioCLIP Paper](https://arxiv.org/abs/2311.18803)
- [OpenCLIP Models](https://github.com/mlfoundations/open_clip)
- [Fewshoter Documentation](../README.md)

---

**Need help?** Open an issue on GitHub with your support set structure and we'll help optimize it.
