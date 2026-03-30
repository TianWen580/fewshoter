#!/usr/bin/env python3
"""
BioCLIP Biology Classification Example

This example demonstrates using BioCLIP models for biodiversity and biological
species classification. BioCLIP is pre-trained on the TreeOfLife dataset
(450K+ species) and performs better than standard CLIP for biology applications.

Usage:
    python bioclip_classification.py \
        --support_dir ./insect_support_set \
        --query_image ./unknown_insect.jpg \
        --model BioCLIP-2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fewshoter import (
    Config,
    create_feature_extractor,
    SupportSetManager,
    create_classifier,
)


def main():
    parser = argparse.ArgumentParser(description="BioCLIP classification for biology/biodiversity")
    parser.add_argument(
        "--support_dir",
        type=str,
        default="assets/supportset/野猪",
        help="Path to biology support set directory",
    )
    parser.add_argument(
        "--query_image",
        type=str,
        default="assets/demo/一群野猪.jpg",
        help="Path to query image of biological specimen",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BioCLIP",
        choices=["BioCLIP", "BioCLIP-2", "BioCLIP-2.5"],
        help="BioCLIP model variant",
    )
    parser.add_argument(
        "--descriptions",
        type=str,
        default="descriptions/wildboar_classification.py",
        help="Path to Python file with class descriptions (optional)",
    )
    parser.add_argument(
        "--enable_nss",
        action="store_true",
        help="Enable Neighborhood Subspace Suppression for fine-grained discrimination",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda/cpu/auto)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🧬 Fewshoter - BioCLIP Biology Classification Example")
    print("=" * 70)
    print("\nBioCLIP: A Vision Foundation Model for Biodiversity")
    print("  - Trained on TreeOfLife-10M/200M dataset")
    print("  - 450K+ biological species coverage")
    print("  - Optimized for fine-grained biological classification")
    print("=" * 70)

    # Configuration optimized for biology
    print("\n[1/5] Configuring for biology classification...")
    config = Config()
    config.model.clip_model_name = args.model
    config.model.device = args.device
    config.model.cache_features = True
    config.model.batch_size = 4

    config.classification.visual_weight = 0.8
    config.classification.text_weight = 0.2
    config.classification.confidence_threshold = 0.6
    config.classification.enable_discriminative_mining = False
    config.classification.enable_impostor_rerank = False

    # Enable NSS if requested (helps distinguish similar species)
    if args.enable_nss:
        config.classification.enable_nss = True
        config.classification.nss_num_neighbors = 3
        print("  ✓ NSS (Neighborhood Subspace Suppression) enabled")

    # Load custom descriptions if provided
    if args.descriptions:
        config.classification.desc_text_path = args.descriptions
        print(f"  ✓ Using custom descriptions: {args.descriptions}")

    print(f"  Model: {args.model}")
    print(f"  Device: {config.model.device}")

    # Create feature extractor with BioCLIP
    print("\n[2/5] Loading BioCLIP model...")
    print("  This may take a moment on first run (downloading from HuggingFace)")
    feature_extractor = create_feature_extractor(config)
    print(f"  ✓ Feature dim: {feature_extractor.feature_dim}")
    print(f"  ✓ Grid size: {feature_extractor.grid_size}x{feature_extractor.grid_size}")

    # Load support set
    print("\n[3/5] Loading biological support set...")
    support_manager = SupportSetManager(
        support_dir=args.support_dir,
        feature_extractor=feature_extractor,
        config=config,
    )
    print(f"  ✓ Loaded {len(support_manager.class_names)} species/classes")

    for class_name in support_manager.class_names:
        stats = support_manager.class_statistics.get(class_name, {})
        num_images = stats.get("num_images", 0)
        num_attrs = stats.get("num_attributes", 0)
        print(f"    - {class_name}: {num_images} images, {num_attrs} attributes")

    # Create classifier
    print("\n[4/5] Building classifier with biological prototypes...")
    classifier = create_classifier(support_manager, config)
    print("  ✓ Classifier ready")

    # Classify
    print("\n[5/5] Classifying biological specimen...")
    print("=" * 70)

    result = classifier.classify(
        args.query_image,
        return_scores=True,
        return_details=True,
    )

    if isinstance(result, tuple):
        predicted_class, details = result
    else:
        predicted_class = result
        details = {}

    confidence = details.get("confidence", 0.0)
    all_scores = details.get("all_scores", {})

    # Display results
    print(f"\n📷 Query Image: {args.query_image}")
    print("\n" + "=" * 70)
    print("🔬 CLASSIFICATION RESULT")
    print("=" * 70)

    print(f"\nPredicted Species: {predicted_class}")
    print(f"Confidence Score: {confidence:.2%}")

    if confidence >= 0.8:
        print("✅ High confidence prediction")
    elif confidence >= 0.6:
        print("⚠️  Moderate confidence - consider verification")
    else:
        print("❌ Low confidence - prediction uncertain")

    print("\n📊 Top-5 Species Matches:")
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (class_name, score) in enumerate(sorted_scores[:5], 1):
        marker = "👉" if class_name == predicted_class else "  "
        confidence_bar = "█" * int(score * 20)
        print(f"{marker} {i}. {class_name:<25} {score:.4f} {confidence_bar}")

    # Show feature contributions
    print("\n📈 Feature Analysis:")
    if "visual_score" in result:
        print(f"  Visual similarity: {result['visual_score']:.4f}")
    if "text_score" in result:
        print(f"  Text similarity:   {result['text_score']:.4f}")

    # Show discriminative mask info if available
    if "discriminative_regions" in result:
        print(f"  Discriminative regions: {result['discriminative_regions']:.1%}")

    # Attributes for predicted class
    attributes = support_manager.get_attributes(predicted_class)
    if attributes:
        print(f"\n🏷️  Key Attributes of {predicted_class}:")
        for attr in attributes[:5]:
            print(f"    • {attr}")

    print("\n" + "=" * 70)
    print("Classification Complete")
    print("=" * 70)

    # Recommendations
    print("\n💡 Recommendations:")
    if confidence < 0.5:
        print("  • Consider collecting more support images for this species")
        print("  • Try using a higher-resolution image")
        print("  • Ensure the diagnostic features are clearly visible")
    elif confidence < 0.7:
        print("  • Prediction looks reasonable but verify with expert")
        print("  • Check if similar species might be confused")

    return 0


if __name__ == "__main__":
    sys.exit(main())
