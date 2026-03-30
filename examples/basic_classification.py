#!/usr/bin/env python3
"""
Basic Classification Example

This example demonstrates how to use Fewshoter for few-shot image classification
with a pre-built support set.

Usage:
    python basic_classification.py --support_dir ./support_set --query_image ./query.jpg
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path if running from examples folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from fewshoter import (
    Config,
    create_feature_extractor,
    SupportSetManager,
    create_classifier,
)


def main():
    parser = argparse.ArgumentParser(description="Basic few-shot classification example")
    parser.add_argument(
        "--support_dir",
        type=str,
        required=True,
        help="Path to support set directory",
    )
    parser.add_argument(
        "--query_image",
        type=str,
        required=True,
        help="Path to query image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "BioCLIP", "BioCLIP-2", "BioCLIP-2.5"],
        help="CLIP model to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda/cpu/auto)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Return top-k predictions",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Fewshoter - Basic Classification Example")
    print("=" * 60)

    # 1. Load configuration
    print("\n[1/4] Loading configuration...")
    config = Config()
    config.model.clip_model_name = args.model
    config.model.device = args.device
    print(f"  Model: {args.model}")
    print(f"  Device: {config.model.device}")

    # 2. Create feature extractor
    print("\n[2/4] Creating feature extractor...")
    feature_extractor = create_feature_extractor(config)
    print(f"  Feature dim: {feature_extractor.feature_dim}")
    print(f"  Grid size: {feature_extractor.grid_size}")

    # 3. Load support set
    print("\n[3/4] Loading support set...")
    support_manager = SupportSetManager(
        support_dir=args.support_dir,
        feature_extractor=feature_extractor,
        config=config,
    )
    print(f"  Classes: {support_manager.class_names}")
    print(f"  Total classes: {len(support_manager.class_names)}")

    # Show class statistics
    for class_name in support_manager.class_names:
        stats = support_manager.class_statistics.get(class_name, {})
        num_images = stats.get("num_images", 0)
        print(f"    - {class_name}: {num_images} images")

    # 4. Create classifier
    print("\n[4/4] Creating classifier...")
    classifier = create_classifier(support_manager, config)
    print("  Classifier ready")

    # 5. Classify query image
    print("\n" + "=" * 60)
    print("Classification Result")
    print("=" * 60)

    result = classifier.classify(
        args.query_image,
        return_scores=True,
        top_k=args.top_k,
    )

    predicted_class = result["label"]
    confidence = result["confidence"]
    all_scores = result.get("all_scores", {})

    print(f"\nQuery: {args.query_image}")
    print(f"\nPrediction: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

    print(f"\nTop-{args.top_k} predictions:")
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (class_name, score) in enumerate(sorted_scores[: args.top_k], 1):
        marker = "✓" if class_name == predicted_class else " "
        print(f"  {marker} {i}. {class_name}: {score:.4f}")

    # Additional details
    if "visual_score" in result:
        print(f"\nVisual similarity: {result['visual_score']:.4f}")
    if "text_score" in result:
        print(f"Text similarity: {result['text_score']:.4f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
