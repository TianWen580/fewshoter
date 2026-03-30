#!/usr/bin/env python3
"""
Batch Classification Example

Demonstrates how to classify multiple images efficiently with batch processing
and save results to a file.

Usage:
    python batch_classification.py \
        --prototypes ./outputs/prototypes.json \
        --query_dir ./test_images \
        --output results.json \
        --model BioCLIP-2
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from fewshoter import (
    Config,
    create_feature_extractor,
    SupportSetManager,
    create_classifier,
)
from fewshoter.core.utils import get_image_files


def main():
    parser = argparse.ArgumentParser(description="Batch classification of multiple images")
    parser.add_argument(
        "--prototypes",
        type=str,
        required=True,
        help="Path to saved prototypes.json",
    )
    parser.add_argument(
        "--query_dir",
        type=str,
        required=True,
        help="Directory containing query images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="batch_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BioCLIP-2",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "BioCLIP", "BioCLIP-2", "BioCLIP-2.5"],
        help="Model to use",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum confidence to accept prediction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cuda/cpu/auto)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("📦 Batch Classification Example")
    print("=" * 60)

    # Setup
    print("\n[1/4] Initializing...")
    config = Config()
    config.model.clip_model_name = args.model
    config.model.device = args.device
    config.classification.confidence_threshold = args.confidence_threshold

    print(f"  Model: {args.model}")
    print(f"  Confidence threshold: {args.confidence_threshold}")

    # Create feature extractor
    print("\n[2/4] Loading model...")
    feature_extractor = create_feature_extractor(config)
    print(f"  ✓ Model loaded")

    # Load support set from prototypes
    print("\n[3/4] Loading support set...")
    support_manager = SupportSetManager.from_prototypes(
        args.prototypes,
        feature_extractor=feature_extractor,
    )
    print(f"  ✓ Loaded {len(support_manager.class_names)} classes")
    print(
        f"     Classes: {', '.join(support_manager.class_names[:5])}{'...' if len(support_manager.class_names) > 5 else ''}"
    )

    # Create classifier
    classifier = create_classifier(support_manager, config)

    # Get query images
    query_images = get_image_files(args.query_dir)
    print(f"\n[4/4] Found {len(query_images)} query images")

    if len(query_images) == 0:
        print("  ❌ No images found in query directory")
        return 1

    # Process batch
    print("\n" + "=" * 60)
    print("🔄 Processing Images...")
    print("=" * 60)

    results = []
    correct = 0
    high_confidence = 0

    for img_path in tqdm(query_images, desc="Classifying"):
        try:
            result = classifier.classify(
                img_path,
                return_scores=True,
                top_k=3,
            )

            # Try to extract ground truth from filename or parent folder
            gt_label = None
            parts = Path(img_path).stem.split("_")
            if len(parts) > 0:
                gt_label = parts[0]  # Assume format: label_001.jpg

            is_correct = gt_label and result["label"] == gt_label
            if is_correct:
                correct += 1

            if result["confidence"] >= args.confidence_threshold:
                high_confidence += 1

            result_entry = {
                "image": str(img_path),
                "predicted": result["label"],
                "confidence": float(result["confidence"]),
                "ground_truth": gt_label,
                "correct": is_correct if gt_label else None,
                "top3": [
                    {"class": k, "score": float(v)}
                    for k, v in sorted(
                        result.get("all_scores", {}).items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3]
                ],
            }
            results.append(result_entry)

        except Exception as e:
            print(f"\n  ⚠️  Error processing {img_path}: {e}")
            results.append(
                {
                    "image": str(img_path),
                    "error": str(e),
                }
            )

    # Statistics
    print("\n" + "=" * 60)
    print("📊 Results Summary")
    print("=" * 60)

    print(f"\nTotal images processed: {len(results)}")
    print(
        f"High confidence (≥{args.confidence_threshold}): {high_confidence} ({high_confidence / len(results) * 100:.1f}%)"
    )

    if correct > 0 or any(r.get("ground_truth") for r in results):
        accuracy = correct / len([r for r in results if r.get("ground_truth")]) * 100
        print(f"Accuracy (if ground truth available): {accuracy:.1f}%")

    # Class distribution
    print("\n📈 Predicted Class Distribution:")
    class_counts = {}
    for r in results:
        if "predicted" in r:
            class_counts[r["predicted"]] = class_counts.get(r["predicted"], 0) + 1

    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * (count * 50 // len(results))
        print(f"  {class_name:<20} {count:>3} {bar}")

    # Save results
    output_data = {
        "config": {
            "model": args.model,
            "confidence_threshold": args.confidence_threshold,
            "num_classes": len(support_manager.class_names),
        },
        "statistics": {
            "total": len(results),
            "high_confidence": high_confidence,
            "accuracy": accuracy if correct > 0 else None,
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: {args.output}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
