#!/usr/bin/env python3
"""
Evaluation Script for CLIP Few-Shot Classification System

This script evaluates the performance of the classification system on a test dataset
with ground truth labels. It computes various metrics and generates evaluation reports.

Usage:
    python evaluate.py --prototypes outputs/prototypes.json --test_dir test_data/
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from ..core.config import Config
from ..features.feature_extractor import create_feature_extractor
from ..data.support_manager import SupportSetManager
from ..engine.classifier import create_classifier
from ..core.utils import (
    setup_logging,
    set_random_seed,
    get_image_files,
    ensure_dir,
    Timer,
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP few-shot classification performance"
    )

    # Required arguments
    parser.add_argument(
        "--prototypes",
        type=str,
        required=True,
        help="Path to prototypes file from training",
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Directory containing test images organized by class",
    )

    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results",
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation",
    )

    parser.add_argument(
        "--visual_weight",
        type=float,
        default=0.7,
        help="Weight for visual similarity (0-1)",
    )

    parser.add_argument(
        "--text_weight",
        type=float,
        default=0.3,
        help="Weight for text similarity (0-1)",
    )

    parser.add_argument(
        "--save_predictions", action="store_true", help="Save individual predictions"
    )

    parser.add_argument(
        "--save_plots", action="store_true", help="Save evaluation plots"
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """Setup and return the appropriate device"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            print("CUDA not available, using CPU")
    else:
        device = device_arg
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device = "cpu"

    return device


def load_test_dataset(test_dir: Path) -> tuple:
    """Load test dataset and return image paths with labels"""
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    image_paths = []
    true_labels = []
    class_names = []

    # Get all class directories
    for class_dir in test_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        class_names.append(class_name)

        # Get all images in this class
        class_images = get_image_files(class_dir)

        for img_path in class_images:
            image_paths.append(str(img_path))
            true_labels.append(class_name)

    print(f"Loaded test dataset:")
    print(f"  Classes: {len(set(true_labels))}")
    print(f"  Total images: {len(image_paths)}")

    # Print class distribution
    class_counts = defaultdict(int)
    for label in true_labels:
        class_counts[label] += 1

    print("  Class distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"    {class_name}: {count} images")

    return image_paths, true_labels, sorted(set(true_labels))


def compute_metrics(y_true: list, y_pred: list, class_names: list) -> dict:
    """Compute classification metrics"""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=class_names
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    # Classification report
    report = classification_report(y_true, y_pred, labels=class_names, output_dict=True)

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "per_class_metrics": {
            class_names[i]: {
                "precision": precision_per_class[i],
                "recall": recall_per_class[i],
                "f1": f1_per_class[i],
                "support": int(support_per_class[i]),
            }
            for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: Path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))

    # Normalize confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_per_class_metrics(metrics: dict, output_path: Path):
    """Plot per-class metrics"""
    class_names = list(metrics["per_class_metrics"].keys())
    precisions = [metrics["per_class_metrics"][cls]["precision"] for cls in class_names]
    recalls = [metrics["per_class_metrics"][cls]["recall"] for cls in class_names]
    f1_scores = [metrics["per_class_metrics"][cls]["f1"] for cls in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.bar(x - width, precisions, width, label="Precision", alpha=0.8)
    ax.bar(x, recalls, width, label="Recall", alpha=0.8)
    ax.bar(x + width, f1_scores, width, label="F1-Score", alpha=0.8)

    ax.set_xlabel("Classes")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_confidence_distribution(results: list, output_path: Path):
    """Analyze and plot confidence distribution"""
    confidences = [r["confidence"] for r in results if "confidence" in r]
    correct_confidences = [
        r["confidence"] for r in results if r["predicted_class"] == r.get("true_class")
    ]
    incorrect_confidences = [
        r["confidence"] for r in results if r["predicted_class"] != r.get("true_class")
    ]

    plt.figure(figsize=(12, 6))

    # Plot 1: Overall confidence distribution
    plt.subplot(1, 2, 1)
    plt.hist(confidences, bins=30, alpha=0.7, color="blue", edgecolor="black")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Overall Confidence Distribution")
    plt.grid(True, alpha=0.3)

    # Plot 2: Correct vs Incorrect predictions
    plt.subplot(1, 2, 2)
    plt.hist(correct_confidences, bins=20, alpha=0.7, label="Correct", color="green")
    plt.hist(incorrect_confidences, bins=20, alpha=0.7, label="Incorrect", color="red")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Confidence: Correct vs Incorrect")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main evaluation function"""
    args = parse_arguments()

    # Validate weights
    if abs(args.visual_weight + args.text_weight - 1.0) > 1e-6:
        print("Error: visual_weight + text_weight must equal 1.0")
        sys.exit(1)

    # Setup paths
    prototypes_path = Path(args.prototypes)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Setup logging
    log_file = output_dir / "evaluation.log"
    logger = setup_logging(args.log_level, str(log_file))

    logger.info("Starting CLIP few-shot classification evaluation")
    logger.info(f"Arguments: {vars(args)}")

    # Validate inputs
    if not prototypes_path.exists():
        logger.error(f"Prototypes file not found: {prototypes_path}")
        sys.exit(1)

    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        sys.exit(1)

    # Setup device
    device = setup_device(args.device)
    set_random_seed(42)

    try:
        # Load test dataset
        logger.info("Loading test dataset...")
        image_paths, true_labels, test_class_names = load_test_dataset(test_dir)

        # Load classifier
        logger.info(f"Loading classifier from: {prototypes_path}")

        # Load config from prototypes
        with open(prototypes_path, "r") as f:
            metadata = json.load(f)

        config = Config.from_dict(metadata.get("config", {}))
        config.model.device = device
        config.classification.visual_weight = args.visual_weight
        config.classification.text_weight = args.text_weight

        # Create components
        feature_extractor = create_feature_extractor(config)
        support_manager = SupportSetManager.from_prototypes(
            prototypes_path, feature_extractor
        )
        classifier = create_classifier(support_manager, config)

        # Check class compatibility
        support_classes = set(support_manager.class_names)
        test_classes = set(test_class_names)

        if not test_classes.issubset(support_classes):
            missing_classes = test_classes - support_classes
            logger.error(f"Test classes not in support set: {missing_classes}")
            sys.exit(1)

        logger.info(
            f"Evaluating on {len(test_classes)} classes with {len(image_paths)} images"
        )

        # Perform inference
        logger.info("Running inference on test set...")

        with Timer("Test set inference"):
            results = classifier.classify_batch(
                image_paths, batch_size=args.batch_size, return_scores=True
            )

        # Add true labels to results
        for i, result in enumerate(results):
            result["true_class"] = true_labels[i]

        # Extract predictions
        predicted_labels = [r["predicted_class"] for r in results]

        # Compute metrics
        logger.info("Computing evaluation metrics...")
        metrics = compute_metrics(true_labels, predicted_labels, test_class_names)

        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision (weighted): {metrics['precision_weighted']:.3f}")
        print(f"Recall (weighted): {metrics['recall_weighted']:.3f}")
        print(f"F1-Score (weighted): {metrics['f1_weighted']:.3f}")
        print("\nPer-class results:")
        for class_name in test_class_names:
            class_metrics = metrics["per_class_metrics"][class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.3f}")
            print(f"    Recall: {class_metrics['recall']:.3f}")
            print(f"    F1-Score: {class_metrics['f1']:.3f}")
            print(f"    Support: {class_metrics['support']}")
        print("=" * 60)

        # Save results
        logger.info("Saving evaluation results...")

        # Save metrics
        metrics_path = output_dir / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save predictions if requested
        if args.save_predictions:
            predictions_path = output_dir / "predictions.json"
            with open(predictions_path, "w") as f:
                json.dump(results, f, indent=2)

        # Generate plots if requested
        if args.save_plots:
            logger.info("Generating evaluation plots...")

            # Confusion matrix
            cm_path = output_dir / "confusion_matrix.png"
            plot_confusion_matrix(
                np.array(metrics["confusion_matrix"]), test_class_names, cm_path
            )

            # Per-class metrics
            metrics_plot_path = output_dir / "per_class_metrics.png"
            plot_per_class_metrics(metrics, metrics_plot_path)

            # Confidence distribution
            confidence_path = output_dir / "confidence_distribution.png"
            analyze_confidence_distribution(results, confidence_path)

        logger.info(f"Evaluation completed. Results saved to: {output_dir}")
        print(f"\nResults saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
