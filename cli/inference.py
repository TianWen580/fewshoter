#!/usr/bin/env python3
"""
Inference Script for CLIP Few-Shot Classification System

This script performs inference on query images using pre-computed prototypes
from the training phase. It supports single image classification and batch processing.

Usage:
    python inference.py --prototypes outputs/prototypes.json --image query.jpg
    python inference.py --prototypes outputs/prototypes.json --image_dir test_images/
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import json
from PIL import Image

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
        description="Perform inference with CLIP few-shot classification"
    )

    # Required arguments
    parser.add_argument(
        "--prototypes",
        type=str,
        required=True,
        help="Path to prototypes file from training",
    )

    # Input specification (one of these required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", type=str, help="Path to single image for classification"
    )
    input_group.add_argument(
        "--image_dir",
        type=str,
        help="Directory containing images for batch classification",
    )
    input_group.add_argument(
        "--image_list", type=str, help="Text file containing list of image paths"
    )

    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing multiple images",
    )

    parser.add_argument(
        "--top_k", type=int, default=3, help="Number of top predictions to return"
    )

    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions",
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation",
    )

    parser.add_argument(
        "--save_details",
        action="store_true",
        help="Save detailed analysis for each prediction",
    )

    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save attention visualizations",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
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


def load_image_list(image_list_path: str) -> list:
    """Load list of image paths from text file"""
    image_paths = []
    with open(image_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                image_paths.append(line)
    return image_paths


def get_input_images(args) -> list:
    """Get list of input images based on arguments"""
    if args.image:
        return [args.image]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        return get_image_files(image_dir)
    elif args.image_list:
        if not Path(args.image_list).exists():
            raise FileNotFoundError(f"Image list file not found: {args.image_list}")
        return load_image_list(args.image_list)
    else:
        raise ValueError("No input images specified")


def format_prediction_result(result: dict, top_k: int = 3) -> str:
    """Format prediction result for display"""
    lines = []
    lines.append(f"Image: {result['image']}")
    lines.append(f"Predicted: {result['predicted_class']}")
    lines.append(f"Confidence: {result['confidence']:.3f}")

    if "all_scores" in result:
        lines.append("Top predictions:")
        sorted_scores = sorted(
            result["all_scores"].items(), key=lambda x: x[1], reverse=True
        )
        for i, (class_name, score) in enumerate(sorted_scores[:top_k]):
            lines.append(f"  {i + 1}. {class_name}: {score:.3f}")

    return "\n".join(lines)


def save_results(results: list, output_dir: Path, save_details: bool = False):
    """Save inference results to files"""
    ensure_dir(output_dir)

    # Save summary results
    summary_results = []
    for result in results:
        summary = {
            "image": result["image"],
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
        }
        if "all_scores" in result:
            # Add top-3 predictions
            sorted_scores = sorted(
                result["all_scores"].items(), key=lambda x: x[1], reverse=True
            )
            summary["top_3"] = [
                {"class": cls, "score": score} for cls, score in sorted_scores[:3]
            ]
        summary_results.append(summary)

    summary_path = output_dir / "inference_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    # Save detailed results if requested
    if save_details:
        detailed_path = output_dir / "inference_detailed.json"
        with open(detailed_path, "w") as f:
            json.dump(results, f, indent=2)

    # Save CSV summary
    csv_path = output_dir / "inference_results.csv"
    with open(csv_path, "w") as f:
        f.write("image,predicted_class,confidence\n")
        for result in results:
            f.write(
                f"{result['image']},{result['predicted_class']},{result['confidence']:.3f}\n"
            )

    return summary_path, csv_path


def main():
    """Main inference function"""
    args = parse_arguments()

    # Validate weights
    if abs(args.visual_weight + args.text_weight - 1.0) > 1e-6:
        print("Error: visual_weight + text_weight must equal 1.0")
        sys.exit(1)

    # Setup paths
    prototypes_path = Path(args.prototypes)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Setup logging
    log_file = output_dir / "inference.log"
    logger = setup_logging(args.log_level, str(log_file))

    logger.info("Starting CLIP few-shot classification inference")
    logger.info(f"Arguments: {vars(args)}")

    # Validate prototypes file
    if not prototypes_path.exists():
        logger.error(f"Prototypes file not found: {prototypes_path}")
        sys.exit(1)

    # Setup device
    device = setup_device(args.device)

    # Set random seed
    set_random_seed(args.seed)

    # Get input images
    try:
        image_paths = get_input_images(args)
        logger.info(f"Found {len(image_paths)} images for inference")
    except Exception as e:
        logger.error(f"Failed to get input images: {e}")
        sys.exit(1)

    try:
        # Load prototypes and create support manager
        logger.info(f"Loading prototypes from: {prototypes_path}")

        # First load the config from prototypes file
        with open(prototypes_path, "r") as f:
            metadata = json.load(f)

        if "config" in metadata:
            config = Config.from_dict(metadata["config"])
        else:
            config = Config()

        # Override device
        config.model.device = device
        config.classification.visual_weight = args.visual_weight
        config.classification.text_weight = args.text_weight
        config.classification.confidence_threshold = args.confidence_threshold

        # Create feature extractor
        feature_extractor = create_feature_extractor(config)

        # Load support manager from prototypes
        support_manager = SupportSetManager.from_prototypes(
            prototypes_path, feature_extractor
        )

        # Create classifier
        classifier = create_classifier(support_manager, config)

        logger.info(
            f"Classifier initialized with {len(support_manager.class_names)} classes"
        )
        logger.info(f"Classes: {', '.join(support_manager.class_names)}")

        # Perform inference
        logger.info("Starting inference...")

        if len(image_paths) == 1:
            # Single image inference
            image_path = image_paths[0]
            logger.info(f"Classifying single image: {image_path}")

            with Timer("Single image classification"):
                pred_class, details = classifier.classify(
                    image_path, return_scores=True, return_details=args.save_details
                )

            # Display result
            result = {
                "image": str(image_path),
                "predicted_class": pred_class,
                "confidence": details["confidence"],
                "all_scores": details["all_scores"],
            }

            if args.save_details:
                result["detailed_analysis"] = details["detailed_analysis"]

            print("\n" + "=" * 50)
            print("CLASSIFICATION RESULT")
            print("=" * 50)
            print(format_prediction_result(result, args.top_k))
            print("=" * 50)

            results = [result]

        else:
            # Batch inference
            logger.info(f"Performing batch inference on {len(image_paths)} images")

            with Timer("Batch inference"):
                results = classifier.classify_batch(
                    image_paths, batch_size=args.batch_size, return_scores=True
                )

            # Display summary
            successful_results = [r for r in results if r["predicted_class"] != "ERROR"]
            failed_results = [r for r in results if r["predicted_class"] == "ERROR"]

            print(f"\nInference completed:")
            print(f"  Successful: {len(successful_results)}")
            print(f"  Failed: {len(failed_results)}")

            if successful_results:
                # Show confidence distribution
                confidences = [r["confidence"] for r in successful_results]
                print(
                    f"  Average confidence: {sum(confidences) / len(confidences):.3f}"
                )
                print(f"  Min confidence: {min(confidences):.3f}")
                print(f"  Max confidence: {max(confidences):.3f}")

        # Save results
        logger.info("Saving results...")
        summary_path, csv_path = save_results(results, output_dir, args.save_details)

        logger.info(f"Results saved:")
        logger.info(f"  Summary: {summary_path}")
        logger.info(f"  CSV: {csv_path}")

        print(f"\nResults saved to: {output_dir}")
        print(f"  Summary: {summary_path}")
        print(f"  CSV: {csv_path}")

        logger.info("Inference completed successfully!")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
