#!/usr/bin/env python3
"""
Flask API Server for CLIP Few-Shot Classification

This module provides a REST API server for the CLIP few-shot classification system.
It supports image upload, batch processing, and real-time classification.

Usage:
    python -m clip_fewshot.api_server --prototypes outputs/prototypes.json --port 5000
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any
import json

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import torch

from ..core.config import Config
from ..features.feature_extractor import create_feature_extractor
from ..data.support_manager import SupportSetManager
from ..engine.classifier import create_classifier
from ..core.utils import setup_logging, ensure_dir, Timer


# Global variables for the classifier
classifier = None
config = None
upload_folder = None


def create_app(
    prototypes_path: str,
    upload_dir: str = "uploads",
    max_content_length: int = 16 * 1024 * 1024,
) -> Flask:
    """Create and configure Flask application"""
    global classifier, config, upload_folder

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_content_length  # 16MB max file size

    # Setup upload directory
    upload_folder = Path(upload_dir)
    ensure_dir(upload_folder)

    # Load classifier
    logger = logging.getLogger(__name__)
    logger.info(f"Loading classifier from: {prototypes_path}")

    # Load prototypes and create classifier
    with open(prototypes_path, "r") as f:
        metadata = json.load(f)

    config = Config.from_dict(metadata.get("config", {}))
    feature_extractor = create_feature_extractor(config)
    support_manager = SupportSetManager.from_prototypes(
        prototypes_path, feature_extractor
    )
    classifier = create_classifier(support_manager, config)

    logger.info(f"Classifier loaded with {len(support_manager.class_names)} classes")

    return app


app = Flask(__name__)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    allowed_extensions = {"png", "jpg", "jpeg", "gif", "bmp", "tiff"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": time.time(),
            "classifier_loaded": classifier is not None,
        }
    )


@app.route("/info", methods=["GET"])
def get_info():
    """Get system information"""
    if classifier is None:
        return jsonify({"error": "Classifier not loaded"}), 500

    system_info = classifier.get_system_info()
    return jsonify(
        {"system_info": system_info, "api_version": "0.2.1", "timestamp": time.time()}
    )


@app.route("/classify", methods=["POST"])
def classify_image():
    """Classify a single uploaded image"""
    if classifier is None:
        return jsonify({"error": "Classifier not loaded"}), 500

    # Check if file is present
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        safe_filename = f"{timestamp}_{filename}"
        file_path = upload_folder / safe_filename
        file.save(file_path)

        # Get optional parameters
        return_scores = request.form.get("return_scores", "true").lower() == "true"
        top_k = int(request.form.get("top_k", 3))

        # Classify image
        with Timer("Image classification"):
            if return_scores:
                predicted_class, details = classifier.classify(
                    str(file_path), return_scores=True
                )

                # Get top-k predictions
                top_predictions = classifier.get_top_k_predictions(
                    details["all_scores"], k=top_k
                )

                # Evaluate confidence
                confidence_analysis = classifier.evaluate_confidence(
                    details["all_scores"]
                )

                result = {
                    "predicted_class": predicted_class,
                    "confidence": details["confidence"],
                    "top_predictions": [
                        {"class": cls, "score": float(score)}
                        for cls, score in top_predictions
                    ],
                    "confidence_analysis": confidence_analysis,
                    "processing_time": time.time(),
                }
            else:
                predicted_class = classifier.classify(
                    str(file_path), return_scores=False
                )
                result = {
                    "predicted_class": predicted_class,
                    "processing_time": time.time(),
                }

        # Clean up uploaded file
        file_path.unlink()

        return jsonify(result)

    except Exception as e:
        # Clean up file if it exists
        if "file_path" in locals() and file_path.exists():
            file_path.unlink()

        logging.error(f"Classification failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/classify_batch", methods=["POST"])
def classify_batch():
    """Classify multiple uploaded images"""
    if classifier is None:
        return jsonify({"error": "Classifier not loaded"}), 500

    # Check if files are present
    if "images" not in request.files:
        return jsonify({"error": "No image files provided"}), 400

    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected"}), 400

    # Validate all files
    for file in files:
        if not allowed_file(file.filename):
            return jsonify({"error": f"Invalid file type: {file.filename}"}), 400

    try:
        # Save all uploaded files
        file_paths = []
        timestamp = str(int(time.time()))

        for i, file in enumerate(files):
            filename = secure_filename(file.filename)
            safe_filename = f"{timestamp}_{i:03d}_{filename}"
            file_path = upload_folder / safe_filename
            file.save(file_path)
            file_paths.append(file_path)

        # Get optional parameters
        batch_size = int(request.form.get("batch_size", 32))
        return_scores = request.form.get("return_scores", "true").lower() == "true"

        # Classify images
        with Timer(f"Batch classification of {len(file_paths)} images"):
            results = classifier.classify_batch(
                [str(p) for p in file_paths],
                batch_size=batch_size,
                return_scores=return_scores,
            )

        # Add original filenames to results
        for i, result in enumerate(results):
            result["original_filename"] = files[i].filename
            result["processing_time"] = time.time()

        # Clean up uploaded files
        for file_path in file_paths:
            if file_path.exists():
                file_path.unlink()

        return jsonify(
            {"results": results, "total_images": len(results), "batch_size": batch_size}
        )

    except Exception as e:
        # Clean up files if they exist
        if "file_paths" in locals():
            for file_path in file_paths:
                if file_path.exists():
                    file_path.unlink()

        logging.error(f"Batch classification failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/update_weights", methods=["POST"])
def update_weights():
    """Update classification weights"""
    if classifier is None:
        return jsonify({"error": "Classifier not loaded"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        visual_weight = float(data.get("visual_weight", 0.7))
        text_weight = float(data.get("text_weight", 0.3))

        # Validate weights
        if abs(visual_weight + text_weight - 1.0) > 1e-6:
            return jsonify({"error": "Weights must sum to 1.0"}), 400

        # Update weights
        classifier.update_weights(visual_weight, text_weight)

        return jsonify(
            {
                "message": "Weights updated successfully",
                "visual_weight": visual_weight,
                "text_weight": text_weight,
            }
        )

    except Exception as e:
        logging.error(f"Weight update failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    """Clear all caches"""
    if classifier is None:
        return jsonify({"error": "Classifier not loaded"}), 500

    try:
        classifier.clear_caches()
        return jsonify({"message": "Caches cleared successfully"})

    except Exception as e:
        logging.error(f"Cache clearing failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({"error": "File too large"}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({"error": "Internal server error"}), 500


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="CLIP Few-Shot Classification API Server"
    )

    parser.add_argument(
        "--prototypes", type=str, required=True, help="Path to prototypes file"
    )

    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host address to bind to"
    )

    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")

    parser.add_argument(
        "--upload_dir",
        type=str,
        default="uploads",
        help="Directory for temporary file uploads",
    )

    parser.add_argument(
        "--max_file_size", type=int, default=16, help="Maximum file size in MB"
    )

    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main():
    """Main function to run the API server"""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Validate prototypes file
    if not Path(args.prototypes).exists():
        logger.error(f"Prototypes file not found: {args.prototypes}")
        return 1

    try:
        # Create Flask app
        global app
        app = create_app(
            prototypes_path=args.prototypes,
            upload_dir=args.upload_dir,
            max_content_length=args.max_file_size * 1024 * 1024,
        )

        logger.info(f"Starting API server on {args.host}:{args.port}")
        logger.info(f"Upload directory: {args.upload_dir}")
        logger.info(f"Max file size: {args.max_file_size}MB")

        # Run the server
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
