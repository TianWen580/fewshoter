"""
Utility functions for CLIP Few-Shot Classification System

This module provides common utility functions used across the system,
including image processing, logging setup, file operations, and helper functions.
"""

import os
import json
import logging
import random
import numpy as np
import torch
import cv2
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper())

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(console_handler)

    # Setup file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_image(image_path: Union[str, Path], convert_rgb: bool = True) -> Image.Image:
    """Load image from file path"""
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path)
        if convert_rgb and image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def save_image(image: Union[Image.Image, np.ndarray], output_path: Union[str, Path]):
    """Save image to file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    image.save(output_path)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute cosine similarity between tensors"""
    return torch.nn.functional.cosine_similarity(a, b, dim=dim)


def normalize_tensor(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize tensor to unit length"""
    return tensor / tensor.norm(dim=dim, keepdim=True)


def create_attention_heatmap(
    attention_map: np.ndarray,
    original_image: Image.Image,
    alpha: float = 0.3,
    colormap: str = "jet",
) -> Image.Image:
    """Create attention heatmap overlay on original image"""
    # Resize attention map to match image size
    h, w = original_image.size[1], original_image.size[0]
    attention_resized = cv2.resize(attention_map, (w, h))

    # Normalize attention map
    attention_norm = (attention_resized - attention_resized.min()) / (
        attention_resized.max() - attention_resized.min()
    )

    # Convert to heatmap
    heatmap = cv2.applyColorMap(
        (attention_norm * 255).astype(np.uint8),
        getattr(cv2, f"COLORMAP_{colormap.upper()}"),
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert original image to numpy array
    img_array = np.array(original_image)

    # Blend images
    blended = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)

    return Image.fromarray(blended)


def save_results(results: Dict[str, Any], output_path: Union[str, Path]):
    """Save classification results to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays and tensors to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    results_json = convert_for_json(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)


def load_class_names(class_file: Union[str, Path]) -> List[str]:
    """Load class names from text file"""
    class_file = Path(class_file)

    if not class_file.exists():
        raise FileNotFoundError(f"Class file not found: {class_file}")

    with open(class_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    return classes


def create_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Create and optionally save confusion matrix"""
    from sklearn.metrics import confusion_matrix

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    # Plot confusion matrix if output path is provided
    if output_path:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    return cm


def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Calculate classification metrics"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device"""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    return device


def memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
    else:
        import psutil

        memory = psutil.virtual_memory()
        return f"RAM Usage: {memory.percent}% ({memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB)"


def ensure_dir(path: Union[str, Path]):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_image_files(
    directory: Union[str, Path],
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
) -> List[Path]:
    """Get all image files in directory"""
    directory = Path(directory)
    image_files = []

    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(image_files)


class Timer:
    """Simple timer context manager"""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )
        if self.start_time:
            self.start_time.record()
        else:
            import time

            self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available() and hasattr(self.start_time, "record"):
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            elapsed = (
                self.start_time.elapsed_time(end_time) / 1000
            )  # Convert to seconds
        else:
            import time

            elapsed = time.time() - self.start_time

        # print(f"{self.name} took {elapsed:.3f} seconds")


def batch_process(items: List[Any], batch_size: int, process_fn):
    """Process items in batches"""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_results = process_fn(batch)
        results.extend(batch_results)
    return results
