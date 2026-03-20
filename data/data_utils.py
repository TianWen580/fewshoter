"""
Data Loading Utilities for CLIP Few-Shot Classification

This module provides utilities for loading and preprocessing datasets,
creating support sets, and managing data splits for few-shot learning.

Key Features:
- Support set creation from existing datasets
- Data augmentation for few-shot scenarios
- Dataset splitting and validation
- Image preprocessing pipelines
- Batch data loading utilities
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import defaultdict
import json

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from ..core.utils import get_image_files, ensure_dir, load_image
from ..core.config import Config


class FewShotDataset(Dataset):
    """Dataset class for few-shot learning scenarios"""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[str],
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize few-shot dataset

        Args:
            image_paths: List of image file paths
            labels: List of corresponding class labels
            transform: Optional image transformations
            class_to_idx: Optional mapping from class names to indices
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        # Create class mapping if not provided
        if class_to_idx is None:
            unique_labels = sorted(set(labels))
            self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = load_image(image_path)
        except Exception as e:
            logging.warning(f"Failed to load image {image_path}: {e}")
            # Return a dummy image
            image = Image.new("RGB", (224, 224), color="black")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label_idx = self.class_to_idx[label]

        return {
            "image": image,
            "label": label_idx,
            "label_name": label,
            "image_path": image_path,
        }

    def get_class_samples(
        self, class_name: str, num_samples: Optional[int] = None
    ) -> List[int]:
        """Get indices of samples for a specific class"""
        class_indices = [
            i for i, label in enumerate(self.labels) if label == class_name
        ]

        if num_samples is not None and num_samples < len(class_indices):
            class_indices = random.sample(class_indices, num_samples)

        return class_indices


class DatasetSplitter:
    """Utility for splitting datasets into support and query sets"""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)

    def create_few_shot_split(
        self,
        dataset_dir: Union[str, Path],
        output_dir: Union[str, Path],
        shots_per_class: int = 5,
        query_per_class: int = 20,
        min_images_per_class: int = 25,
    ) -> Dict[str, int]:
        """
        Create few-shot split from a dataset directory

        Args:
            dataset_dir: Directory with class subdirectories
            output_dir: Output directory for split
            shots_per_class: Number of support examples per class
            query_per_class: Number of query examples per class
            min_images_per_class: Minimum images required per class

        Returns:
            Dictionary with split statistics
        """
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)

        support_dir = output_dir / "support"
        query_dir = output_dir / "query"

        ensure_dir(support_dir)
        ensure_dir(query_dir)

        stats = {
            "total_classes": 0,
            "total_support_images": 0,
            "total_query_images": 0,
            "skipped_classes": [],
        }

        # Process each class
        for class_dir in dataset_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            image_files = get_image_files(class_dir)

            # Skip classes with insufficient images
            if len(image_files) < min_images_per_class:
                stats["skipped_classes"].append(class_name)
                logging.warning(
                    f"Skipping class {class_name}: only {len(image_files)} images"
                )
                continue

            # Shuffle images
            random.shuffle(image_files)

            # Split into support and query
            support_images = image_files[:shots_per_class]
            remaining_images = image_files[shots_per_class:]
            query_images = remaining_images[:query_per_class]

            # Create class directories
            support_class_dir = support_dir / class_name
            query_class_dir = query_dir / class_name

            ensure_dir(support_class_dir)
            ensure_dir(query_class_dir)

            # Copy support images
            for i, img_path in enumerate(support_images):
                dst_path = support_class_dir / f"support_{i:03d}{img_path.suffix}"
                shutil.copy2(img_path, dst_path)

            # Copy query images
            for i, img_path in enumerate(query_images):
                dst_path = query_class_dir / f"query_{i:03d}{img_path.suffix}"
                shutil.copy2(img_path, dst_path)

            stats["total_classes"] += 1
            stats["total_support_images"] += len(support_images)
            stats["total_query_images"] += len(query_images)

            logging.info(
                f"Processed class {class_name}: "
                f"{len(support_images)} support, {len(query_images)} query"
            )

        # Save split metadata
        metadata = {
            "shots_per_class": shots_per_class,
            "query_per_class": query_per_class,
            "min_images_per_class": min_images_per_class,
            "random_seed": self.random_seed,
            "statistics": stats,
        }

        metadata_path = output_dir / "split_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Dataset split completed: {stats}")
        return stats

    def create_balanced_split(
        self,
        dataset_dir: Union[str, Path],
        output_dir: Union[str, Path],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, int]:
        """Create balanced train/val/test split"""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)

        # Create split directories
        for split in ["train", "val", "test"]:
            ensure_dir(output_dir / split)

        stats = {"train": 0, "val": 0, "test": 0}

        # Process each class
        for class_dir in dataset_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            image_files = get_image_files(class_dir)

            if len(image_files) < 3:  # Need at least 3 images for all splits
                logging.warning(f"Skipping class {class_name}: insufficient images")
                continue

            # Shuffle and split
            random.shuffle(image_files)

            n_train = int(len(image_files) * train_ratio)
            n_val = int(len(image_files) * val_ratio)

            train_images = image_files[:n_train]
            val_images = image_files[n_train : n_train + n_val]
            test_images = image_files[n_train + n_val :]

            # Copy images to respective directories
            for split, images in [
                ("train", train_images),
                ("val", val_images),
                ("test", test_images),
            ]:
                split_class_dir = output_dir / split / class_name
                ensure_dir(split_class_dir)

                for i, img_path in enumerate(images):
                    dst_path = split_class_dir / f"{split}_{i:03d}{img_path.suffix}"
                    shutil.copy2(img_path, dst_path)

                stats[split] += len(images)

        return stats


class DataAugmentation:
    """Data augmentation utilities for few-shot learning"""

    def __init__(self, config: Config):
        self.config = config

    def augment_support_set(
        self,
        support_dir: Union[str, Path],
        output_dir: Union[str, Path],
        augmentations_per_image: int = 3,
    ) -> int:
        """
        Augment support set images to increase diversity

        Args:
            support_dir: Directory containing support images
            output_dir: Output directory for augmented images
            augmentations_per_image: Number of augmentations per original image

        Returns:
            Total number of augmented images created
        """
        try:
            from torchvision import transforms
        except ImportError:
            logging.error("torchvision required for data augmentation")
            return 0

        support_dir = Path(support_dir)
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        # Define augmentation pipeline
        augment_transform = transforms.Compose(
            [
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        total_augmented = 0

        # Process each class
        for class_dir in support_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            output_class_dir = output_dir / class_name
            ensure_dir(output_class_dir)

            # Copy original images first
            original_images = get_image_files(class_dir)
            for img_path in original_images:
                dst_path = output_class_dir / f"original_{img_path.name}"
                shutil.copy2(img_path, dst_path)

            # Generate augmentations
            for img_path in original_images:
                try:
                    image = load_image(img_path)

                    for aug_idx in range(augmentations_per_image):
                        # Apply augmentation
                        augmented_image = augment_transform(image)

                        # Save augmented image
                        aug_name = f"aug_{aug_idx:02d}_{img_path.stem}.jpg"
                        aug_path = output_class_dir / aug_name
                        augmented_image.save(aug_path, quality=95)

                        total_augmented += 1

                except Exception as e:
                    logging.warning(f"Failed to augment {img_path}: {e}")

        logging.info(f"Generated {total_augmented} augmented images")
        return total_augmented


def create_example_dataset(
    output_dir: Union[str, Path], num_classes: int = 10, images_per_class: int = 50
):
    """
    Create an example dataset with synthetic images for testing

    Args:
        output_dir: Output directory for the dataset
        num_classes: Number of classes to create
        images_per_class: Number of images per class
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    # Example bird and animal class names
    class_names = [
        "bald_eagle",
        "red_cardinal",
        "blue_jay",
        "robin",
        "sparrow",
        "lion",
        "tiger",
        "elephant",
        "giraffe",
        "zebra",
        "golden_retriever",
        "siamese_cat",
        "brown_bear",
        "gray_wolf",
        "red_fox",
    ][:num_classes]

    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (128, 128, 128),  # Gray
        (0, 128, 0),  # Dark Green
    ]

    for i, class_name in enumerate(class_names):
        class_dir = output_dir / class_name
        ensure_dir(class_dir)

        base_color = colors[i % len(colors)]

        for j in range(images_per_class):
            # Create a simple synthetic image
            image = Image.new("RGB", (224, 224), color=base_color)

            # Add some noise for variation
            pixels = np.array(image)
            noise = np.random.randint(-30, 30, pixels.shape, dtype=np.int16)
            pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixels)

            # Save image
            image_path = class_dir / f"{class_name}_{j:03d}.jpg"
            image.save(image_path, quality=95)

    print(f"Created example dataset with {num_classes} classes at: {output_dir}")
    return output_dir


def load_dataset_from_directory(
    dataset_dir: Union[str, Path], transform=None
) -> FewShotDataset:
    """Load dataset from directory structure"""
    dataset_dir = Path(dataset_dir)

    image_paths = []
    labels = []

    for class_dir in dataset_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        class_images = get_image_files(class_dir)

        for img_path in class_images:
            image_paths.append(str(img_path))
            labels.append(class_name)

    return FewShotDataset(image_paths, labels, transform)


def create_data_loaders(
    dataset: FewShotDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create data loader from dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
