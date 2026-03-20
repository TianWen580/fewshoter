"""
Attribute Text Generation System for CLIP Few-Shot Classification

This module implements the AttributeGenerator class that generates discriminative
text probes based on visual attributes. It creates fine-grained textual descriptions
that help distinguish between similar classes in few-shot scenarios.

Key Features:
- Automatic attribute detection from class names and visual features
- Template-based text probe generation
- Multi-modal attribute validation using CLIP
- Hierarchical attribute organization (generic -> specific)
- Dynamic attribute weighting based on discriminative power
"""

import torch
import open_clip as clip
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
import json
from pathlib import Path
from collections import defaultdict, Counter

from ..core.config import Config
from ..core.utils import Timer


class AttributeGenerator:
    """
    Generator for discriminative text attributes and probes

    Creates textual descriptions that capture fine-grained differences
    between classes, improving few-shot classification performance.
    """

    def __init__(self, clip_model, device: str = "cuda", max_probes_per_class: int = 5):
        """
        Initialize attribute generator

        Args:
            clip_model: CLIP model instance for text encoding
            device: Device for computations
            max_probes_per_class: Maximum number of text probes per class
        """
        self.model = clip_model
        self.device = torch.device(device)
        self.max_probes_per_class = max_probes_per_class

        # Load attribute vocabularies
        self._load_attribute_vocabularies()

        # Text templates for probe generation
        self.text_templates = [
            "a photo of a {}",
            "a picture of a {}",
            "{} in natural habitat",
            "close-up of a {}",
            "{} wildlife photography",
            "a {} with {}",
            "{} showing {}",
            "{} featuring {}",
            "detailed view of {} with {}",
            "{} displaying {}",
        ]

        # Cache for generated probes
        self.probe_cache = {}
        self.attribute_cache = {}

        # Per-class text calibration params
        self.text_calibration = {}  # {class_name: {"temp": float, "bias": float}}

        self.logger = logging.getLogger(__name__)

    def _load_attribute_vocabularies(self):
        """Load predefined attribute vocabularies for different categories"""
        # Bird attributes
        self.bird_attributes = {
            "colors": [
                "red",
                "blue",
                "yellow",
                "green",
                "black",
                "white",
                "brown",
                "gray",
                "orange",
            ],
            "body_parts": [
                "beak",
                "wing",
                "tail",
                "head",
                "neck",
                "breast",
                "back",
                "feet",
                "crest",
            ],
            "patterns": [
                "striped",
                "spotted",
                "solid",
                "mottled",
                "barred",
                "streaked",
            ],
            "sizes": ["small", "medium", "large", "tiny", "huge"],
            "behaviors": ["flying", "perching", "feeding", "singing", "nesting"],
            "habitats": [
                "forest",
                "water",
                "grassland",
                "urban",
                "mountain",
                "coastal",
            ],
        }

        # Animal attributes
        self.animal_attributes = {
            "colors": [
                "brown",
                "black",
                "white",
                "gray",
                "golden",
                "tan",
                "spotted",
                "striped",
            ],
            "body_parts": [
                "ears",
                "tail",
                "paws",
                "mane",
                "horns",
                "antlers",
                "snout",
                "whiskers",
            ],
            "patterns": ["striped", "spotted", "solid", "patched", "mottled"],
            "sizes": ["small", "medium", "large", "massive", "tiny"],
            "behaviors": ["running", "hunting", "grazing", "climbing", "swimming"],
            "habitats": ["forest", "savanna", "mountain", "desert", "arctic", "jungle"],
        }

        # Generic attributes
        self.generic_attributes = {
            "textures": ["smooth", "rough", "fluffy", "sleek", "coarse", "fine"],
            "shapes": ["round", "elongated", "compact", "slender", "robust"],
            "expressions": ["alert", "calm", "aggressive", "curious", "peaceful"],
        }

    def generate_class_attributes(
        self, class_name: str, support_features: Optional[List[torch.Tensor]] = None
    ) -> List[str]:
        """
        Generate attributes for a specific class

        Args:
            class_name: Name of the class
            support_features: Optional visual features from support examples

        Returns:
            List of relevant attributes for the class
        """
        if class_name in self.attribute_cache:
            return self.attribute_cache[class_name]

        attributes = set()

        # Extract attributes from class name
        name_attributes = self._extract_name_attributes(class_name)
        attributes.update(name_attributes)

        # Add category-specific attributes
        category_attributes = self._get_category_attributes(class_name)
        attributes.update(category_attributes)

        # Validate attributes using CLIP if support features available
        if support_features:
            validated_attributes = self._validate_attributes_with_clip(
                class_name, list(attributes), support_features
            )
            attributes = set(validated_attributes)

        # Convert to sorted list
        final_attributes = sorted(list(attributes))

        # Cache results
        self.attribute_cache[class_name] = final_attributes

        return final_attributes

    def _extract_name_attributes(self, class_name: str) -> List[str]:
        """Extract attributes from class name"""
        attributes = []
        name_lower = class_name.lower()

        # Color attributes
        for color in [
            "red",
            "blue",
            "yellow",
            "green",
            "black",
            "white",
            "brown",
            "gray",
            "orange",
            "golden",
        ]:
            if color in name_lower:
                attributes.append(f"{color} colored")

        # Size attributes
        size_words = {
            "small": "small",
            "large": "large",
            "giant": "large",
            "tiny": "small",
            "big": "large",
        }
        for size_word, size_attr in size_words.items():
            if size_word in name_lower:
                attributes.append(size_attr)

        # Pattern attributes
        pattern_words = {"striped": "striped", "spotted": "spotted", "barred": "barred"}
        for pattern_word, pattern_attr in pattern_words.items():
            if pattern_word in name_lower:
                attributes.append(pattern_attr)

        # Habitat attributes
        habitat_words = {
            "forest": "forest dwelling",
            "water": "aquatic",
            "sea": "marine",
            "mountain": "mountain dwelling",
            "desert": "desert dwelling",
        }
        for habitat_word, habitat_attr in habitat_words.items():
            if habitat_word in name_lower:
                attributes.append(habitat_attr)

        return attributes

    def _get_category_attributes(self, class_name: str) -> List[str]:
        """Get category-specific attributes"""
        name_lower = class_name.lower()
        attributes = []

        # Determine category
        bird_keywords = [
            "bird",
            "eagle",
            "hawk",
            "sparrow",
            "robin",
            "owl",
            "duck",
            "goose",
            "swan",
        ]
        animal_keywords = [
            "cat",
            "dog",
            "lion",
            "tiger",
            "bear",
            "wolf",
            "deer",
            "elephant",
            "horse",
        ]

        is_bird = any(keyword in name_lower for keyword in bird_keywords)
        is_animal = any(keyword in name_lower for keyword in animal_keywords)

        if is_bird:
            # Add bird-specific attributes
            attributes.extend(["feathered", "winged", "beaked"])
            # Add some random bird attributes for diversity
            for category, attrs in self.bird_attributes.items():
                if category in ["body_parts", "behaviors"]:
                    attributes.extend(attrs[:2])  # Take first 2 from each category

        elif is_animal:
            # Add animal-specific attributes
            attributes.extend(["furry", "four-legged", "mammalian"])
            # Add some random animal attributes
            for category, attrs in self.animal_attributes.items():
                if category in ["body_parts", "behaviors"]:
                    attributes.extend(attrs[:2])

        # Add generic attributes
        attributes.extend(["natural", "wildlife", "outdoor"])

        return attributes

    def _validate_attributes_with_clip(
        self,
        class_name: str,
        attributes: List[str],
        support_features: List[torch.Tensor],
    ) -> List[str]:
        """Validate attributes using CLIP similarity"""
        if not support_features:
            return attributes

        validated_attributes = []

        # Compute average support feature
        avg_support_feature = torch.stack(support_features).mean(dim=0)
        avg_support_feature = avg_support_feature / avg_support_feature.norm()

        # Test each attribute
        for attr in attributes:
            # Create text with attribute
            text_with_attr = f"a {class_name} with {attr}"
            text_without_attr = f"a {class_name}"

            # Encode texts
            text_tokens_with = clip.tokenize([text_with_attr]).to(self.device)
            text_tokens_without = clip.tokenize([text_without_attr]).to(self.device)

            with torch.no_grad():
                text_feat_with = self.model.encode_text(text_tokens_with)
                text_feat_without = self.model.encode_text(text_tokens_without)

                text_feat_with = text_feat_with / text_feat_with.norm()
                text_feat_without = text_feat_without / text_feat_without.norm()

            # Compute similarities
            sim_with = torch.cosine_similarity(
                avg_support_feature, text_feat_with, dim=-1
            ).item()
            sim_without = torch.cosine_similarity(
                avg_support_feature, text_feat_without, dim=-1
            ).item()

            # Keep attribute if it improves similarity
            if sim_with > sim_without + 0.01:  # Small threshold to avoid noise
                validated_attributes.append(attr)

        return validated_attributes

    def generate_text_probes(
        self,
        class_name: str,
        attributes: Optional[List[str]] = None,
        support_features: Optional[List[torch.Tensor]] = None,
    ) -> List[str]:
        """
        Generate discriminative text probes for a class

        Args:
            class_name: Name of the class
            attributes: Optional list of attributes (will be generated if not provided)
            support_features: Optional visual features for validation

        Returns:
            List of text probes for the class
        """
        cache_key = f"{class_name}_{len(support_features) if support_features else 0}"
        if cache_key in self.probe_cache:
            return self.probe_cache[cache_key]

        # Generate attributes if not provided
        if attributes is None:
            attributes = self.generate_class_attributes(class_name, support_features)

        probes = []

        # Basic probes without attributes
        basic_templates = [
            f"a photo of a {class_name}",
            f"a picture of a {class_name}",
            f"{class_name} in natural habitat",
            f"close-up of a {class_name}",
        ]
        probes.extend(basic_templates)

        # Attribute-based probes
        for attr in attributes[: self.max_probes_per_class]:
            attribute_probes = [
                f"a {class_name} with {attr}",
                f"{class_name} showing {attr}",
                f"a photo of {attr} {class_name}",
            ]
            probes.extend(attribute_probes)

        # Combination probes (if we have multiple attributes)
        if len(attributes) >= 2:
            for i in range(min(2, len(attributes) - 1)):
                for j in range(i + 1, min(i + 3, len(attributes))):
                    combo_probe = (
                        f"a {class_name} with {attributes[i]} and {attributes[j]}"
                    )
                    probes.append(combo_probe)

        # Limit total number of probes
        if len(probes) > self.max_probes_per_class * 3:
            # Rank probes by expected discriminative power
            ranked_probes = self._rank_probes(probes, class_name, support_features)
            probes = ranked_probes[: self.max_probes_per_class * 3]

        # Cache results
        self.probe_cache[cache_key] = probes

        return probes

    def _rank_probes(
        self,
        probes: List[str],
        class_name: str,
        support_features: Optional[List[torch.Tensor]] = None,
    ) -> List[str]:
        """Rank probes by discriminative power"""
        if not support_features:
            return probes  # Return as-is if no features for ranking

        # Compute average support feature
        avg_support_feature = torch.stack(support_features).mean(dim=0)
        avg_support_feature = avg_support_feature / avg_support_feature.norm()

        probe_scores = []

        # Score each probe
        for probe in probes:
            text_tokens = clip.tokenize([probe]).to(self.device)

            with torch.no_grad():
                text_feature = self.model.encode_text(text_tokens)
                text_feature = text_feature / text_feature.norm()

            # Compute similarity with support features
            similarity = torch.cosine_similarity(
                avg_support_feature, text_feature, dim=-1
            ).item()
            probe_scores.append((probe, similarity))

        # Sort by similarity score (descending)
        probe_scores.sort(key=lambda x: x[1], reverse=True)

        return [probe for probe, score in probe_scores]

    def generate_comparative_probes(
        self,
        class_names: List[str],
        support_features_dict: Optional[Dict[str, List[torch.Tensor]]] = None,
    ) -> Dict[str, List[str]]:
        """
        Generate comparative probes that emphasize differences between classes

        Args:
            class_names: List of all class names
            support_features_dict: Optional dict of support features per class

        Returns:
            Dictionary mapping class names to comparative probes
        """
        comparative_probes = {}

        # Generate attributes for all classes
        all_attributes = {}
        for class_name in class_names:
            support_feats = (
                support_features_dict.get(class_name) if support_features_dict else None
            )
            all_attributes[class_name] = self.generate_class_attributes(
                class_name, support_feats
            )

        # Find discriminative attributes for each class
        for class_name in class_names:
            class_attrs = set(all_attributes[class_name])

            # Find attributes unique to this class or rare in others
            discriminative_attrs = []
            for attr in class_attrs:
                # Count how many other classes have this attribute
                other_count = sum(
                    1
                    for other_class in class_names
                    if other_class != class_name and attr in all_attributes[other_class]
                )

                # Prefer attributes that are unique or rare
                if (
                    other_count <= len(class_names) * 0.3
                ):  # Present in <= 30% of other classes
                    discriminative_attrs.append(attr)

            # Generate probes emphasizing discriminative attributes
            probes = []
            for attr in discriminative_attrs[:3]:  # Top 3 discriminative attributes
                probes.extend(
                    [
                        f"a {class_name} with distinctive {attr}",
                        f"{class_name} characterized by {attr}",
                        f"unique {class_name} showing {attr}",
                    ]
                )

    # ----- Calibration utilities -----
    def fit_text_calibration(
        self,
        class_name: str,
        support_image_features: Optional[List[torch.Tensor]],
        default_temp: float = 1.0,
        default_bias: float = 0.0,
    ) -> None:
        """Estimate per-class text calibration params (temperature, bias).
        We use support image global features to measure text similarity distribution
        for this class and set a small bias/temperature to mitigate prior bias.
        This is a lightweight heuristic and stable with few shots.
        """
        try:
            if not support_image_features:
                self.text_calibration[class_name] = {
                    "temp": default_temp,
                    "bias": default_bias,
                }
                return
            # Compute probe features for basic prompts (without attributes) to avoid overfitting
            probes = [f"a photo of a {class_name}", f"a picture of a {class_name}"]
            text_feats = self.encode_text_probes(probes)
            text_feat = text_feats.mean(dim=0)  # [D]
            text_feat = text_feat / text_feat.norm()
            # Aggregate support image features
            S = torch.stack(
                [f.squeeze(0) if f.dim() == 2 else f for f in support_image_features]
            ).to(self.device)
            S = S.float()
            S = S / S.norm(dim=-1, keepdim=True)
            sims = torch.matmul(S, text_feat)
            mu = sims.mean().item()
            std = sims.std(unbiased=False).item() + 1e-6
            # Heuristic: temp scales dispersion to near 1 std; bias recenters mean to ~0
            temp = 1.0 / std
            bias = -mu
            # Clamp to safe ranges
            temp = float(max(0.5, min(temp, 5.0)))
            bias = float(max(-0.5, min(bias, 0.5)))
            self.text_calibration[class_name] = {"temp": temp, "bias": bias}
        except Exception:
            self.text_calibration[class_name] = {
                "temp": default_temp,
                "bias": default_bias,
            }

    def apply_text_calibration(self, class_name: str, raw_similarity: float) -> float:
        """Apply per-class temperature and bias to a text similarity/logit."""
        calib = self.text_calibration.get(class_name)
        if not calib:
            return raw_similarity
        temp = calib.get("temp", 1.0)
        bias = calib.get("bias", 0.0)
        return float((raw_similarity + bias) * temp)

    def encode_text_probes(self, probes: List[str]) -> torch.Tensor:
        """Encode text probes using CLIP/OpenCLIP text encoder"""
        if not probes:
            # Determine embedding dimension robustly
            embed_dim = getattr(self.model, "embed_dim", None)
            if embed_dim is None:
                tp = getattr(self.model, "text_projection", None)
                if tp is not None and hasattr(tp, "shape") and len(tp.shape) == 2:
                    embed_dim = tp.shape[1]
            if embed_dim is None:
                embed_dim = 1024
            return torch.empty(0, embed_dim, device=self.device)

        # Tokenize all probes
        text_tokens = clip.tokenize(probes).to(self.device)

        # Encode
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def clear_cache(self):
        """Clear all caches"""
        self.probe_cache.clear()
        self.attribute_cache.clear()
        self.logger.info("Attribute generator caches cleared")

    def save_attributes(self, output_path: str):
        """Save generated attributes to file"""
        data = {
            "attribute_cache": self.attribute_cache,
            "probe_cache": self.probe_cache,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Attributes saved to: {output_path}")

    def load_attributes(self, input_path: str):
        """Load attributes from file"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.attribute_cache = data.get("attribute_cache", {})
        self.probe_cache = data.get("probe_cache", {})

        self.logger.info(f"Attributes loaded from: {input_path}")


def create_attribute_generator(clip_model, config: Config) -> AttributeGenerator:
    """Factory function to create attribute generator"""
    return AttributeGenerator(
        clip_model=clip_model,
        device=config.model.device,
        max_probes_per_class=config.classification.max_text_probes,
    )
