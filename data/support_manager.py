"""
Support Set Management System for CLIP Few-Shot Classification

This module implements the SupportSetManager class that handles loading,
processing, and managing few-shot support examples. It generates visual prototypes,
attention consensus maps, and attribute dictionaries for each class.

Key Features:
- Automatic support set loading from directory structure
- Visual prototype generation through feature averaging
- Attention consensus computation across support examples
- Attribute extraction and frequency analysis
- Prototype caching and serialization
- Support set quality validation
"""

import os
import json
import numpy as np
import torch
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from PIL import Image

import ast

from ..core.config import Config
from ..features.feature_extractor import MultiScaleFeatureExtractor
from ..core.utils import get_image_files, ensure_dir, Timer


class SupportSetManager:
    """
    Manager for few-shot support sets

    Handles loading support examples, computing prototypes, and managing
    class-specific information for few-shot classification.
    """

    def __init__(
        self,
        support_dir: Optional[Union[str, Path]] = None,
        feature_extractor: Optional[MultiScaleFeatureExtractor] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize support set manager

        Args:
            support_dir: Directory containing support set organized by class
            feature_extractor: Feature extractor instance
            config: Configuration object
        """
        self.support_dir = Path(support_dir) if support_dir else None
        self.feature_extractor = feature_extractor
        self.config = config or Config()

        # Data structures
        self.class_names = []
        self.support_images = {}  # {class_name: [image_paths]}
        self.prototypes = {}  # {class_name: {scale: prototype}}
        self.attention_consensus = {}  # {class_name: consensus_map}
        self.attribute_dict = {}  # {class_name: [attributes]}
        self.class_statistics = {}  # {class_name: stats}
        # Adapter support bank: tensors per class for Tip-Adapter-like scoring
        self.adapter_support_bank = {}  # {class_name: Tensor[num_shots, feat_dim]}
        # Calibration models
        self.text_calibration = {}
        self.visual_calibration = {}  # visual-side per-class temp/bias
        self.class_logit_bias = {}
        # Optional trained small head (cos/arcface/cosface)
        self.small_head = None  # {'type': str, 'W': np.ndarray[num_classes, D], 'scale': float, 'margin': float}

        # NSS structures
        self.nss_neighbors = {}  # {class_name: [neighbor_class_names]}
        self.nss_bases = {}  # {class_name: np.ndarray of shape [D, r]}

        # Stage 1: Discriminative mining outputs
        self.discriminative_masks = {}  # {class_name: {layer: np.ndarray[H, W]}}
        self.subcenter_prototypes = {}  # {class_name: {layer: np.ndarray[K, C]}}

        # Temp holders
        self._tmp_support_globals = {}
        # Margin accumulator for incremental discriminative mining (memory-efficient)
        self._margin_accumulator = {}  # {class_name: np.ndarray[H, W]}
        self._margin_counter = {}  # {class_name: int}

        self.logger = logging.getLogger(__name__)
        # Embedding cache per class for saving to embed.json
        # 每类的嵌入缓存（写入 embed.json）
        self.embed_records = {}  # {class_name: [{'image': str, 'global': List[float]}]}

        # Source of per-image embeddings: 'images' | 'embed' | 'prototypes' | None
        self.embedding_source: Optional[str] = None

        # Load support set if directory provided
        if self.support_dir and self.support_dir.exists():
            self.load_support_set()

        # If no classes discovered, try fallback from classes.py next to SVM/NN models
        if not self.class_names:
            try:
                cc = getattr(self.config, "classification", None)
                if cc:
                    candidates = []
                    svm_path = getattr(cc, "svm_model_path", None)
                    if svm_path:
                        candidates.append(Path(str(svm_path)).parent / "classes.py")
                    nn_path = getattr(cc, "nn_model_path", None)
                    if nn_path:
                        candidates.append(Path(str(nn_path)).parent / "classes.py")
                    for cp in candidates:
                        try:
                            if cp.exists():
                                src = cp.read_text(encoding="utf-8")
                                tree = ast.parse(src)
                                names = []
                                for node in tree.body:
                                    if isinstance(node, ast.Assign):
                                        for t in node.targets:
                                            if (
                                                isinstance(t, ast.Name)
                                                and t.id == "classes"
                                            ):
                                                if isinstance(
                                                    node.value, (ast.List, ast.Tuple)
                                                ):
                                                    for elt in node.value.elts:
                                                        if isinstance(
                                                            elt, ast.Constant
                                                        ) and isinstance(
                                                            elt.value, str
                                                        ):
                                                            names.append(elt.value)
                                if names:
                                    self.class_names = names
                                    break
                        except Exception:
                            continue
            except Exception:
                pass

    def load_support_set(self):
        """Load support set from directory structure"""
        if not self.support_dir or not self.support_dir.exists():
            raise ValueError("Support directory not found or not specified")

        self.logger.info(f"Loading support set from: {self.support_dir}")

        # Discover classes
        self.class_names = []
        self.support_images = {}

        for class_dir in self.support_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            image_files = get_image_files(class_dir)

            if len(image_files) < self.config.support_set.min_shots_per_class:
                self.logger.warning(
                    f"Class {class_name} has only {len(image_files)} images, "
                    f"minimum required: {self.config.support_set.min_shots_per_class}"
                )
                continue

            # Limit number of shots per class
            if len(image_files) > self.config.support_set.max_shots_per_class:
                image_files = image_files[: self.config.support_set.max_shots_per_class]
                self.logger.info(f"Limited {class_name} to {len(image_files)} shots")

            self.class_names.append(class_name)
            self.support_images[class_name] = image_files

        self.logger.info(
            f"Loaded {len(self.class_names)} classes with support examples"
        )

        # Process support set if feature extractor is available
        if self.feature_extractor:
            self.process_support_set()

    def process_support_set(self):
        """Process support set to generate prototypes and consensus maps"""
        # Reset embedding records for this processing run
        self.embed_records = {c: [] for c in self.class_names}
        self.embedding_source = "images"

        if not self.feature_extractor:
            raise ValueError("Feature extractor not available")

        self.logger.info("Processing support set...")

        for class_name in self.class_names:
            with Timer(f"Processing class {class_name}"):
                self._process_class(class_name)
        # After processing all classes, fit per-class text calibration
        try:
            from ..features.attribute_generator import AttributeGenerator

            if (
                hasattr(self, "feature_extractor")
                and self.feature_extractor is not None
            ):
                attr_gen = AttributeGenerator(
                    clip_model=self.feature_extractor.model,
                    device=str(self.config.model.device),
                    max_probes_per_class=self.config.classification.max_text_probes,
                )
                for cname, feats in self._tmp_support_globals.items():
                    attr_gen.fit_text_calibration(cname, feats)
                # Store fitted calibrations for later use inside classifier via support_manager
                self.text_calibration = attr_gen.text_calibration
        except Exception:
            self.text_calibration = {}
        # Estimate per-class logit bias from support set (lightweight)
        try:
            self.class_logit_bias = {}
            # We define bias as small positive prior for each class proportional to shots
            total = sum(len(v) for v in self.support_images.values()) + 1e-6
            for cname, imgs in self.support_images.items():
                freq = len(imgs) / total
                # Map frequency to small bias range [0, 0.2]
                self.class_logit_bias[cname] = float(
                    min(0.2, max(0.0, 0.2 * freq * len(self.class_names)))
                )
        except Exception:
            self.class_logit_bias = {}

        # Build discriminative masks and subcenters (Stage 1)

        try:
            self._compute_discriminative_structures()
        except Exception as e:
            self.logger.warning(f"Discriminative mining skipped: {e}")

        # Save prototypes if configured
        if self.config.support_set.save_prototypes:
            self.save_prototypes()
        # Save per-image embeddings cache (embed.json)
        if getattr(self.config.support_set, "save_embeddings", True):
            try:
                self.save_embeddings()
            except Exception as e:
                self.logger.warning(f"Failed to save embeddings cache: {e}")

        # After building prototypes, compute NSS neighbors and bases if enabled
        try:
            if getattr(self.config.classification, "enable_nss", False):
                self._build_nss_structures()
        except Exception as e:
            self.logger.warning(f"NSS structures build failed: {e}")

        self.logger.info("Support set processing completed")

    def _process_class(self, class_name: str):
        """Process a single class to generate prototypes"""
        image_paths = self.support_images[class_name]

        # Extract features from all support images
        class_features = []
        attention_maps = []
        attribute_counter = defaultdict(int)

        for img_path in image_paths:
            try:
                # Extract multi-scale features
                features = self.feature_extractor.extract_features(
                    img_path, return_attention=True, normalize=True
                )

                class_features.append(features)

                # Optional: training-time augmentation to enrich embeddings
                try:
                    if bool(
                        getattr(
                            self.config.support_set,
                            "enable_support_augmentation",
                            False,
                        )
                    ) and bool(
                        getattr(self.config.support_set, "support_aug_hflip", True)
                    ):
                        from PIL import Image

                        img = Image.open(img_path).convert("RGB")
                        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
                        aug_feats = self.feature_extractor.extract_features(
                            img_flip, return_attention=True, normalize=True
                        )
                        class_features.append(aug_feats)
                        # Store global for calibration later and embed cache
                        try:
                            if "global" in aug_feats:
                                g = aug_feats["global"]
                                if g.dim() == 2 and g.shape[0] == 1:
                                    g = g.squeeze(0)
                                self._tmp_support_globals.setdefault(
                                    class_name, []
                                ).append(g.cpu())
                                # raw unnormalized embedding
                                g0 = None
                                try:
                                    feats_raw = self.feature_extractor.extract_features(
                                        img_flip,
                                        return_attention=False,
                                        normalize=False,
                                    )
                                    g0 = feats_raw.get("global", None)
                                    if (
                                        isinstance(g0, torch.Tensor)
                                        and g0.dim() == 2
                                        and g0.shape[0] == 1
                                    ):
                                        g0 = g0.squeeze(0)
                                except Exception:
                                    g0 = g
                                # record
                                try:
                                    rel = str(img_path) + "|aug=hflip"
                                    if class_name not in self.embed_records:
                                        self.embed_records[class_name] = []
                                    self.embed_records[class_name].append(
                                        {
                                            "image": rel,
                                            "global": g0.detach()
                                            .cpu()
                                            .float()
                                            .numpy()
                                            .tolist(),
                                        }
                                    )
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass

                # Cache layer_9 spatial features for discriminative mining
                except Exception:
                    pass

                # Collect attention maps
                for key, value in features.items():
                    if "attn" in key and isinstance(value, torch.Tensor):
                        if len(attention_maps) <= len(class_features) - 1:
                            attention_maps.append({})
                        attention_maps[-1][key] = value.cpu().numpy()

                # Generate attributes (simplified version)
                attributes = self._extract_attributes(features, class_name)
                for attr in attributes:
                    attribute_counter[attr] += 1

                # Fit per-class text calibration incrementally (use global feature)
                try:
                    if "global" in features:
                        g = features["global"]
                        if g.dim() == 2 and g.shape[0] == 1:
                            g = g.squeeze(0)
                        # Store for calibration later
                        if class_name not in self._tmp_support_globals:
                            self._tmp_support_globals[class_name] = []
                        self._tmp_support_globals[class_name].append(g.cpu())
                        # Record per-image embedding for embed.json
                        try:
                            rel = str(img_path)
                            if self.support_dir is not None:
                                try:
                                    rel = str(
                                        Path(img_path)
                                        .resolve()
                                        .relative_to(self.support_dir.resolve())
                                    )
                                except Exception:
                                    rel = os.path.relpath(
                                        str(img_path), str(self.support_dir)
                                    )
                            # Extract RAW (unnormalized) global embedding for caching
                            g0 = None
                            if self.feature_extractor is not None:
                                feats_raw = self.feature_extractor.extract_features(
                                    img_path, return_attention=False, normalize=False
                                )
                                g0 = feats_raw.get("global", None)
                                if (
                                    isinstance(g0, torch.Tensor)
                                    and g0.dim() == 2
                                    and g0.shape[0] == 1
                                ):
                                    g0 = g0.squeeze(0)
                            if g0 is None:
                                # Fallback to current g (may be normalized) to avoid empty cache
                                g0 = g
                            rec = {
                                "image": rel,
                                "global": g0.detach().cpu().float().numpy().tolist(),
                            }
                            if class_name not in self.embed_records:
                                self.embed_records[class_name] = []
                            self.embed_records[class_name].append(rec)
                        except Exception:
                            pass

                except Exception:
                    pass

            except Exception as e:
                self.logger.error(f"Failed to process {img_path}: {e}")
                continue

        if not class_features:
            self.logger.error(f"No valid features extracted for class {class_name}")
            return

        # Compute prototypes for each feature scale
        prototypes = {}
        for scale_name in ["global", "layer_3", "layer_6", "layer_9"]:
            if scale_name in class_features[0]:
                scale_features = []
                for feat_dict in class_features:
                    if scale_name in feat_dict:
                        t = feat_dict[scale_name]
                        if isinstance(t, torch.Tensor):
                            t = t.detach().cpu()
                        scale_features.append(t)

                if scale_features:
                    # Stack and average features on CPU to avoid device mismatch (cpu vs cuda)
                    stacked_features = torch.stack(scale_features, dim=0)
                    prototype = stacked_features.mean(dim=0)
                    prototypes[scale_name] = prototype.cpu().numpy()

        self.prototypes[class_name] = prototypes

        # Build adapter support bank for this class (Tip-Adapter style)
        try:
            global_feats = []
            for feat_dict in class_features:
                if "global" in feat_dict and isinstance(
                    feat_dict["global"], torch.Tensor
                ):
                    g = feat_dict["global"]
                    # Ensure shape [D]
                    if g.dim() == 2 and g.shape[0] == 1:
                        g = g.squeeze(0)
                    global_feats.append(g.float().cpu())
            if len(global_feats) > 0:
                self.adapter_support_bank[class_name] = torch.stack(global_feats, dim=0)
        except Exception as e:
            self.logger.warning(
                f"Building adapter support bank failed for {class_name}: {e}"
            )

        # Compute attention consensus
        if attention_maps:
            consensus = self._compute_attention_consensus(attention_maps)
            self.attention_consensus[class_name] = consensus

        # Store frequent attributes
        total_images = len(class_features)
        frequent_attributes = [
            attr
            for attr, count in attribute_counter.items()
            if count >= total_images * self.config.support_set.attribute_threshold
        ]
        self.attribute_dict[class_name] = frequent_attributes

        # Compute class statistics
        self.class_statistics[class_name] = {
            "num_images": len(image_paths),
            "num_valid_features": len(class_features),
            "num_attributes": len(frequent_attributes),
            "prototype_scales": list(prototypes.keys()),
            "adapter_shots": int(self.adapter_support_bank.get(class_name).shape[0])
            if class_name in self.adapter_support_bank
            else 0,
        }

    def update_prototype_with_query(
        self, class_name: str, layer: str, query_feature: torch.Tensor, momentum: float
    ) -> None:
        """EMA update of the class prototype at given layer with query feature.
        query_feature shape: [C, H, W] or [D] for global.
        """
        if (
            class_name not in self.prototypes
            or layer not in self.prototypes[class_name]
        ):
            return
        proto = (
            torch.from_numpy(self.prototypes[class_name][layer])
            .to(query_feature.device)
            .float()
        )
        qf = query_feature
        if qf.dim() == 4 and qf.shape[0] == 1:
            qf = qf.squeeze(0)
        # Match shapes
        if proto.dim() != qf.dim():
            return
        updated = (1 - momentum) * proto + momentum * qf
        self.prototypes[class_name][layer] = updated.detach().cpu().numpy()

    def _extract_attributes(
        self, features: Dict[str, torch.Tensor], class_name: str
    ) -> List[str]:
        """
        Extract visual attributes from features (simplified implementation)

        In a full implementation, this would use more sophisticated attribute detection
        """
        attributes = []

        # Basic attribute templates based on class type
        if any(
            bird_word in class_name.lower()
            for bird_word in ["bird", "eagle", "hawk", "sparrow", "robin"]
        ):
            attributes.extend(["feathered", "winged", "beaked"])
        elif any(
            animal_word in class_name.lower()
            for animal_word in ["cat", "dog", "lion", "tiger", "bear"]
        ):
            attributes.extend(["furry", "four_legged", "mammal"])

        # Add generic attributes
        attributes.extend(["natural", "wildlife"])

        return attributes

    def _compute_attention_consensus(
        self, attention_maps: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Compute consensus attention maps across support examples"""
        consensus = {}

        # Get all attention map keys
        all_keys = set()
        for attn_map in attention_maps:
            all_keys.update(attn_map.keys())

        # Compute consensus for each attention type
        for key in all_keys:
            maps = []
            for attn_map in attention_maps:
                if key in attn_map:
                    maps.append(attn_map[key])

            if maps:
                # Stack and average
                stacked_maps = np.stack(maps, axis=0)
                consensus[key] = np.mean(stacked_maps, axis=0)

        return consensus

    def _build_nss_structures(self):
        """Build Neighborhood Subspace Suppression structures.
        For each class, find top-K nearest neighbor classes by cosine similarity
        of global prototypes, then compute an orthonormal basis spanning their
        prototypes (neighbor subspace).
        """
        import numpy as np

        # Collect class order and global prototypes
        names = [c for c in self.class_names if "global" in self.prototypes.get(c, {})]
        if len(names) < 2:
            return
        protos = []
        for c in names:
            p = self.prototypes[c]["global"]
            if p.ndim > 1:
                # flatten if spatial; but global should be 1D
                p = p.reshape(-1)
            protos.append(p)
        P = np.stack(protos, axis=0)  # [N, D]
        # normalize
        P_norm = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
        # cosine sim matrix
        S = P_norm @ P_norm.T  # [N, N]
        K = int(getattr(self.config.classification, "nss_num_neighbors", 3))
        self.nss_neighbors = {}
        self.nss_bases = {}
        for i, ci in enumerate(names):
            # get neighbor indices excluding self
            sims = S[i].copy()
            sims[i] = -1.0
            nn_idx = np.argsort(-sims)[: max(1, K)]
            neigh_names = [names[j] for j in nn_idx.tolist()]
            self.nss_neighbors[ci] = neigh_names
            # build basis using SVD on neighbor prototypes
            neigh_vecs = P_norm[nn_idx]  # [K, D]
            # Compute covariance in D-space via neigh_vecs^T * neigh_vecs
            C = neigh_vecs.T @ neigh_vecs  # [D, D]
            try:
                U, s, _ = np.linalg.svd(C, full_matrices=False)
            except np.linalg.LinAlgError:
                # fallback to QR on stacked neighbor vectors
                Q, _ = np.linalg.qr(neigh_vecs.T)
                U = Q
            # choose rank r = min(K, D)
            r = min(neigh_vecs.shape[0], U.shape[1])
            basis = U[:, :r]  # [D, r]
            self.nss_bases[ci] = basis.astype(np.float32)

    def _compute_discriminative_structures(self):
        """Build per-class discriminative masks and subcenter prototypes (Stage 1).
        Uses layer_9 spatial features and global prototypes contrastively.
        Implements incremental margin computation to reduce memory footprint.
        """
        try:
            if not getattr(
                self.config.classification, "enable_discriminative_mining", False
            ):
                return
            import numpy as np

            self.discriminative_masks = {}
            self.subcenter_prototypes = {}
            # Precompute normalized global prototypes for all classes
            all_names = [
                c for c in self.class_names if "global" in self.prototypes.get(c, {})
            ]
            if len(all_names) < 2:
                return
            proto_mat = {}
            for c in all_names:
                p = self.prototypes[c]["global"]
                p = p.reshape(-1).astype(np.float32)
                n = np.linalg.norm(p) + 1e-8
                proto_mat[c] = p / n
            topk_ratio = float(
                getattr(
                    self.config.classification, "discriminative_token_topk_ratio", 0.15
                )
            )
            num_sub = int(getattr(self.config.classification, "num_subcenters", 3))

            # Initialize margin accumulators for memory-efficient processing
            self._margin_accumulator = {}
            self._margin_counter = {}

            for c in all_names:
                # Process each class incrementally: extract features, compute margin, accumulate
                image_paths = self.support_images.get(c, [])
                if len(image_paths) == 0:
                    continue

                other_names = [x for x in all_names if x != c]
                p_self = proto_mat[c]

                # Accumulate margins incrementally per image
                margin_accumulator = None
                margin_count = 0
                subcenter_features = []  # For k-means subcenters

                for img_path in image_paths:
                    try:
                        # Extract layer_9 features directly from image
                        features = self.feature_extractor.extract_features(
                            img_path, return_attention=False, normalize=True
                        )
                        if "layer_9" not in features:
                            continue

                        f9 = features["layer_9"]
                        if isinstance(f9, torch.Tensor):
                            if f9.dim() == 4 and f9.shape[0] == 1:
                                f9 = f9.squeeze(0)
                            f9 = f9.cpu()

                        C, H, W = f9.shape
                        X = f9.view(C, -1).float().numpy()  # [C, HW]

                        # Store for subcenter computation (sampled patches)
                        x_patches = f9.view(C, -1).t().float().numpy()  # [HW, C]
                        if x_patches.shape[0] > 2000:
                            idx = np.random.choice(
                                x_patches.shape[0], 2000, replace=False
                            )
                            x_patches = x_patches[idx]
                        subcenter_features.append(x_patches)

                        # L2 normalize along feature dim
                        X /= np.linalg.norm(X, axis=0, keepdims=True) + 1e-8

                        # Compute similarity to self and others
                        s_self = (p_self @ X).reshape(H, W)
                        max_other = -1e9
                        for oc in other_names:
                            s = proto_mat[oc] @ X
                            if max_other == -1e9:
                                max_other = s
                            else:
                                max_other = np.maximum(max_other, s)
                        s_other = max_other.reshape(H, W)
                        margin = s_self - s_other

                        # Normalize to [0, 1]
                        m_min, m_max = float(margin.min()), float(margin.max())
                        if m_max > m_min:
                            margin = (margin - m_min) / (m_max - m_min)
                        else:
                            margin = np.zeros_like(margin)

                        # Accumulate margin (incremental computation)
                        if margin_accumulator is None:
                            margin_accumulator = margin
                        else:
                            margin_accumulator += margin
                        margin_count += 1

                        # Explicitly free memory
                        del f9, X, features

                    except Exception:
                        continue

                # Store accumulated results
                if margin_accumulator is not None and margin_count > 0:
                    self._margin_accumulator[c] = margin_accumulator
                    self._margin_counter[c] = margin_count

                    # Compute consensus saliency (average margin)
                    S = margin_accumulator / margin_count

                    # Top-k mask
                    flat = S.flatten()
                    k = max(1, int(round(topk_ratio * flat.size)))
                    thr = np.partition(flat, -k)[-k]
                    mask = (S >= thr).astype(np.float32)
                    self.discriminative_masks.setdefault(c, {})["layer_9"] = mask

                # Subcenters via k-means on collected patch features
                try:
                    if subcenter_features:
                        X_all = np.concatenate(
                            subcenter_features, axis=0
                        ).astype(np.float32)
                        # L2 normalize
                        X_all /= np.linalg.norm(X_all, axis=1, keepdims=True) + 1e-8
                        centers = self._kmeans_lite(X_all, num_sub)
                        self.subcenter_prototypes.setdefault(c, {})["layer_9"] = (
                            centers.astype(np.float32)
                        )
                except Exception:
                    pass
            # Optional: Global subcenters on global features
            try:
                if bool(
                    getattr(
                        self.config.classification, "enable_global_subcenters", False
                    )
                ):
                    import numpy as np

                    K = int(
                        getattr(self.config.classification, "global_subcenters_k", 3)
                    )
                    iters = int(
                        getattr(
                            self.config.classification, "global_subcenters_iters", 2
                        )
                    )
                    for c in all_names:
                        g_list = self._tmp_support_globals.get(c, [])
                        if len(g_list) == 0:
                            continue
                        X = np.stack(
                            [
                                g.view(-1).cpu().numpy().astype(np.float32)
                                if hasattr(g, "view")
                                else np.asarray(g, dtype=np.float32).reshape(-1)
                                for g in g_list
                            ],
                            axis=0,
                        )
                        # L2 normalize rows
                        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
                        centers = self._kmeans_lite(X, K, iters=iters)
                        self.subcenter_prototypes.setdefault(c, {})["global"] = (
                            centers.astype(np.float32)
                        )
            except Exception:
                pass

        except Exception as e:
            self.logger.warning(f"Discriminative structures build failed: {e}")

    def _kmeans_lite(self, X: np.ndarray, K: int, iters: int = 2) -> np.ndarray:
        """Very small k-means on L2-normalized X [N, C] with cosine distance.
        Returns centers [K, C]."""
        import numpy as np

        N = X.shape[0]
        K = max(1, min(K, N))
        # k-means++ style init
        centers = [X[np.random.randint(0, N)]]
        if K > 1:
            d2 = 1 - (X @ centers[0].reshape(-1, 1)).squeeze()  # cosine dist ~ 1 - cos
            for _ in range(1, K):
                idx = int(np.argmax(d2))
                centers.append(X[idx])
                sims = X @ centers[-1].reshape(-1, 1)
                d2 = np.minimum(d2, 1 - sims.squeeze())
        C_mat = np.stack(centers, axis=0)
        for _ in range(max(1, iters)):
            sims = X @ C_mat.T  # [N, K]
            assign = np.argmax(sims, axis=1)
            new_centers = []
            for k in range(K):
                sel = X[assign == k]
                if sel.shape[0] == 0:
                    new_centers.append(C_mat[k])
                else:
                    c = sel.mean(axis=0)
                    c /= np.linalg.norm(c) + 1e-8
                    new_centers.append(c)
            C_mat = np.stack(new_centers, axis=0)
        return C_mat

    def get_discriminative_mask(
        self, class_name: str, layer: str = "layer_9"
    ) -> Optional[np.ndarray]:
        return self.discriminative_masks.get(class_name, {}).get(layer, None)

    def get_subcenters(
        self, class_name: str, layer: str = "layer_9"
    ) -> Optional[np.ndarray]:
        return self.subcenter_prototypes.get(class_name, {}).get(layer, None)

    def get_prototype(
        self, class_name: str, scale: str = "global"
    ) -> Optional[np.ndarray]:
        """Get prototype for a specific class and scale"""
        if class_name in self.prototypes and scale in self.prototypes[class_name]:
            return self.prototypes[class_name][scale]
        return None

    def get_attention_consensus(
        self, class_name: str, attention_type: str = "attn_9"
    ) -> Optional[np.ndarray]:
        """Get attention consensus for a specific class"""
        if (
            class_name in self.attention_consensus
            and attention_type in self.attention_consensus[class_name]
        ):
            return self.attention_consensus[class_name][attention_type]
        return None

    def get_attributes(self, class_name: str) -> List[str]:
        """Get attributes for a specific class"""
        return self.attribute_dict.get(class_name, [])

    def validate_support_set(self) -> Dict[str, Any]:
        """Validate support set quality and return statistics"""
        validation_results = {
            "total_classes": len(self.class_names),
            "class_details": {},
            "warnings": [],
            "errors": [],
        }

        for class_name in self.class_names:
            class_info = {
                "num_images": len(self.support_images.get(class_name, [])),
                "has_prototypes": class_name in self.prototypes,
                "has_attention": class_name in self.attention_consensus,
                "num_attributes": len(self.attribute_dict.get(class_name, [])),
            }

            # Check for issues
            if class_info["num_images"] < self.config.support_set.min_shots_per_class:
                validation_results["warnings"].append(
                    f"Class {class_name} has insufficient images: {class_info['num_images']}"
                )

            if not class_info["has_prototypes"]:
                validation_results["errors"].append(
                    f"Class {class_name} missing prototypes"
                )

            if class_info["num_attributes"] == 0:
                validation_results["warnings"].append(
                    f"Class {class_name} has no attributes"
                )

            validation_results["class_details"][class_name] = class_info

        return validation_results

    def save_prototypes(self, output_path: Optional[Union[str, Path]] = None):
        """Save prototypes and metadata to file

        Args:
            output_path: Output file path. Defaults to support_dir/prototypes.json

        The saved content is controlled by config.support_set.save_full_cache:
        - False (default): Saves minimal inference-essential data only
        - True: Saves all debugging/visualization data (larger file size)

        Always saved: class_names, prototypes (global), config, text_calibration,
                      visual_calibration, small_head, subcenters
        Only when save_full_cache=True: attention_consensus, discriminative_masks,
                                        nss_bases, nss_neighbors, full adapter_support_bank
        """
        if output_path is None:
            output_path = (
                self.support_dir / "prototypes.json"
                if self.support_dir
                else "prototypes.json"
            )

        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        # Check if we should save full cache or minimal cache
        save_full = getattr(self.config.support_set, "save_full_cache", False)

        # Prepare data for serialization - minimal required for inference
        metadata = {
            "class_names": self.class_names,
            "prototypes": {},
            "config": self.config.to_dict(),
            "text_calibration": getattr(self, "text_calibration", {}),
            "visual_calibration": getattr(self, "visual_calibration", {}),
            "subcenters": {},
            "small_head": None,
        }

        # Only include these fields when save_full_cache is True
        if save_full:
            metadata["attention_consensus"] = {}
            metadata["attributes"] = self.attribute_dict
            metadata["statistics"] = self.class_statistics
            metadata["adapter_support_bank"] = {}
            metadata["nss_neighbors"] = getattr(self, "nss_neighbors", {})
            metadata["nss_bases"] = {}
            metadata["discriminative_masks"] = {}

        # Convert numpy arrays to lists - always save prototypes (required)
        for class_name in self.class_names:
            if class_name in self.prototypes:
                metadata["prototypes"][class_name] = {}
                for scale, prototype in self.prototypes[class_name].items():
                    metadata["prototypes"][class_name][scale] = prototype.tolist()

            # Save subcenter prototypes (useful for enhanced inference)
            if (
                getattr(self, "subcenter_prototypes", None)
                and class_name in self.subcenter_prototypes
            ):
                try:
                    for layer, centers in self.subcenter_prototypes[class_name].items():
                        metadata["subcenters"][class_name][layer] = centers.tolist()
                except Exception:
                    pass

            # Below fields only saved when save_full_cache is True
            if save_full:
                # Attention consensus (for visualization/debugging)
                if class_name in self.attention_consensus:
                    metadata["attention_consensus"][class_name] = {}
                    for attn_type, consensus in self.attention_consensus[
                        class_name
                    ].items():
                        metadata["attention_consensus"][class_name][attn_type] = (
                            consensus.tolist()
                        )

                # Discriminative masks (for visualization/debugging)
                if (
                    getattr(self, "discriminative_masks", None)
                    and class_name in self.discriminative_masks
                ):
                    try:
                        metadata["discriminative_masks"][class_name] = {}
                        for layer, mask in self.discriminative_masks[class_name].items():
                            metadata["discriminative_masks"][class_name][layer] = (
                                mask.tolist()
                            )
                    except Exception:
                        pass

                # Adapter support bank (only in embed.json when save_full_cache=False)
                if class_name in self.adapter_support_bank:
                    try:
                        metadata["adapter_support_bank"][class_name] = (
                            self.adapter_support_bank[class_name].tolist()
                        )
                    except Exception:
                        pass

                # NSS basis (for visualization/debugging)
                if getattr(self, "nss_bases", None) and class_name in self.nss_bases:
                    try:
                        metadata["nss_bases"][class_name] = self.nss_bases[
                            class_name
                        ].tolist()
                    except Exception:
                        pass

        # Save to file
        # Save small head if available (always saved when present)
        try:
            if getattr(self, "small_head", None) is not None:
                sh = self.small_head
                to_save = {
                    "type": sh.get("type"),
                    "scale": float(sh.get("scale", 1.0)),
                    "margin": float(sh.get("margin", 0.0)),
                    "W": sh.get("W").tolist()
                    if isinstance(sh.get("W"), np.ndarray)
                    else None,
                }
                metadata["small_head"] = to_save
        except Exception:
            pass

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Prototypes saved to: {output_path} (full_cache={save_full})")

    def save_embeddings(self, output_path: Optional[Union[str, Path]] = None):
        """
        Save per-image embeddings to embed.json for SVM/NN training reuse.
        Also include per-class text embeddings computed from optional descriptions or generated probes.

        When save_full_cache=False (default), adapter_support_bank is ONLY saved here in embed.json,
        not in prototypes.json, to avoid duplication and reduce cache file sizes.

        Format:
        {
          "class_names": [...],
          "embeddings": { class_name: [ {"image": rel_path, "global": [..] }, ... ] },
          "text_features": { class_name: [..] },
          "adapter_support_bank": { class_name: [[...], ...] }  # only when save_full_cache=False
        }
        """
        if output_path is None:
            fname = getattr(
                self.config.support_set, "embedding_cache_filename", "embed.json"
            )
            output_path = self.support_dir / fname if self.support_dir else fname
        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        # Check if we should save full cache or minimal cache
        save_full = getattr(self.config.support_set, "save_full_cache", False)

        # Build text features per class
        text_feat_map: Dict[str, List[float]] = {}
        try:
            from ..features.attribute_generator import AttributeGenerator

            # Try to load optional descriptions
            desc_path = getattr(self.config.classification, "desc_text_path", None)
            desc_dict = None
            if desc_path:
                try:
                    p = Path(str(desc_path))
                    if p.exists():
                        with open(p, "r", encoding="utf-8") as f:
                            desc_dict = json.load(f)
                except Exception:
                    desc_dict = None
            attr_gen = None
            if (
                hasattr(self, "feature_extractor")
                and self.feature_extractor is not None
            ):
                attr_gen = AttributeGenerator(
                    clip_model=self.feature_extractor.model,
                    device=str(self.config.model.device),
                    max_probes_per_class=self.config.classification.max_text_probes,
                )
            for cname in self.class_names:
                try:
                    if (
                        desc_dict
                        and cname in desc_dict
                        and isinstance(desc_dict[cname], str)
                        and desc_dict[cname].strip()
                    ):
                        probes = [desc_dict[cname].strip()]
                    else:
                        # Fallback to generated probes from attributes
                        attrs = self.get_attributes(cname)
                        if attr_gen is not None:
                            probes = attr_gen.generate_text_probes(cname, attrs)
                        else:
                            # Minimal fallback
                            probes = [f"a photo of a {cname}"]
                    if attr_gen is not None:
                        tf = attr_gen.encode_text_probes(probes)
                        tf = tf.mean(dim=0)
                        tf = tf / (tf.norm() + 1e-8)
                        text_feat_map[cname] = tf.detach().cpu().tolist()
                    else:
                        text_feat_map[cname] = []
                except Exception:
                    text_feat_map[cname] = []
        except Exception:
            # If anything fails, keep empty map
            text_feat_map = {c: [] for c in self.class_names}

        # Build main data structure
        data = {
            "class_names": self.class_names,
            "embeddings": {c: self.embed_records.get(c, []) for c in self.class_names},
            "text_features": text_feat_map,
        }

        # When save_full_cache=False, save adapter_support_bank ONLY in embed.json
        # (not in prototypes.json) to avoid duplication
        if not save_full:
            adapter_bank: Dict[str, Any] = {}
            for class_name in self.class_names:
                if class_name in self.adapter_support_bank:
                    try:
                        adapter_bank[class_name] = self.adapter_support_bank[
                            class_name
                        ].tolist()
                    except Exception:
                        pass
            data["adapter_support_bank"] = adapter_bank

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.info(
            f"Embeddings saved to: {output_path} (adapter_bank={'included' if not save_full else 'omitted, in prototypes.json'})"
        )

    def load_embeddings(self, embed_path: Union[str, Path]):
        """
        Load per-image embeddings from embed.json and rebuild support_images and adapter_support_bank.

        Supports two formats:
        1. New format (save_full_cache=False): adapter_support_bank stored directly in embed.json
        2. Legacy format: adapter_support_bank reconstructed from embeddings[].global
        """
        embed_path = Path(embed_path)
        if not embed_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embed_path}")
        with open(embed_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        emb = data.get("embeddings", {})
        # Preserve class_names if present; else keep existing
        self.class_names = data.get("class_names", self.class_names)
        self.support_images = {c: [] for c in self.class_names}
        self.adapter_support_bank = {}
        self.embedding_source = "embed"

        # Try to load adapter_support_bank directly first (new format)
        direct_bank = data.get("adapter_support_bank", {})
        for cname, arr in direct_bank.items():
            try:
                self.adapter_support_bank[cname] = torch.tensor(arr, dtype=torch.float32)
            except Exception:
                pass

        for cname, recs in emb.items():
            feats = []
            imgs = []
            for rec in recs:
                try:
                    g = np.asarray(rec.get("global", []), dtype=np.float32)
                    if g.size == 0:
                        continue
                    feats.append(g)
                    image_rel = rec.get("image", "")
                    if self.support_dir is not None:
                        imgs.append(str((self.support_dir / image_rel).resolve()))
                    else:
                        imgs.append(image_rel)
                except Exception:
                    continue
            if feats:
                # Only reconstruct from embeddings if not already loaded directly
                if cname not in self.adapter_support_bank:
                    try:
                        self.adapter_support_bank[cname] = torch.tensor(
                            np.stack(feats, axis=0)
                        )
                    except Exception:
                        pass
            self.support_images[cname] = imgs
        # Mirror embed content into embed_records for round-trip consistency
        self.embed_records = {c: emb.get(c, []) for c in self.class_names}

    @classmethod
    def from_prototypes(
        cls,
        prototypes_path: Union[str, Path],
        feature_extractor: Optional[MultiScaleFeatureExtractor] = None,
    ) -> "SupportSetManager":
        """Load support set manager from saved prototypes"""
        prototypes_path = Path(prototypes_path)

        if not prototypes_path.exists():
            raise FileNotFoundError(f"Prototypes file not found: {prototypes_path}")

        with open(prototypes_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Build config first so we can decide lazy-loading behaviors
        cfg = (
            Config.from_dict(metadata.get("config", {}))
            if isinstance(metadata.get("config", {}), dict)
            else Config()
        )

        # Create instance with config (do NOT pass support_dir here to avoid auto-loading images)
        manager = cls(feature_extractor=feature_extractor, config=cfg)

        # Set support_dir to prototypes directory
        manager.support_dir = prototypes_path.parent

        # Decide whether to load embed.json (can be very large). Default: skip for fast init.
        try:
            prefer_load = bool(
                getattr(manager.config.support_set, "load_embeddings_on_init", False)
            )
            env_flag = (
                os.environ.get("CLIPFEWSHOT_LOAD_EMBED_ON_INIT", "").strip().lower()
            )
            if env_flag in ("1", "true", "yes"):
                prefer_load = True
            elif env_flag in ("0", "false", "no"):
                prefer_load = False

            if prefer_load and manager.support_dir is not None:
                fname = getattr(
                    manager.config.support_set, "embedding_cache_filename", "embed.json"
                )
                embed_path = manager.support_dir / fname
                if embed_path.exists():
                    manager.load_embeddings(embed_path)
            else:
                try:
                    manager.logger.info(
                        "Fast init: skipped loading embed.json; will load on-demand if needed."
                    )
                except Exception:
                    pass
        except Exception as e:
            try:
                manager.logger.warning(f"Embeddings cache not loaded (fast init): {e}")
            except Exception:
                pass

        # Load core metadata from prototypes
        manager.class_names = metadata.get("class_names", manager.class_names)
        manager.attribute_dict = metadata.get("attributes", {})
        manager.class_statistics = metadata.get("statistics", {})
        # If embeddings were not loaded, mark source as 'prototypes' to avoid training from them
        if manager.embedding_source is None:
            manager.embedding_source = "prototypes"

        # Load adapter support bank - with backward compatibility
        # New behavior (save_full_cache=False): adapter_support_bank is in embed.json only
        # Old behavior (save_full_cache=True or legacy): adapter_support_bank is in prototypes.json
        manager.adapter_support_bank = {}
        try:
            if getattr(manager.config.adapter, "use_tip_adapter", True):
                # First try to load from prototypes.json (for save_full_cache=True or legacy)
                adapter_bank_meta = metadata.get("adapter_support_bank", {})
                if adapter_bank_meta:
                    for class_name, arr in adapter_bank_meta.items():
                        try:
                            manager.adapter_support_bank[class_name] = torch.tensor(arr)
                        except Exception:
                            pass
                # If not found in prototypes.json, try embed.json (new default behavior)
                if not manager.adapter_support_bank:
                    embed_path = manager.support_dir / "embed.json"
                    if embed_path.exists():
                        try:
                            with open(embed_path, "r", encoding="utf-8") as f:
                                embed_data = json.load(f)
                            embed_bank = embed_data.get("adapter_support_bank", {})
                            for class_name, arr in embed_bank.items():
                                try:
                                    manager.adapter_support_bank[class_name] = torch.tensor(arr, dtype=torch.float32)
                                except Exception:
                                    pass
                        except Exception:
                            pass
        except Exception:
            manager.adapter_support_bank = {}

        # Load discriminative masks and subcenters if enabled
        manager.discriminative_masks = {}
        manager.subcenter_prototypes = {}
        try:
            cc = getattr(manager.config, "classification", None)
            if cc and bool(getattr(cc, "enable_discriminative_mining", True)):
                for class_name, layers in metadata.get(
                    "discriminative_masks", {}
                ).items():
                    try:
                        manager.discriminative_masks[class_name] = {}
                        for layer, mask_list in layers.items():
                            manager.discriminative_masks[class_name][layer] = np.array(
                                mask_list, dtype=np.float32
                            )
                    except Exception:
                        pass
                for class_name, layers in metadata.get("subcenters", {}).items():
                    try:
                        manager.subcenter_prototypes[class_name] = {}
                        for layer, centers_list in layers.items():
                            manager.subcenter_prototypes[class_name][layer] = np.array(
                                centers_list, dtype=np.float32
                            )
                    except Exception:
                        pass
        except Exception:
            manager.discriminative_masks = {}
            manager.subcenter_prototypes = {}

        # Load text calibration if present
        manager.text_calibration = metadata.get("text_calibration", {})

        # Load visual calibration if present
        manager.visual_calibration = metadata.get("visual_calibration", {})

        # Load small head if present
        sh_meta = metadata.get("small_head")
        if isinstance(sh_meta, dict) and sh_meta.get("W") is not None:
            try:
                manager.small_head = {
                    "type": sh_meta.get("type", "cosine"),
                    "scale": float(sh_meta.get("scale", 1.0)),
                    "margin": float(sh_meta.get("margin", 0.0)),
                    "W": np.array(sh_meta.get("W"), dtype=np.float32),
                }
            except Exception:
                manager.small_head = None

        # Convert prototypes back to numpy arrays
        manager.prototypes = {}
        for class_name, class_prototypes in metadata["prototypes"].items():
            manager.prototypes[class_name] = {}
            for scale, prototype_list in class_prototypes.items():
                manager.prototypes[class_name][scale] = np.array(prototype_list)

        # Convert attention consensus back to numpy arrays (only if present - optional data)
        manager.attention_consensus = {}
        if "attention_consensus" in metadata:
            for class_name, class_attention in metadata["attention_consensus"].items():
                manager.attention_consensus[class_name] = {}
                for attn_type, consensus_list in class_attention.items():
                    manager.attention_consensus[class_name][attn_type] = np.array(
                        consensus_list
                    )

        # Load NSS neighbors and bases if NSS is enabled
        manager.nss_neighbors = {}
        manager.nss_bases = {}
        try:
            cc = getattr(manager.config, "classification", None)
            if cc and bool(getattr(cc, "enable_nss", False)):
                manager.nss_neighbors = metadata.get("nss_neighbors", {})
                nss_bases_meta = metadata.get("nss_bases", {})
                for class_name, basis_list in nss_bases_meta.items():
                    try:
                        manager.nss_bases[class_name] = np.array(
                            basis_list, dtype=np.float32
                        )
                    except Exception:
                        pass
        except Exception:
            manager.nss_neighbors = {}
            manager.nss_bases = {}

        # Fallback: if class_names empty, try to load from classes.py near SVM/NN models in config
        if not manager.class_names:
            try:
                cc = getattr(manager.config, "classification", None)
                if cc:
                    candidates = []
                    svm_path = getattr(cc, "svm_model_path", None)
                    if svm_path:
                        candidates.append(Path(str(svm_path)).parent / "classes.py")
                    nn_path = getattr(cc, "nn_model_path", None)
                    if nn_path:
                        candidates.append(Path(str(nn_path)).parent / "classes.py")
                    for cp in candidates:
                        try:
                            if cp.exists():
                                src = cp.read_text(encoding="utf-8")
                                tree = ast.parse(src)
                                names = []
                                for node in tree.body:
                                    if isinstance(node, ast.Assign):
                                        for t in node.targets:
                                            if (
                                                isinstance(t, ast.Name)
                                                and t.id == "classes"
                                            ):
                                                if isinstance(
                                                    node.value, (ast.List, ast.Tuple)
                                                ):
                                                    for elt in node.value.elts:
                                                        if isinstance(
                                                            elt, ast.Constant
                                                        ) and isinstance(
                                                            elt.value, str
                                                        ):
                                                            names.append(elt.value)
                                if names:
                                    manager.class_names = names
                                    break
                        except Exception:
                            continue
            except Exception:
                pass

        return manager

    def __len__(self):
        return len(self.class_names)

    def __repr__(self):
        return (
            f"SupportSetManager("
            f"classes={len(self.class_names)}, "
            f"support_dir={self.support_dir})"
        )


def create_support_manager(
    support_dir: Union[str, Path],
    feature_extractor: MultiScaleFeatureExtractor,
    config: Config,
) -> SupportSetManager:
    """Factory function to create support set manager"""
    return SupportSetManager(
        support_dir=support_dir, feature_extractor=feature_extractor, config=config
    )


def create_minimal_support_manager_from_models(
    feature_extractor: MultiScaleFeatureExtractor, config: Config
) -> Optional[SupportSetManager]:
    """
    Create a minimal SupportSetManager using only class names derived from SVM/NN models.
    This avoids any dependency on prototypes.json, embed.json, or support images.

    Returns:
        SupportSetManager if class names can be resolved; otherwise None.
    """
    try:
        cc = getattr(config, "classification", None)
        if not cc:
            return None

        class_names: Optional[List[str]] = None
        # Prefer reading classes.py next to model to avoid loading model files
        try:
            svm_model_path = getattr(cc, "svm_model_path", None)
            use_svm = bool(getattr(cc, "use_svm", False))
            if use_svm and svm_model_path:
                sp = Path(str(svm_model_path))
                if sp.exists():
                    try:
                        from ..engine.svm_classifier import (
                            _load_classes_from_py as _svm_load_classes,
                        )  # type: ignore

                        override = _svm_load_classes(sp.parent)
                    except Exception:
                        override = None
                    if override:
                        class_names = override
                    else:
                        try:
                            from ..engine.svm_classifier import SVMClassifier  # type: ignore

                            svm = SVMClassifier.load(sp)
                            class_names = getattr(svm, "class_names", None)
                            if (
                                not class_names
                                and getattr(svm, "label_encoder", None) is not None
                            ):
                                class_names = list(svm.label_encoder.classes_)  # type: ignore
                        except Exception:
                            pass
        except Exception:
            pass

        # If SVM path did not yield classes, try NN
        try:
            if class_names is None:
                nn_model_path = getattr(cc, "nn_model_path", None)
                use_nn = bool(getattr(cc, "use_nn", False))
                if use_nn and nn_model_path:
                    npth = Path(str(nn_model_path))
                    if npth.exists():
                        try:
                            from ..engine.nn_classifier import (
                                _load_classes_from_py as _nn_load_classes,
                            )  # type: ignore

                            override = _nn_load_classes(npth.parent)
                        except Exception:
                            override = None
                        if override:
                            class_names = override
                        else:
                            try:
                                from ..engine.nn_classifier import NNClassifier  # type: ignore

                                nnc = NNClassifier.load(npth)
                                class_names = getattr(nnc, "class_names", None)
                            except Exception:
                                pass
        except Exception:
            pass

        if class_names:
            sm = SupportSetManager(feature_extractor=feature_extractor, config=config)
            sm.class_names = list(class_names)
            try:
                sm.embedding_source = None
            except Exception:
                pass
            return sm
        return None
    except Exception:
        return None
