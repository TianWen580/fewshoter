"""
Fine-Grained Classification Engine for CLIP Few-Shot Learning

This module implements the FineGrainedClassifier class that combines all components
of the system to perform few-shot fine-grained bird and animal classification.
It integrates visual prototypes, text probes, and feature alignment for robust
zero-training classification.

Key Features:
- Multi-modal similarity computation (visual + textual)
- Attention-guided feature alignment
- Dynamic weight adjustment based on confidence
- Batch processing support
- Comprehensive result analysis and visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
from PIL import Image
import runpy

from ..core.config import Config
from ..features.feature_extractor import MultiScaleFeatureExtractor
from ..data.support_manager import SupportSetManager
from ..features.feature_aligner import FeatureAligner
from ..features.attribute_generator import AttributeGenerator
from ..core.utils import Timer, cosine_similarity, normalize_tensor, save_results
from .svm_classifier import SVMClassifier, SVMConfig
from .nn_classifier import NNClassifier, NNConfig
from ..features.dim_reducer import DimensionalityReducer, DimReduceConfig


class FineGrainedClassifier:
    """
    Main classification engine for fine-grained few-shot learning

    Combines visual prototypes from support examples with discriminative
    text probes to perform robust classification without training.
    """

    def __init__(self, support_manager: SupportSetManager, config: Optional[Config] = None):
        """
        Initialize the classifier

        Args:
            support_manager: Initialized support set manager
            config: Configuration object
        """
        self.support_manager = support_manager
        self.config = config or Config()

        # Initialize components
        self.feature_extractor = support_manager.feature_extractor
        if not self.feature_extractor:
            raise ValueError("Support manager must have a feature extractor")

        self.feature_aligner = FeatureAligner(
            device=self.config.model.device,
            alignment_strength=self.config.classification.alignment_strength,
            use_optical_flow=self.config.classification.use_feature_alignment,
        )

        self.attribute_generator = AttributeGenerator(
            clip_model=self.feature_extractor.model,
            device=self.config.model.device,
            max_probes_per_class=self.config.classification.max_text_probes,
        )

        # Initialize logger first
        self.logger = logging.getLogger(__name__)

        # Native reduction state (only used when not using SVM/NN)
        self.native_reducer: Optional[DimensionalityReducer] = None
        self.native_reduced_prototypes: Dict[str, torch.Tensor] = {}

        # Load optional descriptions dict for text override
        self._class_desc_dict = {}
        desc_path = getattr(self.config.classification, "desc_text_path", None)
        if desc_path:
            try:
                # Allow both absolute and relative paths
                desc_path = str(Path(desc_path))
                module_globals = runpy.run_path(desc_path)
                dd = module_globals.get("desc_dict")
                if isinstance(dd, dict):
                    self._class_desc_dict = dd
                    self.logger.info(
                        f"Loaded class descriptions from: {desc_path} ({len(dd)} entries)"
                    )
                else:
                    self.logger.warning(f"desc_dict not found or not a dict in {desc_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load class descriptions from {desc_path}: {e}")

        # Pre-compute text features for all classes unless pure SVM-only path
        self.text_features_cache = {}
        try:
            cc = self.config.classification
            svm_only = (
                bool(getattr(cc, "use_svm", False))
                and str(getattr(cc, "svm_infer_mode", "svm_only")) == "svm_only"
                and not bool(getattr(cc, "use_nn", False))
            )
        except Exception:
            svm_only = False
        if not svm_only:
            self._precompute_text_features()
        else:
            self.logger.info("Skipping text feature precomputation (SVM-only fast init)")

        # Optional: initialize SVM classifier trained on support-set embeddings
        self.svm: Optional[SVMClassifier] = None
        try:
            if getattr(self.config.classification, "use_svm", False):
                self._init_svm()
        except Exception as e:
            self.logger.warning(f"SVM initialization failed: {e}")

        # Optional: initialize Neural Network classifier trained on support-set embeddings
        self.nn: Optional[NNClassifier] = None
        try:
            if getattr(self.config.classification, "use_nn", False):
                self._init_nn()
        except Exception as e:
            self.logger.warning(f"NN initialization failed: {e}")

        # Optional: initialize dimensionality reducer for native prototype path
        try:
            cc = self.config.classification
            if (
                bool(getattr(cc, "enable_dim_reduction", False))
                and not bool(getattr(cc, "use_svm", False))
                and not bool(getattr(cc, "use_nn", False))
            ):
                self._init_native_reducer()
        except Exception as e:
            self.logger.warning(f"Native reducer init failed: {e}")

    def _precompute_text_features(self):
        """Pre-compute text features for all classes"""
        self.logger.info("Pre-computing text features for all classes...")

        for class_name in self.support_manager.class_names:
            # If a description is provided for this class, use it directly
            desc_text = None
            if self._class_desc_dict:
                # direct match first
                desc_text = self._class_desc_dict.get(class_name)
                # try simple variants if not found
                if desc_text is None:
                    desc_text = self._class_desc_dict.get(class_name.strip())
                if desc_text is None:
                    desc_text = self._class_desc_dict.get(str(class_name))

            if isinstance(desc_text, str) and desc_text.strip():
                self.logger.info(f"Using custom description for {class_name}")
                probes = [desc_text.strip()]
            else:
                # Otherwise, use generated probes from attributes and templates
                attributes = self.support_manager.get_attributes(class_name)
                probes = self.attribute_generator.generate_text_probes(class_name, attributes)

            # Encode probes
            text_features = self.attribute_generator.encode_text_probes(probes)

            # Store average text feature
            if len(text_features) > 0:
                avg_text_feature = text_features.mean(dim=0)
                self.text_features_cache[class_name] = normalize_tensor(avg_text_feature)
            else:
                # Fallback to basic class name
                basic_probe = f"a photo of a {class_name}"
                basic_features = self.attribute_generator.encode_text_probes([basic_probe])
                self.text_features_cache[class_name] = normalize_tensor(basic_features[0])

        self.logger.info(f"Text features computed for {len(self.text_features_cache)} classes")

    def _move_features_to_cpu(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensor features to CPU to reduce GPU memory usage."""
        cpu_features: Dict[str, Any] = {}
        for key, val in features.items():
            if isinstance(val, torch.Tensor):
                cpu_features[key] = val.detach().cpu()
            else:
                cpu_features[key] = val
        return cpu_features

    def _move_text_cache_to_cpu(self) -> None:
        """Move text features cache to CPU (once)."""
        if not hasattr(self, "_text_cache_on_cpu"):
            for class_name, feat in self.text_features_cache.items():
                if isinstance(feat, torch.Tensor):
                    self.text_features_cache[class_name] = feat.detach().cpu()
            self._text_cache_on_cpu = True

    def _build_tta_inputs(self, image_input: Union[str, Path, Image.Image, Any]) -> List[Any]:
        """Build a list of augmented inputs for TTA.
        Currently supports: original + optional horizontal flip.
        If input is not a path or PIL image, return as-is.
        """
        try:
            img: Optional[Image.Image] = None
            if isinstance(image_input, (str, Path)):
                p = Path(image_input)
                img = Image.open(p).convert("RGB")
            elif isinstance(image_input, Image.Image):
                img = image_input
            if img is None:
                return [image_input]
            inputs: List[Image.Image] = [img]
            if bool(getattr(self.config.classification, "tta_hflip", True)):
                inputs.append(img.transpose(Image.FLIP_LEFT_RIGHT))
            return inputs
        except Exception:
            return [image_input]

    def _aggregate_tta_features(self, feats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate a list of feature dicts from TTA views by simple averaging."""
        if not feats_list:
            return {}
        agg: Dict[str, Any] = {}
        keys = set().union(*[f.keys() for f in feats_list])
        for k in keys:
            vals = [f[k] for f in feats_list if k in f and f[k] is not None]
            if not vals:
                continue
            v0 = vals[0]
            if isinstance(v0, torch.Tensor):
                stacked = torch.stack([v if v.dim() > 0 else v.unsqueeze(0) for v in vals], dim=0)
                agg[k] = torch.mean(stacked, dim=0)
            else:
                # Fallback: keep the first value (e.g., metadata) or average if numpy
                try:
                    import numpy as _np

                    if isinstance(v0, _np.ndarray):
                        stacked = _np.stack(vals, axis=0)
                        agg[k] = stacked.mean(axis=0)
                    else:
                        agg[k] = v0
                except Exception:
                    agg[k] = v0
        return agg

    def classify(
        self,
        image_input: Union[str, Path, Image.Image, torch.Tensor],
        return_scores: bool = True,
        return_details: bool = False,
        allowed_classes: Optional[List[str]] = None,
    ) -> Union[str, Tuple[str, Dict]]:
        """
        Classify a single image

        Args:
            image_input: Input image (path, PIL Image, or tensor)
            return_scores: Whether to return confidence scores
            return_details: Whether to return detailed analysis

        Returns:
            Classification result and optional details
        """
        with Timer("Image classification"):
            # Extract query features (with optional TTA)
            if bool(getattr(self.config.classification, "tta_enable", False)):
                tta_inputs = self._build_tta_inputs(image_input)
                feats_list = [
                    self.feature_extractor.extract_features(
                        inp, return_attention=True, normalize=True
                    )
                    for inp in tta_inputs
                ]
                query_features = self._aggregate_tta_features(feats_list)
            else:
                query_features = self.feature_extractor.extract_features(
                    image_input, return_attention=True, normalize=True
                )

            # Move features to CPU if inference_device is set to cpu (avoids OOM with many classes)
            if getattr(self.config.classification, "inference_device", "same") == "cpu":
                query_features = self._move_features_to_cpu(query_features)
                # Also move text features cache to CPU if not already done
                self._move_text_cache_to_cpu()

            # Optional NN-only inference path (takes precedence if enabled)
            try:
                if (
                    getattr(self, "nn", None) is not None
                    and getattr(self.config.classification, "nn_infer_mode", "nn_only") == "nn_only"
                ):
                    pred_cls, nn_scores = self._nn_scores_for_features(
                        query_features, allowed_classes
                    )
                    if return_scores or return_details:
                        results = {
                            "predicted_class": pred_cls,
                            "confidence": float(nn_scores.get(pred_cls, 0.0)),
                            "all_scores": nn_scores,
                        }
                        return pred_cls, results
                    else:
                        return pred_cls
            except Exception as e:
                self.logger.debug(f"NN-only path failed, falling back: {e}")

            # Optional SVM-only inference path
            try:
                if (
                    self.svm is not None
                    and getattr(self.config.classification, "svm_infer_mode", "svm_only")
                    == "svm_only"
                ):
                    pred_cls, svm_scores = self._svm_scores_for_features(
                        query_features, allowed_classes
                    )
                    if return_scores or return_details:
                        results = {
                            "predicted_class": pred_cls,
                            "confidence": float(svm_scores.get(pred_cls, 0.0)),
                            "all_scores": svm_scores,
                        }
                        return pred_cls, results
                    else:
                        return pred_cls
            except Exception as e:
                self.logger.debug(f"SVM-only path failed, falling back: {e}")

            # Compute similarities for allowed classes (if provided)
            class_scores = {}
            detailed_results = {} if return_details else None

            class_iter = self.support_manager.class_names
            if allowed_classes is not None:
                try:
                    allowed_set = set(allowed_classes)
                    class_iter = [c for c in self.support_manager.class_names if c in allowed_set]
                except Exception:
                    # Fallback to all classes if filtering fails
                    class_iter = self.support_manager.class_names

            for class_name in class_iter:
                score, details = self._compute_class_similarity(
                    query_features, class_name, return_details
                )
                class_scores[class_name] = score

                if return_details:
                    detailed_results[class_name] = details

                # Periodic garbage collection to prevent memory accumulation
                if len(class_scores) % 10 == 0:
                    import gc
                    gc.collect()

            # Stage 4: Impostor-aware reranking (top-2)
            if (
                getattr(self.config.classification, "enable_impostor_rerank", False)
                and len(class_scores) >= 2
            ):
                try:
                    strength = float(
                        getattr(self.config.classification, "impostor_rerank_strength", 0.5)
                    )
                    # top-2 by current scores
                    top2 = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)[:2]
                    (c1, s1), (c2, s2) = top2[0], top2[1]
                    p1 = self._compute_part_score(query_features, c1)
                    p2 = self._compute_part_score(query_features, c2)
                    if p1 is not None and p2 is not None:
                        delta = strength * (p1 - p2)
                        class_scores[c1] = s1 + 0.5 * delta
                        class_scores[c2] = s2 - 0.5 * delta
                        if return_details:
                            detailed_results.setdefault(c1, {})["impostor_rerank_delta"] = (
                                0.5 * delta
                            )
                            detailed_results.setdefault(c2, {})["impostor_rerank_delta"] = (
                                -0.5 * delta
                            )
                except Exception as e:
                    self.logger.debug(f"Impostor rerank skipped: {e}")

            # Optional hybrid fusion with NN scores (blend NN and prototype scores)
            try:
                if (
                    getattr(self, "nn", None) is not None
                    and getattr(self.config.classification, "nn_infer_mode", "nn_only") == "hybrid"
                ):
                    _, nn_scores = self._nn_scores_for_features(query_features, allowed_classes)
                    # Min-max normalize current class_scores to [0,1]
                    if class_scores:
                        vals = list(class_scores.values())
                        vmin, vmax = min(vals), max(vals)
                        if vmax > vmin:
                            norm_scores = {
                                k: (v - vmin) / (vmax - vmin) for k, v in class_scores.items()
                            }
                        else:
                            norm_scores = {k: 0.0 for k in class_scores.keys()}
                    else:
                        norm_scores = {}
                    w_nn = float(getattr(self.config.classification, "nn_hybrid_weight", 0.5))
                    fused = {}
                    for c in class_scores.keys():
                        fused[c] = (1.0 - w_nn) * norm_scores.get(c, 0.0) + w_nn * float(
                            nn_scores.get(c, 0.0)
                        )
                    class_scores = fused
            except Exception as e:
                self.logger.debug(f"Hybrid NN fusion skipped: {e}")

            # Optional hybrid fusion with SVM scores (blend SVM and prototype scores)
            try:
                if (
                    self.svm is not None
                    and getattr(self.config.classification, "svm_infer_mode", "svm_only")
                    == "hybrid"
                ):
                    _, svm_scores = self._svm_scores_for_features(query_features, allowed_classes)
                    # Min-max normalize current class_scores to [0,1]
                    if class_scores:
                        vals = list(class_scores.values())
                        vmin, vmax = min(vals), max(vals)
                        if vmax > vmin:
                            norm_scores = {
                                k: (v - vmin) / (vmax - vmin) for k, v in class_scores.items()
                            }
                        else:
                            norm_scores = {k: 0.0 for k in class_scores.keys()}
                    else:
                        norm_scores = {}
                    w = float(getattr(self.config.classification, "svm_hybrid_weight", 0.5))
                    fused = {}
                    for c in class_scores.keys():
                        fused[c] = (1.0 - w) * norm_scores.get(c, 0.0) + w * float(
                            svm_scores.get(c, 0.0)
                        )
                    class_scores = fused
            except Exception as e:
                self.logger.debug(f"Hybrid SVM fusion skipped: {e}")

            # Optional per-query score normalization across classes (mitigates global shifts)
            try:
                norm_mode = str(getattr(self.config.classification, "score_normalization", "none"))
                if norm_mode in ("zscore", "mean_center") and len(class_scores) >= 2:
                    vals = list(class_scores.values())
                    import math

                    mu = sum(vals) / float(len(vals))
                    # population std (unbiased=False)
                    var = sum((v - mu) * (v - mu) for v in vals) / float(len(vals))
                    sd = math.sqrt(var) + 1e-8
                    if norm_mode == "zscore":
                        class_scores = {k: float((v - mu) / sd) for k, v in class_scores.items()}
                    else:  # 'mean_center'
                        class_scores = {k: float(v - mu) for k, v in class_scores.items()}
            except Exception as e:
                self.logger.debug(f"Score normalization skipped: {e}")

            if not class_scores:
                raise ValueError(
                    "Cannot classify: no class scores computed. "
                    "This usually means the support set has 0 classes. "
                    "Please check that your support directory contains valid class subdirectories with images."
                )

            best_class = max(class_scores, key=class_scores.get)
            best_score = class_scores[best_class]

            # Optional rejection by margin threshold
            rejection_info = None

            # Open-set/unknown rejection (optional): decide to label UNKNOWN based on strategy
            unknown_pred = False
            unknown_label = str(getattr(self.config.classification, "unknown_label", "UNKNOWN"))
            try:
                if bool(getattr(self.config.classification, "unknown_enable", False)):
                    strategy = str(
                        getattr(self.config.classification, "unknown_strategy", "zscore")
                    )
                    # Prepare common stats
                    vals = list(class_scores.values())
                    import math

                    mu = sum(vals) / float(len(vals)) if vals else 0.0
                    var = sum((v - mu) * (v - mu) for v in vals) / float(len(vals)) if vals else 0.0
                    sd = math.sqrt(var) + 1e-8
                    # Compute margin if possible
                    margin = None
                    if len(class_scores) >= 2:
                        sorted_pairs = sorted(
                            class_scores.items(), key=lambda x: x[1], reverse=True
                        )
                        second_score = sorted_pairs[1][1]
                        margin = float(best_score - second_score)
                    if strategy == "absolute":
                        thr = float(
                            getattr(
                                self.config.classification,
                                "unknown_abs_threshold",
                                0.25,
                            )
                        )
                        unknown_pred = best_score < thr
                    elif strategy == "margin":
                        thr = float(
                            getattr(
                                self.config.classification,
                                "unknown_margin_threshold",
                                getattr(
                                    self.config.classification,
                                    "rejection_margin_threshold",
                                    0.05,
                                ),
                            )
                        )

                        unknown_pred = (margin is not None) and (margin < thr)
                    elif strategy == "softmax_p":
                        T = float(getattr(self.config.classification, "softmax_temperature", 10.0))
                        # Stable softmax
                        m = max(vals) if vals else 0.0
                        exps = [math.exp((v - m) * T) for v in vals]
                        sum_exps = sum(exps) + 1e-12
                        p_max = max(exps) / sum_exps
                        thr = float(
                            getattr(
                                self.config.classification,
                                "unknown_pmax_threshold",
                                0.6,
                            )
                        )
                        unknown_pred = p_max < thr
                    else:  # 'zscore' or default
                        z = (best_score - mu) / sd
                        thr = float(getattr(self.config.classification, "unknown_z_threshold", 1.0))
                        unknown_pred = z < thr
            except Exception as e:
                self.logger.debug(f"Unknown rejection skipped: {e}")

            # If unknown, override prediction
            if unknown_pred:
                if return_scores or return_details:
                    results = {
                        "predicted_class": unknown_label,
                        "confidence": float(best_score),
                        "all_scores": class_scores,
                    }
                    if rejection_info is not None:
                        results["rejection"] = rejection_info
                    return unknown_label, results
                else:
                    return unknown_label

            if (
                getattr(self.config.classification, "rejection_enable", False)
                and len(class_scores) >= 2
            ):
                try:
                    sorted_pairs = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
                    second_score = sorted_pairs[1][1]
                    margin = float(best_score - second_score)
                    thr = float(
                        getattr(
                            self.config.classification,
                            "rejection_margin_threshold",
                            0.05,
                        )
                    )
                    rejection_info = {
                        "second_best_class": sorted_pairs[1][0],
                        "second_best_score": second_score,
                        "margin": margin,
                        "threshold": thr,
                        "rejected": margin < thr,
                    }
                except Exception as e:
                    self.logger.debug(f"Rejection computation skipped: {e}")

            # Prepare results
            if return_scores or return_details:
                results = {
                    "predicted_class": best_class,
                    "confidence": best_score,
                    "all_scores": class_scores,
                }

                if rejection_info is not None:
                    results["rejection"] = rejection_info

                if return_details:
                    results["detailed_analysis"] = detailed_results

                return best_class, results
            else:
                return best_class

    def _compute_class_similarity(
        self,
        query_features: Dict[str, torch.Tensor],
        class_name: str,
        return_details: bool = False,
    ) -> Tuple[float, Optional[Dict]]:
        """Compute similarity between query and a specific class"""
        details = {} if return_details else None

        # Get class prototypes and attention consensus
        visual_prototype = self.support_manager.get_prototype(class_name, "global")
        attention_consensus = self.support_manager.get_attention_consensus(class_name)

        if visual_prototype is None:
            self.logger.warning(f"No visual prototype found for class {class_name}")
            return 0.0, details

        # Convert prototype to tensor (use CPU if inference_device is cpu)
        inference_device = "cpu" if getattr(self.config.classification, "inference_device", "same") == "cpu" else self.config.model.device
        visual_prototype_tensor = torch.from_numpy(visual_prototype).to(inference_device)
        visual_prototype_tensor = normalize_tensor(visual_prototype_tensor)

        # Compute visual similarity (query_features already on correct device from classify())
        visual_sim = self._compute_visual_similarity(
            query_features, visual_prototype_tensor, attention_consensus, class_name
        )

        # Compute text similarity
        text_sim = self._compute_text_similarity(query_features, class_name)

        # Optional adapter-based support bank similarity (Tip-Adapter style)
        adapter_sim = None
        if getattr(self.config, "adapter", None) and self.config.adapter.use_tip_adapter:
            adapter_sim = self._compute_adapter_similarity(query_features, class_name)

        # Combine similarities
        if adapter_sim is not None:
            # Fuse adapter with text into a semantic prior, then combine with visual
            semantic_sim = (
                1 - self.config.adapter.alpha
            ) * text_sim + self.config.adapter.alpha * adapter_sim
            combined_sim = (
                self.config.classification.visual_weight * visual_sim
                + self.config.classification.text_weight * semantic_sim
            )
        else:
            combined_sim = (
                self.config.classification.visual_weight * visual_sim
                + self.config.classification.text_weight * text_sim
            )

        # Local patch enhancement (optional)
        if getattr(self.config.classification, "local_patch_enhance", False):
            try:
                patch_bonus = self._compute_local_patch_bonus(query_features, class_name)
                combined_sim = (
                    1 - self.config.classification.local_patch_weight
                ) * combined_sim + self.config.classification.local_patch_weight * patch_bonus
                if return_details:
                    if details is None:
                        details = {}
                    details["local_patch_bonus"] = patch_bonus
            except Exception as e:
                self.logger.debug(f"Local patch enhancement failed: {e}")

        # Optional small-head fusion (cosine score), default off unless enabled
        try:
            if getattr(self.config, "adapter", None) and getattr(
                self.config.adapter, "use_small_head", False
            ):
                head_sim = self._compute_head_score(query_features, class_name)
                if head_sim is not None:
                    combined_sim = 0.5 * combined_sim + 0.5 * head_sim
        except Exception as e:
            self.logger.debug(f"Small head fusion skipped: {e}")

        # Logit calibration (post-hoc): global temperature and per-class bias
        calibrated_sim = combined_sim
        try:
            # global temperature
            global_temp = getattr(self.config.classification, "global_logit_temperature", None)
            if global_temp is not None and global_temp > 0:
                calibrated_sim = calibrated_sim / global_temp
            # class bias
            class_biases = getattr(self.support_manager, "class_logit_bias", None)
            if (
                getattr(self.config.classification, "enable_logit_bias", True)
                and class_biases
                and class_name in class_biases
            ):
                calibrated_sim = calibrated_sim + float(class_biases[class_name])
            # visual-side per-class calibration
            vcal = getattr(self.support_manager, "visual_calibration", None)
            if vcal and class_name in vcal:
                try:
                    vtemp = float(vcal[class_name].get("temp", 1.0))
                    vbias = float(vcal[class_name].get("bias", 0.0))
                    calibrated_sim = (calibrated_sim + vbias) * vtemp
                except Exception:
                    pass

        except Exception:
            pass

        if return_details:
            if details is None:
                details = {}
            details.update(
                {
                    "visual_similarity": visual_sim,
                    "text_similarity": text_sim,
                    "adapter_similarity": adapter_sim,
                    "combined_similarity": combined_sim,
                    "calibrated_similarity": calibrated_sim,
                    "visual_weight": self.config.classification.visual_weight,
                    "text_weight": self.config.classification.text_weight,
                    "adapter_alpha": getattr(self.config.adapter, "alpha", None),
                    "global_logit_temperature": getattr(
                        self.config.classification, "global_logit_temperature", None
                    ),
                    "class_logit_bias": getattr(self.support_manager, "class_logit_bias", {}).get(
                        class_name, None
                    ),
                }
            )

        # Prototype EMA refinement using confident predictions
        try:
            if self.config.support_set.enable_prototype_ema:
                if combined_sim >= self.config.support_set.prototype_ema_min_conf:
                    m = self.config.support_set.prototype_ema_momentum
                    # Update global prototype
                    qg = query_features["global"]
                    if qg.dim() == 2:
                        qg = qg.squeeze(0)
                    self.support_manager.update_prototype_with_query(class_name, "global", qg, m)
                    # Optionally update layer_9 if available
                    if "layer_9" in query_features:
                        self.support_manager.update_prototype_with_query(
                            class_name, "layer_9", query_features["layer_9"], m
                        )
        except Exception as e:
            self.logger.debug(f"Prototype EMA update skipped: {e}")

        return calibrated_sim, details

    def _compute_visual_similarity(
        self,
        query_features: Dict[str, torch.Tensor],
        visual_prototype: torch.Tensor,
        attention_consensus: Optional[Dict[str, np.ndarray]],
        class_name: str,
    ) -> float:
        """Compute visual similarity with feature alignment"""
        query_global = query_features["global"]

        # Basic global similarity (optionally in reduced space)
        try:
            if getattr(self, "native_reducer", None) is not None and class_name in getattr(
                self, "native_reduced_prototypes", {}
            ):
                q = query_global
                if isinstance(q, torch.Tensor) and q.dim() == 2:
                    q = q.squeeze(0)
                q = q.float()
                q = q / (q.norm() + 1e-8)
                qr = self.native_reducer.transform(q)
                if isinstance(qr, torch.Tensor) and qr.dim() == 2:
                    qr = qr.squeeze(0)
                qr = qr.float()
                qr = qr / (qr.norm() + 1e-8)
                pr = self.native_reduced_prototypes[class_name]
                if pr.device != qr.device:
                    pr = pr.to(qr.device)
                pr = pr / (pr.norm() + 1e-8)
                global_sim = torch.dot(qr, pr).item()
            else:
                global_sim = cosine_similarity(query_global, visual_prototype, dim=-1).item()
        except Exception:
            global_sim = cosine_similarity(query_global, visual_prototype, dim=-1).item()

        # Masked-Global pooling fusion (foreground emphasis)
        try:
            if bool(getattr(self.config.classification, "enable_masked_global", False)):
                layer = str(getattr(self.config.classification, "masked_global_layer", "layer_9"))
                src = str(
                    getattr(
                        self.config.classification,
                        "masked_global_source",
                        "discriminative_mask",
                    )
                )
                w = float(getattr(self.config.classification, "masked_global_weight", 0.5))
                if layer in query_features:
                    # Build mask from support discriminative mask or attention consensus
                    mask_np = None
                    if src == "discriminative_mask":
                        mask_np = self.support_manager.get_discriminative_mask(class_name, layer)
                    elif src == "attention" and attention_consensus is not None:
                        # pick attn_9 if exists else any available
                        key = (
                            "attn_9"
                            if "attn_9" in attention_consensus
                            else next(iter(attention_consensus.keys()), None)
                        )
                        if key is not None:
                            mask_np = attention_consensus.get(key, None)
                    if mask_np is not None:
                        # Use device from prototype/query features instead of config
                        inference_device = visual_prototype.device
                        m = torch.from_numpy(mask_np).to(inference_device).float()
                        Fq = query_features[layer]
                        if isinstance(Fq, torch.Tensor) and Fq.dim() == 4 and Fq.shape[0] == 1:
                            Fq = Fq.squeeze(0)
                        Pm = self.support_manager.get_prototype(class_name, layer)
                        if Pm is not None:
                            Pv = torch.from_numpy(Pm).to(Fq.device).float()
                            # Resize mask if needed
                            if m.dim() == 2 and (
                                m.shape[0] != Fq.shape[-2] or m.shape[1] != Fq.shape[-1]
                            ):
                                m = F.interpolate(
                                    m.unsqueeze(0).unsqueeze(0),
                                    size=(Fq.shape[-2], Fq.shape[-1]),
                                    mode="bilinear",
                                    align_corners=True,
                                ).squeeze()
                            # Weighted average pooling
                            m_pos = (m - m.min()) / (m.max() - m.min() + 1e-8)
                            qv = F.normalize((Fq * m_pos).mean(dim=(-2, -1)), dim=-1)
                            if Pv.dim() == 4 and Pv.shape[0] == 1:
                                Pv = Pv.squeeze(0)
                            pv = F.normalize((Pv * m_pos).mean(dim=(-2, -1)), dim=-1)
                            masked_sim = torch.dot(qv, pv).item()
                            global_sim = (1.0 - w) * global_sim + w * masked_sim
        except Exception:
            pass

        # Global subcenters fusion on global features
        try:
            if bool(getattr(self.config.classification, "enable_global_subcenters", False)):
                centers_np = self.support_manager.get_subcenters(class_name, "global")
                if centers_np is not None and isinstance(query_global, torch.Tensor):
                    centers = torch.from_numpy(centers_np).to(query_global.device).float()
                    if query_global.dim() == 2:
                        qg = query_global.squeeze(0)
                    else:
                        qg = query_global
                    qg = F.normalize(qg, dim=-1)
                    centers = F.normalize(centers, dim=-1)
                    sims = torch.matmul(centers, qg)
                    sub_sim = float(torch.max(sims).item())
                    if bool(getattr(self.config.classification, "global_subcenters_fusion", True)):
                        gw = float(
                            getattr(
                                self.config.classification,
                                "global_subcenters_weight",
                                0.5,
                            )
                        )
                        global_sim = (1.0 - gw) * global_sim + gw * sub_sim
                    else:
                        global_sim = sub_sim
        except Exception:
            pass

        # Enhanced similarity with feature alignment
        if (
            self.config.classification.use_feature_alignment
            and attention_consensus
            and "attn_9" in attention_consensus
        ):
            try:
                # Get query attention
                query_attention = None
                for key in query_features:
                    if "attn" in key:
                        query_attention = query_features[key]
                        break

                if query_attention is not None:
                    # Get support attention consensus (use inference device)
                    inference_device = visual_prototype.device
                    support_attention = torch.from_numpy(attention_consensus["attn_9"]).to(
                        inference_device
                    )

                    # Align features (using layer features if available)
                    if "layer_9" in query_features:
                        aligned_features = self.feature_aligner.align_features(
                            query_features["layer_9"],
                            query_attention,
                            support_attention,
                        )

                        # Compute similarity with aligned features
                        aligned_global = aligned_features.mean(dim=(-2, -1))  # Global pool
                        aligned_global = normalize_tensor(aligned_global)

                        aligned_sim = cosine_similarity(
                            aligned_global, visual_prototype, dim=-1
                        ).item()

                        # Combine global and aligned similarities
                        global_sim = 0.6 * global_sim + 0.4 * aligned_sim

            except Exception as e:
                self.logger.warning(f"Feature alignment failed for {class_name}: {e}")

        # Neighborhood Subspace Suppression (NSS)
        try:
            if getattr(self.config.classification, "enable_nss", False):
                # Only act if we have precomputed bases
                bases = getattr(self.support_manager, "nss_bases", {})
                neighbors = getattr(self.support_manager, "nss_neighbors", {})
                if class_name in bases and isinstance(query_global, torch.Tensor):
                    B = bases[class_name]
                    # ensure numpy array -> torch tensor
                    if not isinstance(B, torch.Tensor):
                        B = torch.from_numpy(B)
                    B = B.to(visual_prototype.device).float()  # [D, r]
                    # Flatten query_global to [D]
                    q = query_global
                    if q.dim() == 2:
                        q = q.squeeze(0)
                    q = q.float()
                    # project onto neighbor subspace
                    # proj = B @ (B.T @ q)
                    Bt = B.t()
                    coeff = torch.matmul(Bt, q)
                    proj = torch.matmul(B, coeff)
                    # residual
                    res = q - proj
                    res = F.normalize(res, dim=-1)
                    # residual similarity
                    res_sim = cosine_similarity(res, visual_prototype, dim=-1).item()
                    # neighbor affinity: average sim to neighbor class prototypes
                    neigh_aff = 0.0
                    if class_name in neighbors:
                        neighs = neighbors[class_name]
                        cnt = 0
                        for nc in neighs:
                            proto_np = self.support_manager.get_prototype(nc, "global")
                            if proto_np is None:
                                continue
                            p = torch.from_numpy(proto_np).to(visual_prototype.device).float()
                            p = F.normalize(p, dim=-1)
                            s = cosine_similarity(q, p, dim=-1).item()
                            neigh_aff += s
                            cnt += 1
                        if cnt > 0:
                            neigh_aff /= cnt
                    # fuse
                    pw = float(getattr(self.config.classification, "nss_proj_weight", 0.5))
                    bw = float(getattr(self.config.classification, "nss_bias_weight", 0.05))
                    global_sim = (1 - pw) * global_sim + pw * res_sim - bw * neigh_aff
        except Exception as e:
            self.logger.debug(f"NSS step skipped for {class_name}: {e}")

        # Stage 1: Discriminative mining fusion (masked local-part + global)
        try:
            if (
                getattr(self.config.classification, "enable_discriminative_mining", False)
                and "layer_9" in query_features
            ):
                part_w = float(
                    getattr(self.config.classification, "discriminative_part_weight", 0.4)
                )
                glob_w = float(
                    getattr(self.config.classification, "discriminative_global_weight", 0.6)
                )
                # Compute part similarity via subcenters if available
                part_sim = self._compute_part_score(query_features, class_name)
                if part_sim is not None:
                    global_sim = glob_w * global_sim + part_w * part_sim
        except Exception as e:
            self.logger.debug(f"Discriminative fusion skipped for {class_name}: {e}")

        return global_sim

    def _compute_text_similarity(
        self, query_features: Dict[str, torch.Tensor], class_name: str
    ) -> float:
        """Compute text similarity using cached text features"""
        query_global = query_features["global"]

        if class_name in self.text_features_cache:
            text_feature = self.text_features_cache[class_name]
            base_sim = cosine_similarity(query_global, text_feature, dim=-1).item()
        else:
            # Fallback to basic text encoding
            basic_probe = f"a photo of a {class_name}"
            text_features = self.attribute_generator.encode_text_probes([basic_probe])
            text_feature = normalize_tensor(text_features[0])
            base_sim = cosine_similarity(query_global, text_feature, dim=-1).item()

        # Apply per-class text calibration if available via support manager
        try:
            calib = getattr(self.support_manager, "text_calibration", None)
            if calib and class_name in calib:
                temp = calib[class_name].get("temp", 1.0)
                bias = calib[class_name].get("bias", 0.0)
                text_sim = float((base_sim + bias) * temp)
            else:
                text_sim = base_sim
        except Exception:
            text_sim = base_sim

        return text_sim

    def _compute_part_score(
        self, query_features: Dict[str, torch.Tensor], class_name: str
    ) -> Optional[float]:
        """Compute masked local part similarity using class subcenters (Stage 1).
        Returns mean of max token-to-center sims; None if centers unavailable.
        """
        try:
            if "layer_9" not in query_features:
                return None
            centers_np = self.support_manager.get_subcenters(class_name, "layer_9")
            if centers_np is None:
                return None
            Fq = query_features["layer_9"]  # [C,H,W] or [1,C,H,W]
            if Fq.dim() == 4:
                Fq = Fq.squeeze(0)
            C, H, W = Fq.shape
            T = F.normalize(Fq.view(C, -1).t(), dim=1)  # [HW, C]
            # Apply discriminative mask if exists
            mask_np = self.support_manager.get_discriminative_mask(class_name, "layer_9")
            if mask_np is not None and mask_np.shape == (H, W):
                m = torch.from_numpy(mask_np.astype(np.bool_)).to(T.device).view(-1)
                if m.any():
                    T = T[m]
            # Centers
            centers = torch.from_numpy(centers_np).to(T.device).float()  # [K, C]
            centers = F.normalize(centers, dim=1)
            sims = torch.matmul(T, centers.t())  # [HWm, K]
            max_per_center = sims.max(dim=0).values  # [K]
            return float(max_per_center.mean().item())
        except Exception:
            return None

    def _svm_scores_for_features(
        self,
        query_features: Dict[str, torch.Tensor],
        allowed_classes: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Compute SVM scores from query 'global' feature (optionally augmented with text sim vector)
        and optionally filter allowed classes."""
        if self.svm is None:
            raise RuntimeError("SVM is not initialized")
        if "global" not in query_features:
            raise RuntimeError("Global feature not found for SVM scoring")
        qg = query_features["global"]
        if isinstance(qg, torch.Tensor) and qg.dim() == 2:
            qg = qg.squeeze(0)

        # Normalize
        qn = qg / (qg.norm() + 1e-8)

        # Optional: apply DR first on visual feature if SVM was trained with DR
        q_base = qn
        try:
            if getattr(self.svm, "reducer", None) is not None:
                q_base = self.svm.reducer.transform(qn)
        except Exception:
            q_base = qn

        # Optionally augment with text similarities if model expects it or config enables it
        qx = q_base
        try:
            use_text_feat = bool(
                getattr(self.config.classification, "use_text_sim_features", False)
            )
            beta = float(getattr(self.config.classification, "text_feat_beta", 0.5))
            expected_with_text = getattr(
                self.svm, "feature_dim", None
            ) is not None and self.svm.feature_dim == q_base.shape[-1] + len(self.svm.class_names)
            if use_text_feat or expected_with_text:
                # Build S vector [C] computed from pre-reduction normalized visual qn
                t_list = []
                for cname in self.svm.class_names:
                    tf = self.text_features_cache.get(cname, None)
                    if tf is None:
                        # Fallback: basic probe
                        basic = f"a photo of a {cname}"
                        tf = self.attribute_generator.encode_text_probes([basic])[0]
                        tf = tf / (tf.norm() + 1e-8)
                    else:
                        if tf.dim() == 2:
                            tf = tf.squeeze(0)
                    t_list.append(tf)
                T = torch.stack(t_list, dim=0).to(qg.device)  # [C, D]
                S = torch.matmul(T, qn)  # [C]
                mu = S.mean()
                sd = S.std() + 1e-8
                Sz = (S - mu) / sd
                qx = torch.cat([q_base, beta * Sz], dim=-1)
        except Exception:
            qx = q_base

        pred_cls, scores = self.svm.predict_from_features(qx)
        # Optional filter by allowed classes
        if allowed_classes is not None:
            allowed_set = set(allowed_classes)
            scores = {k: v for k, v in scores.items() if k in allowed_set}
            if not scores:
                # If none allowed, return original but signal low confidence
                return pred_cls, {pred_cls: 0.0}
            pred_cls = max(scores.items(), key=lambda x: x[1])[0]
        # Renormalize to sum=1 for stability
        ssum = float(sum(scores.values()))
        if ssum > 0:
            scores = {k: float(v / ssum) for k, v in scores.items()}
        return pred_cls, scores

    def _init_svm(self) -> None:
        """Initialize SVM from config: load if path exists else train from support set and save."""
        cc = self.config.classification
        svm_cfg = SVMConfig(
            kernel=getattr(cc, "svm_kernel", "rbf"),
            C=float(getattr(cc, "svm_C", 1.0)),
            gamma=getattr(cc, "svm_gamma", "scale"),
            probability=bool(getattr(cc, "svm_probability", True)),
            feature_layer=getattr(cc, "svm_feature_layer", "global"),
            use_class_weight=bool(getattr(cc, "svm_use_class_weight", True)),
        )
        model_path = getattr(cc, "svm_model_path", None)
        if model_path:
            p = Path(str(model_path))
            if p.exists():
                self.svm = SVMClassifier.load(p)
                self.logger.info(f"Loaded SVM model from {p}")
                return
        # Train from support set only if embeddings are available; otherwise skip to keep fast init
        embed_src = getattr(self.support_manager, "embedding_source", None)
        if embed_src not in ("embed", "images"):
            raise RuntimeError(
                "SVM model path missing and support embeddings not loaded (fast init)."
            )
        self.svm = SVMClassifier(class_names=self.support_manager.class_names, cfg=svm_cfg)
        self.svm.fit_from_support_manager(
            self.support_manager,
            svm_cfg.feature_layer,
            text_features=self.text_features_cache,
            use_text_sim_features=bool(getattr(cc, "use_text_sim_features", False)),
            beta=float(getattr(cc, "text_feat_beta", 0.5)),
        )
        self.logger.info("Trained SVM on support-set embeddings")
        # Save if path provided
        if model_path:
            p = Path(str(model_path))
            try:
                self.svm.save(p)
                self.logger.info(f"Saved SVM model to {p}")
            except Exception as e:
                self.logger.warning(f"Failed to save SVM model to {p}: {e}")

        return None

    def _nn_scores_for_features(
        self,
        query_features: Dict[str, torch.Tensor],
        allowed_classes: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Compute NN scores from query 'global' feature and optionally filter allowed classes."""
        if getattr(self, "nn", None) is None:
            raise RuntimeError("NN is not initialized")
        if "global" not in query_features:
            raise RuntimeError("Global feature not found for NN scoring")
        qg = query_features["global"]
        if isinstance(qg, torch.Tensor) and qg.dim() == 2:
            qg = qg.squeeze(0)
        pred_cls, scores = self.nn.predict_from_features(qg)
        if allowed_classes is not None:
            allowed_set = set(allowed_classes)
            scores = {k: v for k, v in scores.items() if k in allowed_set}
            if not scores:
                return pred_cls, {pred_cls: 0.0}
            pred_cls = max(scores.items(), key=lambda x: x[1])[0]
        # Renormalize
        ssum = float(sum(scores.values()))
        if ssum > 0:
            scores = {k: float(v / ssum) for k, v in scores.items()}
        return pred_cls, scores

    def _init_native_reducer(self) -> None:
        """Fit a dimensionality reducer on support-set visual embeddings and
        precompute reduced global prototypes for native classifier path.
        """
        try:
            cc = self.config.classification
            dr_cfg = DimReduceConfig(
                method=str(getattr(cc, "dim_reduction_method", "pca")),
                n_components=getattr(cc, "dim_reduction_n_components", 0.95),
                whiten=bool(getattr(cc, "dim_reduction_whiten", False)),
                random_state=getattr(cc, "dim_reduction_random_state", 42),
            )
            # Build training matrix X from per-image embeddings if available
            X_list = []
            used_records = False
            try:
                if isinstance(getattr(self.support_manager, "embed_records", None), dict):
                    for cname in self.support_manager.class_names:
                        recs = self.support_manager.embed_records.get(cname, [])
                        if recs:
                            Xc = np.stack(
                                [
                                    np.asarray(r.get("global", []), dtype=np.float32).reshape(-1)
                                    for r in recs
                                    if r.get("global") is not None
                                ],
                                axis=0,
                            )
                            if Xc.size > 0:
                                X_list.append(Xc)
                                used_records = True
            except Exception:
                used_records = False
            if not used_records and getattr(self.support_manager, "embedding_source", None) in (
                "embed",
                "images",
            ):
                for cname in self.support_manager.class_names:
                    bank = self.support_manager.adapter_support_bank.get(cname)
                    if bank is not None and isinstance(bank, torch.Tensor) and bank.numel() > 0:
                        Xc = bank.detach().cpu().numpy()
                        X_list.append(Xc)
            # Fallback to prototypes if no per-image embeddings
            if not X_list:
                for cname in self.support_manager.class_names:
                    p = self.support_manager.get_prototype(cname, "global")
                    if p is not None:
                        X_list.append(np.asarray(p, dtype=np.float32).reshape(1, -1))
            if not X_list:
                self.logger.warning("Native reducer: no embeddings available; skipping")
                return
            X = np.concatenate(X_list, axis=0).astype(np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            Xn = X / norms
            # Fit reducer
            self.native_reducer = DimensionalityReducer(dr_cfg).fit(Xn)
            # Precompute reduced prototypes per class
            self.native_reduced_prototypes = {}
            for cname in self.support_manager.class_names:
                proto_np = self.support_manager.get_prototype(cname, "global")
                if proto_np is None:
                    continue
                pt = torch.from_numpy(np.asarray(proto_np, dtype=np.float32)).to(
                    self.config.model.device
                )
                pt = pt / (pt.norm() + 1e-8)
                pr = self.native_reducer.transform(pt)
                if isinstance(pr, torch.Tensor) and pr.dim() == 2:
                    pr = pr.squeeze(0)
                pr = pr.to(self.config.model.device).float()
                pr = pr / (pr.norm() + 1e-8)
                self.native_reduced_prototypes[cname] = pr
            self.logger.info(
                f"Native reducer initialized with method={dr_cfg.method}, output_dim={self.native_reducer.output_dim}"
            )
        except Exception as e:
            self.logger.warning(f"Native reducer initialization failed: {e}")

    def _init_nn(self) -> None:
        """Initialize NN from config: load if path exists else train from support set and save."""
        cc = self.config.classification
        nn_cfg = NNConfig(
            hidden_sizes=list(getattr(cc, "nn_hidden_sizes", [512])),
            activation=getattr(cc, "nn_activation", "relu"),
            dropout=float(getattr(cc, "nn_dropout", 0.1)),
            epochs=int(getattr(cc, "nn_epochs", 20)),
            lr=float(getattr(cc, "nn_lr", 1e-3)),
            weight_decay=float(getattr(cc, "nn_weight_decay", 1e-4)),
            batch_size=int(getattr(cc, "nn_batch_size", 64)),
            early_stopping_patience=int(getattr(cc, "nn_early_stopping_patience", 5)),
            class_weight_balanced=bool(getattr(cc, "nn_class_weight_balanced", True)),
            feature_layer=getattr(cc, "nn_feature_layer", "global"),
            model_path=getattr(cc, "nn_model_path", None),
            device=getattr(self.config.model, "device", "cpu"),
        )
        model_path = getattr(nn_cfg, "model_path", None)
        if model_path:
            p = Path(str(model_path))
            if p.exists():
                self.nn = NNClassifier.load(p)
                self.logger.info(f"Loaded NN model from {p}")
                return
        # Train from support set only if embeddings are available; otherwise skip to keep fast init
        embed_src = getattr(self.support_manager, "embedding_source", None)
        if embed_src not in ("embed", "images"):
            raise RuntimeError(
                "NN model path missing and support embeddings not loaded (fast init)."
            )
        self.nn = NNClassifier(class_names=self.support_manager.class_names, cfg=nn_cfg)
        self.nn.fit_from_support_manager(
            self.support_manager,
            nn_cfg.feature_layer,
            text_features=self.text_features_cache,
            use_text_sim_features=bool(getattr(cc, "use_text_sim_features", False)),
            beta=float(getattr(cc, "text_feat_beta", 0.5)),
        )
        self.logger.info("Trained NN on support-set embeddings")
        # Save if path provided
        if model_path:
            p = Path(str(model_path))
            try:
                self.nn.save(p)
                self.logger.info(f"Saved NN model to {p}")
            except Exception as e:
                self.logger.warning(f"Failed to save NN model to {p}: {e}")

    def _compute_head_score(
        self, query_features: Dict[str, torch.Tensor], class_name: str
    ) -> Optional[float]:
        """Compute cosine score from small head if available (no margin at inference)."""
        try:
            sh = getattr(self.support_manager, "small_head", None)
            if sh is None or sh.get("W") is None:
                return None
            if "global" not in query_features:
                return None
            # query global feature
            qg = query_features["global"]
            if qg.dim() == 2:
                qg = qg.squeeze(0)
            qg = F.normalize(qg, dim=0)
            # class index
            idx = self.support_manager.class_names.index(class_name)
            W = torch.from_numpy(sh["W"]).to(qg.device).float()  # [C, D]
            w = F.normalize(W[idx], dim=0)
            return float(torch.dot(qg, w).item())
        except Exception:
            return None

    def _compute_local_patch_bonus(
        self, query_features: Dict[str, torch.Tensor], class_name: str
    ) -> float:
        """Local patch matching bonus using aligned intermediate features.
        Heuristic: compare top-K query patches against class prototype patches on a chosen layer.
        """
        layer_key = "layer_9" if "layer_9" in query_features else None
        if layer_key is None:
            return 0.0
        Fq = query_features[layer_key]  # [C, H, W] or [1, C, H, W]
        if Fq.dim() == 4:
            Fq = Fq.squeeze(0)
        C, H, W = Fq.shape
        Fq = F.normalize(Fq.view(C, -1), dim=0)  # [C, HW]
        # Apply discriminative mask if available
        try:
            if getattr(self.config.classification, "enable_discriminative_mining", False):
                mask_np = self.support_manager.get_discriminative_mask(class_name, layer_key)
                if mask_np is not None and mask_np.shape == (H, W):
                    m = torch.from_numpy(mask_np.astype(np.bool_)).to(Fq.device).view(-1)
                    if m.any():
                        Fq = Fq[:, m]
        except Exception:
            pass
        # Build (or approximate) class prototype patches by using prototype tensor if available
        proto_np = self.support_manager.get_prototype(class_name, layer_key)
        if proto_np is None:
            return 0.0
        P = torch.from_numpy(proto_np).to(self.config.model.device)
        if P.dim() == 4:
            P = P.squeeze(0)
        Pc, _, _ = P.shape
        P = F.normalize(P.view(Pc, -1), dim=0)  # [C, HWp]
        # Compute similarity matrix between query patches and prototype patches
        S = torch.matmul(Fq.t(), P)  # [HWq, HWp]
        # Take top-k per query patch, then average
        topk = max(1, int(self.config.classification.local_patch_topk))
        vals, _ = torch.topk(S, k=min(topk, S.shape[1]), dim=1)
        patch_score = vals.mean().item()
        return float(patch_score)

    def _compute_adapter_similarity(
        self, query_features: Dict[str, torch.Tensor], class_name: str
    ) -> Optional[float]:
        """Compute similarity between query and support bank (Tip-Adapter style).
        For each class, we keep support global features. We compute cosine similarity
        between query global and each support, apply softmax with temperature beta to
        get weights, and aggregate similarities to produce a class score.
        """
        try:
            device = self.config.model.device

            # Check if we should use subcenters instead of full adapter support bank
            if getattr(self.config.adapter, "use_subcenters_for_adapter", False):
                centers = self.support_manager.get_subcenters(class_name, "global")
                if centers is not None:
                    support_bank = torch.from_numpy(centers).to(device)
                else:
                    # Fallback to prototype if subcenters not available
                    proto = self.support_manager.get_prototype(class_name, "global")
                    if proto is not None:
                        support_bank = torch.from_numpy(proto).unsqueeze(0).to(device)
                    else:
                        return None
            else:
                support_bank = self.support_manager.adapter_support_bank.get(class_name)
                if support_bank is None or support_bank.numel() == 0:
                    return None
                support_bank = support_bank.to(device)

            q = query_features["global"]
            if q.dim() == 2:
                q = q.squeeze(0)
            # Normalize
            q = F.normalize(q.float(), dim=-1)
            S = F.normalize(support_bank.float(), dim=-1)
            # [shots]
            sims = torch.matmul(S, q)
            # Temperature softmax
            beta = getattr(self.config.adapter, "beta", 50.0)
            weights = torch.softmax(beta * sims, dim=0)
            adapter_score = torch.sum(weights * sims).item()
            return float(adapter_score)
        except Exception as e:
            self.logger.warning(f"Adapter similarity failed for {class_name}: {e}")
            return None

    def classify_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
        return_scores: bool = True,
        allowed_classes: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Classify a batch of images

        Args:
            image_inputs: List of input images
            batch_size: Batch size for processing
            return_scores: Whether to return confidence scores

        Returns:
            List of classification results
        """
        results = []

        for i in range(0, len(image_inputs), batch_size):
            batch_inputs = image_inputs[i : i + batch_size]

            with Timer(f"Processing batch {i // batch_size + 1}"):
                batch_results = []
                for image_input in batch_inputs:
                    try:
                        if return_scores:
                            pred_class, details = self.classify(
                                image_input,
                                return_scores=True,
                                allowed_classes=allowed_classes,
                            )
                            batch_results.append(
                                {
                                    "image": str(image_input),
                                    "predicted_class": pred_class,
                                    "confidence": details["confidence"],
                                    "all_scores": details["all_scores"],
                                }
                            )
                        else:
                            pred_class = self.classify(
                                image_input,
                                return_scores=False,
                                allowed_classes=allowed_classes,
                            )
                            batch_results.append(
                                {
                                    "image": str(image_input),
                                    "predicted_class": pred_class,
                                }
                            )
                    except Exception as e:
                        self.logger.error(f"Failed to classify {image_input}: {e}")
                        batch_results.append(
                            {
                                "image": str(image_input),
                                "predicted_class": "ERROR",
                                "error": str(e),
                            }
                        )

                results.extend(batch_results)

        return results

    def evaluate_confidence(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate classification confidence"""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        top1_score = sorted_scores[0][1]
        top2_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

        confidence_gap = top1_score - top2_score

        # Determine confidence level
        if confidence_gap > 0.3:
            confidence_level = "HIGH"
        elif confidence_gap > 0.1:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        return {
            "top1_score": top1_score,
            "top2_score": top2_score,
            "confidence_gap": confidence_gap,
            "confidence_level": confidence_level,
            "is_confident": confidence_gap > self.config.classification.confidence_threshold,
        }

    def get_top_k_predictions(
        self, scores: Dict[str, float], k: int = 3
    ) -> List[Tuple[str, float]]:
        """Get top-k predictions with scores"""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:k]

    def update_weights(self, visual_weight: float, text_weight: float):
        """Update classification weights"""
        if abs(visual_weight + text_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        self.config.classification.visual_weight = visual_weight
        self.config.classification.text_weight = text_weight

        self.logger.info(f"Updated weights: visual={visual_weight}, text={text_weight}")

    def save_classification_results(self, results: List[Dict], output_path: Union[str, Path]):
        """Save classification results to file"""
        save_results(results, output_path)
        self.logger.info(f"Results saved to: {output_path}")

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and statistics"""
        return {
            "num_classes": len(self.support_manager.class_names),
            "class_names": self.support_manager.class_names,
            "feature_extractor_model": self.feature_extractor.model_name,
            "device": str(self.config.model.device),
            "visual_weight": self.config.classification.visual_weight,
            "text_weight": self.config.classification.text_weight,
            "use_feature_alignment": self.config.classification.use_feature_alignment,
            "confidence_threshold": self.config.classification.confidence_threshold,
            "support_set_stats": self.support_manager.class_statistics,
        }

    def clear_caches(self):
        """Clear all caches"""
        self.feature_extractor.clear_cache()
        self.attribute_generator.clear_cache()
        self.text_features_cache.clear()
        self.logger.info("All caches cleared")

    def __repr__(self):
        return (
            f"FineGrainedClassifier("
            f"classes={len(self.support_manager.class_names)}, "
            f"model={self.feature_extractor.model_name}, "
            f"device={self.config.model.device})"
        )


def create_classifier(support_manager: SupportSetManager, config: Config) -> FineGrainedClassifier:
    """Factory function to create classifier"""
    return FineGrainedClassifier(support_manager=support_manager, config=config)
