"""
Feature Alignment Module for CLIP Few-Shot Classification

This module implements the FeatureAligner class that performs spatial alignment
of query features with support set prototypes using attention maps and optical flow.
This addresses pose variations and spatial misalignments in fine-grained classification.

Key Features:
- Attention-guided spatial alignment
- Optical flow-based deformation field computation
- Thin-plate spline transformation for smooth warping
- Multi-scale alignment support
- Efficient GPU-accelerated operations
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Union
import logging
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata

from ..core.config import Config
from ..core.utils import Timer


class FeatureAligner:
    """
    Feature alignment using attention maps and spatial transformations

    Aligns query image features with support set prototypes by computing
    deformation fields based on attention map differences and applying
    spatial transformations to the feature maps.
    """

    def __init__(
        self,
        device: str = "cuda",
        alignment_strength: float = 0.1,
        use_optical_flow: bool = True,
        use_tps: bool = False,
    ):
        """
        Initialize feature aligner

        Args:
            device: Device for computations
            alignment_strength: Strength of alignment deformation
            use_optical_flow: Whether to use optical flow for alignment
            use_tps: Whether to use thin-plate spline transformation
        """
        self.device = torch.device(device)
        self.alignment_strength = alignment_strength
        self.use_optical_flow = use_optical_flow
        self.use_tps = use_tps

        self.logger = logging.getLogger(__name__)

    def align_features(
        self,
        query_features: torch.Tensor,
        query_attention: torch.Tensor,
        support_attention: torch.Tensor,
        method: str = "optical_flow",
    ) -> torch.Tensor:
        """
        Align query features using attention maps

        Args:
            query_features: Query image features [C, H, W] or [B, C, H, W]
            query_attention: Query attention map [H, W] or [B, H, W]
            support_attention: Support attention map [H, W] or [B, H, W]
            method: Alignment method ("optical_flow", "tps", "simple")

        Returns:
            Aligned query features with same shape as input
        """
        # Ensure tensors are on correct device
        query_features = query_features.to(self.device)
        query_attention = query_attention.to(self.device)
        support_attention = support_attention.to(self.device)

        # Handle batch dimension
        if query_features.dim() == 3:
            query_features = query_features.unsqueeze(0)
            query_attention = query_attention.unsqueeze(0)
            support_attention = support_attention.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Choose alignment method
        if method == "optical_flow" and self.use_optical_flow:
            aligned_features = self._align_with_optical_flow(
                query_features, query_attention, support_attention
            )
        elif method == "tps" and self.use_tps:
            aligned_features = self._align_with_tps(
                query_features, query_attention, support_attention
            )
        else:
            aligned_features = self._align_simple(
                query_features, query_attention, support_attention
            )

        # Remove batch dimension if it was added
        if squeeze_output:
            aligned_features = aligned_features.squeeze(0)

        return aligned_features

    def _align_with_optical_flow(
        self,
        query_features: torch.Tensor,
        query_attention: torch.Tensor,
        support_attention: torch.Tensor,
    ) -> torch.Tensor:
        """Align features using optical flow"""
        batch_size, channels, height, width = query_features.shape
        aligned_features = []

        for b in range(batch_size):
            # Convert attention maps to numpy for OpenCV
            query_attn_np = query_attention[b].cpu().numpy()
            support_attn_np = support_attention[b].cpu().numpy()

            # Resize attention maps to feature map size if needed
            if query_attn_np.shape != (height, width):
                query_attn_np = cv2.resize(query_attn_np, (width, height))
                support_attn_np = cv2.resize(support_attn_np, (width, height))

            # Normalize attention maps
            query_attn_norm = self._normalize_attention(query_attn_np)
            support_attn_norm = self._normalize_attention(support_attn_np)

            # Compute optical flow
            flow = self._compute_optical_flow(support_attn_norm, query_attn_norm)

            # Create deformation grid
            grid = self._create_deformation_grid(flow, height, width)
            grid_tensor = torch.from_numpy(grid).float().to(self.device).unsqueeze(0)

            # Apply deformation to features
            feature_batch = query_features[b : b + 1]
            aligned_feature = F.grid_sample(
                feature_batch,
                grid_tensor,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

            aligned_features.append(aligned_feature)

        return torch.cat(aligned_features, dim=0)

    def _align_with_tps(
        self,
        query_features: torch.Tensor,
        query_attention: torch.Tensor,
        support_attention: torch.Tensor,
    ) -> torch.Tensor:
        """Align features using thin-plate spline transformation"""
        batch_size, channels, height, width = query_features.shape
        aligned_features = []

        for b in range(batch_size):
            # Get attention maps
            query_attn = query_attention[b].cpu().numpy()
            support_attn = support_attention[b].cpu().numpy()

            # Resize if needed
            if query_attn.shape != (height, width):
                query_attn = cv2.resize(query_attn, (width, height))
                support_attn = cv2.resize(support_attn, (width, height))

            # Find control points from attention peaks
            query_points = self._find_attention_peaks(query_attn)
            support_points = self._find_attention_peaks(support_attn)

            # Create TPS transformation
            if len(query_points) >= 3 and len(support_points) >= 3:
                grid = self._create_tps_grid(
                    query_points, support_points, height, width
                )
                grid_tensor = (
                    torch.from_numpy(grid).float().to(self.device).unsqueeze(0)
                )

                # Apply transformation
                feature_batch = query_features[b : b + 1]
                aligned_feature = F.grid_sample(
                    feature_batch,
                    grid_tensor,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
            else:
                # Fallback to no alignment if insufficient control points
                aligned_feature = query_features[b : b + 1]

            aligned_features.append(aligned_feature)

        return torch.cat(aligned_features, dim=0)

    def _align_simple(
        self,
        query_features: torch.Tensor,
        query_attention: torch.Tensor,
        support_attention: torch.Tensor,
    ) -> torch.Tensor:
        """Simple alignment based on attention center shift"""
        batch_size, channels, height, width = query_features.shape
        aligned_features = []

        for b in range(batch_size):
            # Get attention maps
            query_attn = query_attention[b]
            support_attn = support_attention[b]

            # Resize if needed
            if query_attn.shape != (height, width):
                query_attn = F.interpolate(
                    query_attn.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze()
                support_attn = F.interpolate(
                    support_attn.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze()

            # Compute attention centers
            query_center = self._compute_attention_center(query_attn)
            support_center = self._compute_attention_center(support_attn)

            # Compute shift
            shift = (support_center - query_center) * self.alignment_strength

            # Create translation grid
            grid = self._create_translation_grid(shift, height, width)
            grid_tensor = grid.to(self.device).unsqueeze(0)

            # Apply translation
            feature_batch = query_features[b : b + 1]
            aligned_feature = F.grid_sample(
                feature_batch,
                grid_tensor,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

            aligned_features.append(aligned_feature)

        return torch.cat(aligned_features, dim=0)

    def _normalize_attention(self, attention: np.ndarray) -> np.ndarray:
        """Normalize attention map to [0, 255] uint8"""
        attention_norm = (attention - attention.min()) / (
            attention.max() - attention.min() + 1e-8
        )
        return (attention_norm * 255).astype(np.uint8)

    def _compute_optical_flow(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Compute optical flow between attention maps"""
        # Use Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            src,
            dst,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Scale flow by alignment strength
        flow *= self.alignment_strength

        return flow

    def _create_deformation_grid(
        self, flow: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """Create deformation grid from optical flow"""
        # Create base grid
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)

        # Apply flow
        flow_x = flow[:, :, 0] / width * 2  # Normalize to [-1, 1]
        flow_y = flow[:, :, 1] / height * 2

        grid_x = xx + flow_x
        grid_y = yy + flow_y

        # Clamp to valid range
        grid_x = np.clip(grid_x, -1, 1)
        grid_y = np.clip(grid_y, -1, 1)

        # Stack to create grid [H, W, 2]
        grid = np.stack([grid_x, grid_y], axis=-1)

        return grid

    def _find_attention_peaks(
        self, attention: np.ndarray, num_peaks: int = 5
    ) -> np.ndarray:
        """Find peaks in attention map for control points"""
        # Apply Gaussian blur to smooth attention
        attention_smooth = cv2.GaussianBlur(attention, (5, 5), 1.0)

        # Find local maxima
        from scipy.ndimage import maximum_filter

        local_maxima = maximum_filter(attention_smooth, size=3) == attention_smooth

        # Get peak coordinates
        peak_coords = np.column_stack(np.where(local_maxima))

        # Sort by attention value and take top peaks
        peak_values = attention_smooth[local_maxima]
        sorted_indices = np.argsort(peak_values)[::-1]

        top_peaks = peak_coords[sorted_indices[:num_peaks]]

        # Convert to (x, y) format and normalize to [-1, 1]
        height, width = attention.shape
        peaks_normalized = np.column_stack(
            [
                (top_peaks[:, 1] / width) * 2 - 1,  # x coordinate
                (top_peaks[:, 0] / height) * 2 - 1,  # y coordinate
            ]
        )

        return peaks_normalized

    def _create_tps_grid(
        self, src_points: np.ndarray, dst_points: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """Create thin-plate spline transformation grid"""
        # Match points (simple nearest neighbor for now)
        if len(src_points) != len(dst_points):
            min_len = min(len(src_points), len(dst_points))
            src_points = src_points[:min_len]
            dst_points = dst_points[:min_len]

        # Create target grid
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)
        target_points = np.column_stack([xx.ravel(), yy.ravel()])

        # Compute TPS transformation (simplified)
        if len(src_points) >= 3:
            # Use griddata for interpolation
            grid_x = griddata(
                src_points,
                dst_points[:, 0],
                target_points,
                method="linear",
                fill_value=0,
            )
            grid_y = griddata(
                src_points,
                dst_points[:, 1],
                target_points,
                method="linear",
                fill_value=0,
            )

            grid_x = grid_x.reshape(height, width)
            grid_y = grid_y.reshape(height, width)
        else:
            # Fallback to identity
            grid_x = xx
            grid_y = yy

        # Stack to create grid
        grid = np.stack([grid_x, grid_y], axis=-1)

        return grid

    def _compute_attention_center(self, attention: torch.Tensor) -> torch.Tensor:
        """Compute center of mass of attention map"""
        height, width = attention.shape

        # Create coordinate grids
        y_coords = torch.arange(height, dtype=torch.float32, device=attention.device)
        x_coords = torch.arange(width, dtype=torch.float32, device=attention.device)

        # Compute weighted centers
        total_weight = attention.sum()
        if total_weight > 0:
            center_y = (attention.sum(dim=1) * y_coords).sum() / total_weight
            center_x = (attention.sum(dim=0) * x_coords).sum() / total_weight
        else:
            center_y = height / 2
            center_x = width / 2

        # Normalize to [-1, 1]
        center_y = (center_y / height) * 2 - 1
        center_x = (center_x / width) * 2 - 1

        return torch.tensor([center_x, center_y], device=attention.device)

    def _create_translation_grid(
        self, shift: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Create translation grid"""
        # Create base grid
        x = torch.linspace(-1, 1, width, device=self.device)
        y = torch.linspace(-1, 1, height, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing="xy")

        # Apply shift
        grid_x = xx + shift[0]
        grid_y = yy + shift[1]

        # Clamp to valid range
        grid_x = torch.clamp(grid_x, -1, 1)
        grid_y = torch.clamp(grid_y, -1, 1)

        # Stack to create grid [H, W, 2]
        grid = torch.stack([grid_x, grid_y], dim=-1)

        return grid

    def compute_alignment_quality(
        self, aligned_attention: torch.Tensor, target_attention: torch.Tensor
    ) -> float:
        """Compute quality metric for alignment"""
        # Compute correlation between aligned and target attention
        aligned_flat = aligned_attention.flatten()
        target_flat = target_attention.flatten()

        # Normalize
        aligned_norm = (aligned_flat - aligned_flat.mean()) / (
            aligned_flat.std() + 1e-8
        )
        target_norm = (target_flat - target_flat.mean()) / (target_flat.std() + 1e-8)

        # Compute correlation
        correlation = torch.mean(aligned_norm * target_norm)

        return correlation.item()


def create_feature_aligner(config: Config) -> FeatureAligner:
    """Factory function to create feature aligner from config"""
    return FeatureAligner(
        device=config.model.device,
        alignment_strength=config.classification.alignment_strength,
        use_optical_flow=True,
        use_tps=False,
    )
