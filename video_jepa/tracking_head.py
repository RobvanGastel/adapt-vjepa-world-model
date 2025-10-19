import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from typing import Tuple, Mapping

from video_jepa.transformations import (
    convert_grid_coordinates, map_coordinates_3d, convert_grid_coordinates, heatmaps_to_points
)

class TrackingHead(nn.Module):
    def __init__(self, softmax_temperature: float = 20.0):
        super().__init__()
        self.softmax_temperature = softmax_temperature
        self.occ_linear = nn.Linear(3, 2) 

    def get_query_features(
            self,
            features,
            query_points
    ):
        B, T, H, W, D = features.shape

        position_in_grid = convert_grid_coordinates( 
            query_points,
            (T, H, W), # Shape of the *grid* (Time, Height, Width)
            (T, H, W), # Target grid shape (same as source for now)
            coordinate_format='tyx',
        )

        # Reshape for interpolation: B N 1 3 -> B N 3
        position_support = position_in_grid[..., None, :]
        position_support = rearrange(position_support, 'b n s c -> b (n s) c')

        query_feats = map_coordinates_3d(features, position_support) 
        return query_feats

    def estimate_trajectory(
            self,
            inter_query_points,
            features,
            query_points,
            video_size
    ):
        """Samples the feature grid at the query point locations."""
        
        
        # 1. Compute Cost Volume (Similarity)
        # Einsum: (B N D) x (B T H W D) -> (T B N H W)
        cost_volume = torch.einsum(
            'bnd,bthwd->tbnhw',
            inter_query_points,
            features,
        )
        
        # Reshape for Softmax (Flatten H*W)
        pos = rearrange(cost_volume, 't b n h w -> b n t h w') # B N T H W
        pos_sm = pos.reshape(pos.size(0), pos.size(1), pos.size(2), -1) # B N T (H*W)
        softmaxed = F.softmax(pos_sm * self.softmax_temperature, dim=-1) # B N T (H*W)
        pos = softmaxed.view_as(pos) # B N T H W

        # 2. Convert Heatmaps to Points
        # Assumes original video size is passed to utility for correct coordinate scaling
        tracks = heatmaps_to_points(pos, video_size, query_points=query_points)
        
        # 3. Predict Occlusion (LocoTrack's logic)
        occlusion_features = torch.cat(
            [
                torch.mean(cost_volume, dim=(-1, -2))[..., None], # Mean
                torch.amax(cost_volume, dim=(-1, -2))[..., None], # Max
                torch.amin(cost_volume, dim=(-1, -2))[..., None], # Min
            ], dim=-1 # Result is T B N 3
        )
        
        # Rearrange to B N T 3 for the linear layer
        occlusion_features = rearrange(occlusion_features, 't b n c -> b n t c')

        # Linear projection for occlusion/expected_dist logits
        occ_logits = self.occ_linear(occlusion_features.detach()) # B N T 2
        
        # Separate the logits
        occlusion = occ_logits[..., 0:1] # Occlusion logit: B N T 1
        expected_dist = occ_logits[..., 1:2] # Uncertainty logit: B N T 1

        # Final output shapes: B N T, B N T 2, B N T
        return {
            'tracks': tracks, 
            'occlusion': rearrange(occlusion, 'b n t 1 -> b n t'),
            'expected_dist': rearrange(expected_dist, 'b n t 1 -> b n t'),
        }

    def forward(
        self,
        features: torch.Tensor, # B T Hf Wf D (The raw output of V-JEPA)
        query_points: torch.Tensor,  # B N 3 (t, y, x in original pixel space)
        video_size: Tuple[int, int], # Original (H, W) for final scaling
    ) -> Mapping[str, torch.Tensor]:

        # TODO: Normalize input features?
        normalized_features = features 

        # Get Query Features via Interpolation
        interp_query_points = self.get_query_features(
            normalized_features,
            query_points
        ) # B N D

        results = self.estimate_trajectory(
            interp_query_points,
            normalized_features,
            query_points=query_points,
            video_size=video_size
        )
        return results