# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Pytorch model utilities."""
from typing import Sequence

import torch
import torch.nn.functional as F

def map_coordinates_3d(
    feats: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
  """Maps 3D coordinates to corresponding features using bilinear interpolation.

  Args:
    feats: A 5D tensor of features with shape (B, W, H, D, C), where B is batch
      size, W is width, H is height, D is depth, and C is the number of
      channels.
    coordinates: A 3D tensor of coordinates with shape (B, N, 3), where N is the
      number of coordinates and the last dimension represents (W, H, D)
      coordinates.

  Returns:
    The mapped features tensor.
  """
  x = feats.permute(0, 4, 1, 2, 3)
  y = coordinates[:, :, None, None, :].float().clone()
  y[..., 0] = y[..., 0] + 0.5
  y = 2 * (y / torch.tensor(x.shape[2:], device=y.device)) - 1
  y = torch.flip(y, dims=(-1,))
  out = (
      F.grid_sample(
          x, y, mode='bilinear', align_corners=False, padding_mode='border'
      )
      .squeeze(dim=(3, 4))
      .permute(0, 2, 1)
  )
  return out

def convert_grid_coordinates(coords, input_grid_size, output_grid_size, coordinate_format='xy'):
    input_grid_size = torch.as_tensor(input_grid_size, device=coords.device, dtype=coords.dtype)
    output_grid_size = torch.as_tensor(output_grid_size, device=coords.device, dtype=coords.dtype)

    if coordinate_format == 'xy' and len(input_grid_size) != 2:
        raise ValueError("Expected 2-D coordinates for 'xy'")
    if coordinate_format == 'tyx' and len(input_grid_size) != 3:
        raise ValueError("Expected 3-D coordinates for 'tyx'")

    if coordinate_format == 'tyx' and input_grid_size[0] != output_grid_size[0]:
        raise ValueError("Time resizing not supported.")

    scale = (output_grid_size / input_grid_size).view(*([1] * (coords.ndim - 1)), -1)
    return coords * scale


def soft_argmax_heatmap_batched(softmax_val, threshold=5):
  """Test if two image resolutions are the same."""
  b, h, w, d1, d2 = softmax_val.shape
  y, x = torch.meshgrid(
      torch.arange(d1, device=softmax_val.device),
      torch.arange(d2, device=softmax_val.device),
      indexing='ij',
  )
  coords = torch.stack([x + 0.5, y + 0.5], dim=-1).to(softmax_val.device)
  softmax_val_flat = softmax_val.reshape(b, h, w, -1)
  argmax_pos = torch.argmax(softmax_val_flat, dim=-1)

  pos = coords.reshape(-1, 2)[argmax_pos]
  valid = (
      torch.sum(
          torch.square(
              coords[None, None, None, :, :, :] - pos[:, :, :, None, None, :]
          ),
          dim=-1,
          keepdims=True,
      )
      < threshold**2
  )

  weighted_sum = torch.sum(
      coords[None, None, None, :, :, :]
      * valid
      * softmax_val[:, :, :, :, :, None],
      dim=(3, 4),
  )
  sum_of_weights = torch.maximum(
      torch.sum(valid * softmax_val[:, :, :, :, :, None], dim=(3, 4)),
      torch.tensor(1e-12, device=softmax_val.device),
  )
  return weighted_sum / sum_of_weights


def heatmaps_to_points(
    all_pairs_softmax,
    image_shape,
    threshold=5,
    query_points=None,
):
  """Convert heatmaps to points using soft argmax."""

  out_points = soft_argmax_heatmap_batched(all_pairs_softmax, threshold)
  feature_grid_shape = all_pairs_softmax.shape[1:]
  # Note: out_points is now [x, y]; we need to divide by [width, height].
  # image_shape[3] is width and image_shape[2] is height.
  out_points = convert_grid_coordinates(
      out_points,
      feature_grid_shape[3:1:-1],
      image_shape[3:1:-1],
  )
  assert feature_grid_shape[1] == image_shape[1]
  if query_points is not None:
    # The [..., 0:1] is because we only care about the frame index.
    query_frame = convert_grid_coordinates(
        query_points.detach(),
        image_shape[1:4],
        feature_grid_shape[1:4],
        coordinate_format='tyx',
    )[..., 0:1]

    query_frame = torch.round(query_frame)
    frame_indices = torch.arange(image_shape[1], device=query_frame.device)[
        None, None, :
    ]
    is_query_point = query_frame == frame_indices

    is_query_point = is_query_point[:, :, :, None]
    out_points = (
        out_points * ~is_query_point
        + torch.flip(query_points[:, :, None], dims=(-1,))[..., 0:2]
        * is_query_point
    )

  return out_points
