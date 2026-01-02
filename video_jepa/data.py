# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging

import cv2
import torch
import numpy as np


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        seq_len=24,
        crop_size=(384, 512),
        max_frame_stride=5,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.crop_size = crop_size
        self.data_root = data_root
        self.max_frame_stride = max_frame_stride

        self.seq_names = [
            f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))
        ]
        logging.info("found %d unique videos in %s" % (len(self.seq_names), self.data_root))

    def __getitem__(self, index):
        seq_name = self.seq_names[index]
        rgb_path = os.path.join(self.data_root, seq_name, "frames")

        img_paths = sorted(os.listdir(rgb_path))
        T = len(img_paths)

        # Sample the frame rate
        max_allowed_stride = min(
            self.max_frame_stride,
            max(1, (T - 1) // (self.seq_len - 1))
        )
        stride = np.random.randint(1, max_allowed_stride + 1)
        
        # Get a series of images with given sequence length
        max_start = T - stride * (self.seq_len - 1)
        start = np.random.randint(0, max_start)
        img_paths = img_paths[start : start + stride * self.seq_len : stride]

        # Load the video frames
        rgbs = [
            cv2.imread(os.path.join(rgb_path, p))
            for p in img_paths
        ]

        rgbs = np.stack(rgbs)[: self.seq_len]  # optional truncation
        if self.crop_size is not None:
            rgbs = self.crop_rgbs(rgbs, self.crop_size)

        rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float() / 255

        return {
            "video": rgbs,
            "name": seq_name,
        }

    def crop_rgbs(self, rgbs, crop_size):
        H, W = rgbs[0].shape[:2]

        y0 = 0 if crop_size[0] >= H else (H - crop_size[0]) // 2
        x0 = 0 if crop_size[1] >= W else np.random.randint(0, W - crop_size[1])

        rgbs = [rgb[y0:y0 + crop_size[0], x0:x0 + crop_size[1]] for rgb in rgbs]
        return np.stack(rgbs)

    def __len__(self):
        return len(self.seq_names)