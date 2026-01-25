import numpy as np
import torch.nn as nn


class ProprioceptiveEmbedding(nn.Module):
    def __init__(
        self,
        tubelet_size=1,
        in_chans=8,
        emb_dim=384,
    ):
        super().__init__()

        # Map input to predictor dimension
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim

        self.patch_embed = nn.Conv1d(
            in_chans,
            emb_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size,
        )

    def forward(self, x):
        # x: proprioceptive vectors of shape [B T D]
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return x