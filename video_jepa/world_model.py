import torch
import torch.nn as nn

from video_jepa.latent_decoder import VQVAE
from video_jepa.latent_predictor import ViT


class WorldModel(nn.Module):
    def __init__(
            self,
            num_hist : int,
            num_pred : int,
            video_encoder : nn.Module,
            input_size : tuple[int, int],
    ):
        super().__init__()
        # Experiment settings
        self.num_hist = num_hist
        self.num_pred = num_pred

        # Encoder, Video JEPA 
        self.encoder = video_encoder
        self.embed_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_size
        self.tubelet_size = self.encoder.tubelet_size

        self.patch_h = input_size[0] // self.patch_size 
        self.patch_w = input_size[1] // self.patch_size

        # TODO: Frames divided by tubelet size?
        # TODO: Input size to calculate # of patches
        self.latent_predictor = ViT(
            num_patches=self.patch_h * self.patch_w,
            num_frames=self.num_hist,
            dim=self.embed_dim,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0,
            pool="mean"
        )
        self.predictor_criterion = nn.MSELoss()

        # Initial stage could be PCA
        self.latent_loss_weight = 0.25
        self.decoder = VQVAE(
            in_channel=1024,
            channel=384,
            n_embed=1024,
            emb_dim=1024,
            n_res_block=4,
            n_res_channel=128,
            quantize=False
        )
        self.decoder_criterion = nn.MSELoss()
    
    def forward(self, x):
        print(x.shape)
        z = self.encoder(x)
        
        # (B, frames, patches, embed_dim)
        patch_t = z.shape[1] // (self.patch_w * self.patch_h)
        z = z.reshape(z.shape[0], patch_t, -1, z.shape[-1])

        # (b, num_hist / num_pred, num_patches, dim)
        z_src = z[:, : self.num_hist, :, :]
        z_tgt = z[:, self.num_pred :, :, :]

        # Latent Predictor
        z_src = z_src.reshape(z.shape[0], -1, z.shape[-1]).detach()
        z_pred = self.latent_predictor(z_src)

        # Decoder, VQVAE, TODO: Initial test with PCA?
        visual_pred, diff_pred = self.decoder(
            z_tgt,
            self.patch_h,
            self.patch_w
        )

        # Loss
        # (B, num_pred, 3, *input_size), TODO: Include tubelet size
        visual_tgt = x[:, self.num_pred :, ...]
        
        # Decoder loss
        print(visual_tgt.shape, visual_pred.shape)
        recon_loss = self.decoder_criterion(visual_pred, visual_tgt)
        decoder_loss = recon_loss + self.latent_loss_weight * diff_pred

        # Predictor loss
        z_pred = z_pred.reshape(z_tgt.shape[0], self.num_pred, self.patch_w * self.patch_h, self.embed_dim)

        torch.save(z_pred[0][0], "z_pred.pt")
        torch.save(z_tgt[0][0], "z_tgt.pt")

        z_loss = self.predictor_criterion(z_pred, z_tgt)

        return z_loss, decoder_loss