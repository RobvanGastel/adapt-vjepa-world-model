import cv2
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

        # Latent predictor
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

        # Decoder, initial stage could be PCA
        self.latent_loss_weight = 0.25
        self.decoder = VQVAE(
            channel=384,
            n_embed=2048,
            emb_dim=1024,
            n_res_block=4,
            n_res_channel=128,
            quantize=False,
            frames_per_latent=2,
        )
        self.decoder_criterion = nn.MSELoss()
    
    def forward(self, x : torch.Tensor):
        B, C, T, H, W = x.shape
        z = self.encoder(x)
        
        # (B, frames, patches, embed_dim)
        patch_t = z.shape[1] // (self.patch_w * self.patch_h)
        z = z.reshape(B, patch_t, -1, z.shape[-1])

        # (b, num_hist or num_pred, num_patches, dim)
        z_src = z[:, : self.num_hist, :, :]
        z_tgt = z[:, self.num_pred :, :, :]

        # Latent Predictor, ViT
        # (b * frames * num_patches, dim)
        z_src = z_src.reshape(B, -1, z.shape[-1]).detach()
        z_pred = self.latent_predictor(z_src)

        # Decoder, VQVAE
        # TODO: Currently diff_pred is not useful.
        visual_pred, diff_pred = self.decoder(
            z_tgt,
            self.patch_h,
            self.patch_w,
            frames_per_latent=self.tubelet_size
        )

        # Loss
        # (B, C, num_pred, H, W)
        visual_tgt = x[:, :, self.num_pred * self.tubelet_size :, ...].moveaxis(1, 2)
        # Reshape to (B, tubelet, num_pred, C, H, W) average to fit
        visual_tgt = visual_tgt.view(B, self.num_pred * self.tubelet_size, C, H, W)
        visual_pred = visual_pred.view(B, self.num_pred * self.tubelet_size, C, H, W)

        # Decoder loss
        recon_loss = self.decoder_criterion(visual_pred, visual_tgt)
        decoder_loss = recon_loss + self.latent_loss_weight * diff_pred

        # Predictor loss
        z_pred = z_pred.reshape(B, self.num_pred, self.patch_w * self.patch_h, self.embed_dim)
        z_loss = self.predictor_criterion(z_pred, z_tgt)
        return z_loss, decoder_loss