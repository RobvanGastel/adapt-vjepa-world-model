import torch
import torch.nn as nn
from torch.nn import functional as F

from video_jepa.model import VQVAE, ViT, ProprioceptiveEmbedding


class WorldModel(nn.Module):
    def __init__(
        self,
        num_hist: int,
        num_pred: int,
        video_encoder: nn.Module,
        input_size: tuple[int, int],
        action_dim: int,
        normalize_latents: bool = False,
    ):
        super().__init__()
        # Experiment settings
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.action_dim = action_dim
        self.input_size = input_size
        self.normalize_latents = normalize_latents

        # Encoder, Video JEPA
        self.encoder = video_encoder
        self.embed_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_size
        self.tubelet_size = self.encoder.tubelet_size

        self.patch_h = input_size[0] // self.patch_size
        self.patch_w = input_size[1] // self.patch_size
        self.n_patches = self.patch_h * self.patch_w

        # Action encoder
        self.action_encoder = ProprioceptiveEmbedding(
            tubelet_size=self.tubelet_size,
            in_chans=self.action_dim,
            emb_dim=self.embed_dim,
        )

        # Latent predictor
        self.latent_predictor = ViT(
            num_patches=self.n_patches + 1,
            num_frames=self.num_hist,
            dim=self.embed_dim,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0,
            pool="mean",
        )
        self.predictor_criterion = nn.MSELoss()

        # Decoder, same structure as VQVAE
        self.decoder = VQVAE(
            channel=self.n_patches * self.num_pred,
            n_embed=2048,
            emb_dim=self.embed_dim,
            n_res_block=4,
            n_res_channel=128,
            quantize=False,
            frames_per_latent=self.tubelet_size,
        )
        self.decoder_criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor, action: torch.Tensor):
        B, C, T, H, W = x.shape

        with torch.no_grad():
            z = self.encoder(x)

        # Reshape source input
        z = z.reshape(B, z.shape[1] // (self.n_patches), -1, self.embed_dim)
        z_src = z[:, : self.num_hist, :, :]  # (B, latents, num_patches, dim)

        # Action encoder
        # (B, T / tubelet_size, embed_dim)
        z_act = self.action_encoder(action)
        z_act = z_act[:, : self.num_hist].unsqueeze(2)
        z_src = torch.cat([z_src, z_act], dim=2)  # Add action token to each frame

        # Latent Predictor
        z_pred = self.latent_predictor(z_src.reshape(B, -1, self.embed_dim))

        # Decoder, decode latents
        z_pred = z_pred.reshape(B, self.num_hist, -1, self.embed_dim)

        # Only take the last frame! It is autoregressively generated.
        # Important to check!
        z_pred = z_pred[:, -1:]

        visual_pred, _ = self.decoder(
            z_pred[:, :, :-1, :].detach(),  # Remove action token again
            self.patch_h,
            self.patch_w,
            frames_per_latent=self.tubelet_size,
        )

        # Decoder loss
        visual_pred = visual_pred.reshape(B, -1, C, H, W)
        visual_tgt = x.moveaxis(1, 2)[:, self.num_hist * self.tubelet_size :]
        decoder_loss = self.decoder_criterion(visual_pred, visual_tgt)

        # Predictor loss
        z_tgt = z[:, self.num_hist : self.num_hist + self.num_pred]
        z_pred = z_pred[:, :, :-1, :]  # Remove the last action token

        # Normalize latents should give better performance?
        if self.normalize_latents:
            z_pred = F.normalize(z_pred, p=2, dim=-1)
            z_tgt = F.normalize(z_tgt, p=2, dim=-1)
        z_loss = self.predictor_criterion(z_pred, z_tgt.detach())
        return z_loss, decoder_loss

    @torch.inference_mode()
    def rollout(
        self,
        src_obs: torch.Tensor,
        src_act: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        src_obs : [B, C, T, H, W]
        src_act : [B, T, A]
        actions : [B, H, A], CEM samples.
        """
        B, H, _ = actions.shape
        T = src_obs.shape[2]

        # Expand context to fit sample size
        src_obs = src_obs.expand(B, -1, -1, -1, -1)
        src_act = src_act.expand(B, -1, -1)

        # Encode the observations
        z = self.encoder(src_obs.float())
        vid_feats = z.reshape(B, T // self.tubelet_size, self.n_patches, self.embed_dim)

        # Combine the planned CEM actions
        stacked_act = torch.cat([src_act, actions], dim=1)

        for h in range(H // self.tubelet_size):
            # Keep adding video_features always take the last 3 frames
            z_ctx = vid_feats[:, -self.num_hist :]
            curr_ctxt_len = z_ctx.shape[1]

            # Select the actions for planning
            end_frame_idx = (self.num_hist + h) * self.tubelet_size
            start_frame_idx = end_frame_idx - (curr_ctxt_len * self.tubelet_size)
            act_slice = stacked_act[:, start_frame_idx:end_frame_idx]

            z_act = self.action_encoder(act_slice).unsqueeze(2)
            z_src = torch.cat([z_ctx, z_act], dim=2)

            # Predict future steps
            z_pred = self.latent_predictor(z_src.reshape(B, -1, self.embed_dim))
            z_pred = z_pred.reshape(B, self.num_hist, -1, self.embed_dim)

            # last latent, removing action token
            vid_feats = torch.cat([vid_feats, z_pred[:, -1:, :-1, :]], dim=1)

        # Remove historic predictions
        vid_feats = vid_feats[:, T // self.tubelet_size :]
        vid_feats = vid_feats.reshape(B, -1, self.patch_h, self.patch_w, self.embed_dim)
        return vid_feats
