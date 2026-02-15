import torch
import torch.nn as nn
from einops import repeat

from video_jepa.model import VQVAE, ViT, ProprioceptiveEmbedding 


class WorldModel(nn.Module):
    def __init__(
            self,
            num_hist : int,
            num_pred : int,
            video_encoder : nn.Module,
            input_size : tuple[int, int],
            action_dim : int,
            action_embed_dim : int
    ):
        super().__init__()
        # Experiment settings
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.action_dim = action_dim
        self.input_size = input_size
        self.action_embed_dim = action_embed_dim

        # Encoder, Video JEPA 
        self.encoder = video_encoder
        self.embed_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_size
        self.tubelet_size = self.encoder.tubelet_size

        self.patch_h = input_size[0] // self.patch_size 
        self.patch_w = input_size[1] // self.patch_size

        # Action encoder
        self.action_encoder = ProprioceptiveEmbedding(
            tubelet_size=self.tubelet_size,
            in_chans=self.action_dim,
            emb_dim=self.action_embed_dim
        )

        # Latent predictor
        self.latent_predictor = ViT(
            num_patches=self.patch_h * self.patch_w,
            num_frames=self.num_hist,
            dim=self.embed_dim + self.action_embed_dim,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0,
            pool="mean"
        )
        self.predictor_criterion = nn.MSELoss()

        # Decoder, same structure as VQVAE
        self.latent_loss_weight = 0.25
        self.decoder = VQVAE(
            channel=384,
            n_embed=2048,
            emb_dim=1024,
            n_res_block=4,
            n_res_channel=128,
            quantize=False,
            frames_per_latent=self.tubelet_size,
        )
        self.decoder_criterion = nn.MSELoss()
    
    def forward(self, x : torch.Tensor, action : torch.Tensor):
        B, C, T, H, W = x.shape
        z = self.encoder(x)

        # (B, frames, patches, embed_dim)
        patch_t = z.shape[1] // (self.patch_w * self.patch_h)
        z = z.reshape(B, patch_t, -1, z.shape[-1])

        # (B, num_hist or num_pred, num_patches, dim)
        z_src = z[:, : self.num_hist, :, :]
        z_tgt = z[:, self.num_pred :, :, :]

        # Action encoder
        # (B, frames, action_embed_dim)
        z_act = self.action_encoder(action)
        z_act = z_act[:, : self.num_hist].unsqueeze(2)
        z_act_tiled = repeat(
            z_act,
            "b t 1 a -> b t f a",
            f=z_tgt.shape[2]
        )

        # Latent Predictor, ViT
        # (B, num_pred, num_patches, 2)
        z_src = torch.cat([z_src, z_act_tiled], dim=3)

        # (B * frames * num_patches, dim)
        z_src = z_src.reshape(B, -1, z.shape[-1] + self.action_embed_dim).detach()
        z_pred = self.latent_predictor(z_src)

        # Decoder
        # TODO: Currently diff_pred is not useful.
        visual_pred, _ = self.decoder(
            z.detach(),
            self.patch_h,
            self.patch_w,
            frames_per_latent=self.tubelet_size
        )

        # Loss
        # Reshape to (B, tubelet, T, C, H, W) average to fit
        visual_pred = visual_pred.view(B, -1, C, H, W)

        # Decoder loss
        decoder_loss = self.decoder_criterion(visual_pred, x.moveaxis(1, 2))

        # Predictor loss
        z_pred = z_pred.reshape(
            B,
            self.num_pred,
            self.patch_w * self.patch_h,
            self.embed_dim + self.action_embed_dim
        )
        z_pred = z_pred[..., :-self.action_embed_dim]
        z_loss = self.predictor_criterion(z_pred, z_tgt)
        return z_loss, decoder_loss
    
    def encode_obs(self, x : torch.Tensor):
        B, C, T, H, W = x.shape
        z = self.encoder(x)
        return z.reshape(B, -1, self.patch_h * self.patch_w, self.embed_dim)

    @torch.no_grad()
    def rollout(self, src_obs, src_act, actions):
        """
        src_obs : [B, C, K, H, W]
        src_act : [B, K, A]
        actions : [B, H, A], CEM samples.
        """
        B = actions.shape[0]
        H = actions.shape[1]
        K = src_obs.shape[2]
        P = self.patch_h * self.patch_w
        
        # Expand context
        src_obs = src_obs.expand(B, -1, -1, -1, -1) # [B, C, K, H, W]
        src_act = src_act.expand(B, -1, -1) # [B, K, A]

        # Encode the observations
        z = self.encoder(src_obs.float())
        vid_feats = z.reshape(B, K // self.tubelet_size, P, self.embed_dim)

        # Combine the planned CEM actions
        stacked_act = torch.cat([src_act, actions], dim=1)
        
        for h in range(H // self.tubelet_size):
            # Keep adding video_features always take the last 3 frames
            z_ctx = vid_feats[:, -self.num_hist:]
            curr_ctxt_len = z_ctx.shape[1]

            # Select the actions for planning
            end_frame_idx = (self.num_hist + h) * 2
            start_frame_idx = end_frame_idx - (curr_ctxt_len * 2)
            act_slice = stacked_act[:, start_frame_idx : end_frame_idx]
            z_act = self.action_encoder(act_slice).unsqueeze(2).repeat(1, 1, P, 1)

            combined = torch.cat([z_ctx, z_act], dim=-1)
            combined = combined.reshape(B, -1, self.embed_dim + self.action_embed_dim)

            # Predict future steps
            z_next = self.latent_predictor(combined)
            # TODO: Or [:, -1:], double check latent prediction order
            z_next = z_next.reshape(B, 3, P, -1)[:, 0:1]
            z_next = z_next[..., :-self.action_embed_dim]

            vid_feats = torch.cat([vid_feats, z_next], dim=1)

        vid_feats = vid_feats.reshape(B, -1, self.patch_h, self.patch_w, self.embed_dim) 
        return vid_feats