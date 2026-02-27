import argparse
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from video_jepa.data import PendulumDataset
from video_jepa.world_model import WorldModel


def train_world_model(config: argparse.Namespace):
    logging.info(f"config: {config}")

    # Init the VJEPA2 model weights
    video_encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
    for param in video_encoder.parameters():
        param.requires_grad = False
    video_encoder.eval()

    seq_len = (config.hist_n_frames + config.pred_n_frames) * video_encoder.tubelet_size
    train_dataset = PendulumDataset(
        seq_len=seq_len,  # length of historic plus future frame predictions
        input_size=config.crop_size,
        include_states=False,
        include_actions=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=6,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        shuffle=True,
    )

    model = WorldModel(
        num_hist=config.hist_n_frames,
        num_pred=config.pred_n_frames,
        video_encoder=video_encoder,
        input_size=config.crop_size,
        action_dim=1,
        normalize_latents=False,
    ).cuda()
    # Normalization probably necessary to focus on structure and not magnitudes.

    optimizer = optim.AdamW(
        [
            {"params": model.latent_predictor.parameters(), "lr": config.pred_lr},
            {"params": model.action_encoder.parameters(), "lr": config.lr},
            {"params": model.decoder.parameters(), "lr": config.lr},
        ]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    for epoch in range(config.epochs):
        for batch in train_loader:
            video = batch["video"].moveaxis(1, 2).cuda()
            actions = batch["actions"].cuda()
            z_loss, decoder_loss = model(video, actions)

            optimizer.zero_grad()

            (z_loss + decoder_loss).backward()

            optimizer.step()
        scheduler.step()

        if epoch % 1 == 0:
            latent_param = model.latent_predictor.state_dict()
            torch.save(latent_param, f"output/latent_predictor.pt")
            torch.save(model.action_encoder.state_dict(), f"output/action_emb.pt")
            torch.save(model.decoder.state_dict(), f"output/decoder.pt")

            logging.info(
                f"Epoch: {epoch} - predictor loss: {z_loss.item()} - "
                f"decoder loss: {decoder_loss.item()}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=96,
        help="Finetuning batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--crop_size",
        type=tuple,
        default=(128, 128),
        help="Size (H, W) of the video frames",
    )
    parser.add_argument(
        "--hist_n_frames",
        type=int,
        default=3,
        help="N frames to predict (divided by tubelet size 2)",
    )
    parser.add_argument(
        "--pred_n_frames",
        type=int,
        default=1,
        help="N context frames (divided by tubelet size 2)",
    )
    parser.add_argument(
        "--pred_lr",
        type=float,
        default=5e-4,
        help="Latent predictor adamW learning rate",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Decoder, action embedding adamW learning rate",
    )
    config = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    train_world_model(config)
