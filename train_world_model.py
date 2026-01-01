import argparse
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from video_jepa.data import VideoDataset
from video_jepa.world_model import WorldModel


def train_world_model(config: argparse.Namespace):
    # Init the VJEPA2 model weights
    video_encoder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
    for param in video_encoder.parameters():
        param.requires_grad = False

    train_dataset = VideoDataset(
        config.data_path,
        crop_size=config.crop_size,
        seq_len=config.seq_len
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        persistent_workers=True,
        shuffle=True,
    )

    model = WorldModel(
        num_hist=config.pred_n_frames,
        num_pred=config.pred_n_frames,
        video_encoder=video_encoder,
        input_size=config.crop_size
    ).cuda()

    predictor_opt = optim.AdamW(model.latent_predictor.parameters(), lr=1e-3)
    # encoder_opt  = optim.AdamW(model.encoder.parameters(), lr=1e-5)
    decoder_opt = optim.AdamW(model.decoder.parameters(), lr=3e-4)

    for epoch in range(config.epochs):
        for batch in train_loader:
            video = batch["video"].moveaxis(1, 2).cuda()
            z_loss, decoder_loss = model(video)

            predictor_opt.zero_grad()
            decoder_opt.zero_grad()
            # encoder_opt.zero_grad()

            (z_loss + decoder_loss).backward()

            # encoder_opt.step()
            predictor_opt.step()
            decoder_opt.step()

        if epoch % 5 == 0:
            torch.save(model.latent_predictor.state_dict(), f"output/latent_predictor.pt")
            torch.save(model.decoder.state_dict(), f"output/decoder.pt")
            logging.info(
                f"Epoch: {epoch} - predictor loss: {z_loss.item()} - "
                f"decoder loss: {decoder_loss.item()}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Dataset path"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="Finetuning batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--crop_size",
        type=tuple,
        default=(384, 384),
        help="Size (H, W) of the video frames"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=12,
        help="Sequence length T of the video"
    )
    parser.add_argument(
        "--pred_n_frames",
        type=int, 
        default=3,
        help="N frames to predict, and context"
    )
    config = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    train_world_model(config)