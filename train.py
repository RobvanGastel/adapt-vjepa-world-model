import argparse
import logging

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from video_jepa.data import KubricMovifDataset
from video_jepa.tracking_head import TrackingHead
from video_jepa.loss import tapnet_loss


def train_point_tracker(config: argparse.Namespace):
    encoder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
    decoder = TrackingHead()

    for param in encoder.parameters():
        param.requires_grad = False
        
    decoder.cuda()
    encoder.cuda()
    
    optimizer = optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=1e-3)

    train_dataset = KubricMovifDataset("/data/movi_f/")
    train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=4,
            persistent_workers=True,
            shuffle=True,
    )

    for epoch in range(config.epochs):
        for batch in train_loader:
            # TODO: Better batch output
            video = batch[0].video.unsqueeze(0).moveaxis(1, 2).cuda()
            trajectory = batch[0].trajectory.unsqueeze(0).reshape(1, -1, 3).cuda()
            query_points = batch[0].query_points.unsqueeze(0).reshape(1, -1, 3).cuda()
            occlusion = batch[0].occluded.unsqueeze(0)

            patch_h = video.shape[3] // encoder.patch_size
            patch_w = video.shape[4] // encoder.patch_size
            patch_t = video.shape[2] // encoder.tubelet_size

            vjepa_features = encoder(video)
            # TODO: Proper batching, fix reshaping automatically
            vjepa_features = vjepa_features.reshape(
                vjepa_features.shape[0],
                patch_t, 
                patch_h,
                patch_w,
                encoder.embed_dim,
            )
            output = decoder(vjepa_features, query_points, video.shape[-2:])  
            
            optimizer.zero_grad()
            loss_huber, loss_occ, loss_prob = tapnet_loss(
            output['tracks'],
            output['occlusion'],
            query_points,
            ~occlusion,
            video.shape,  # pytype: disable=attribute-error  # numpy-scalars
            expected_dist=output['expected_dist'] if 'expected_dist' in output else None,
            position_loss_weight=0.05,
            expected_dist_thresh=6.0,
            )
            loss = loss_huber + loss_occ + loss_prob

            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            decoder.save_parameters(f"output/tracking_e{epoch}.pt")
            logging.info(
                f"{epoch}: loss {loss}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Finetuning batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    config = parser.parse_args()

    train_point_tracker(config)