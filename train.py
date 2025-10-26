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

    train_dataset = KubricMovifDataset("./data")
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
            B = batch["video"].size(0)
            video = batch["video"].moveaxis(1, 2).cuda()
            trajectory = batch["trajectory"].reshape(B, -1, 3).cuda()
            query_points = batch["query_points"].reshape(B, -1, 3).cuda()
            occlusion = batch["occluded"]

            # torch.Size([2, 3, 24, 384, 512]) torch.Size([1, 24576, 3]) torch.Size([1, 36864, 3]) torch.Size([2, 768, 24])
            # print(video.shape, trajectory.shape, query_points.shape, occlusion.shape)
            patch_h = video.shape[3] // encoder.patch_size
            patch_w = video.shape[4] // encoder.patch_size
            patch_t = video.shape[2] // encoder.tubelet_size


            features = encoder(video)
            # TODO: Proper batching, fix reshaping automatically
            features = features.reshape(
                features.shape[0],
                patch_t, 
                patch_h,
                patch_w,
                encoder.embed_dim,
            )

            output = decoder(features, query_points, video.shape[-2:])  
            
            optimizer.zero_grad()
            loss_huber, loss_occ, loss_prob = tapnet_loss(
                output['tracks'],
                output['occlusion'],
                trajectory,
                occlusion,
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
        default=1,
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