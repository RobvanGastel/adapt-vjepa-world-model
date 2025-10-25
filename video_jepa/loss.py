import torch
from torch.nn import functional as F
from video_jepa.transformations import convert_grid_coordinates 


def huber_loss(tracks, target_points, occluded, delta=4.0, reduction_axes=(1, 2)):
    """Huber loss for point trajectories."""
    error = tracks - target_points
    distsqr = torch.sum(error ** 2, dim=-1)
    dist = torch.sqrt(distsqr + 1e-6)  # add eps to prevent nan
    loss_huber = torch.where(dist < delta, distsqr / 2, delta * (torch.abs(dist) - delta / 2))
    loss_huber = loss_huber * (1.0 - occluded.float())

    if reduction_axes:
        loss_huber = torch.mean(loss_huber, dim=reduction_axes)

    return loss_huber

def prob_loss(tracks, expd, target_points, occluded, expected_dist_thresh=8.0, reduction_axes=(1, 2)):
    """Loss for classifying if a point is within pixel threshold of its target."""
    err = torch.sum((tracks - target_points) ** 2, dim=-1)
    invalid = (err > expected_dist_thresh ** 2).float()
    logprob = F.binary_cross_entropy_with_logits(expd, invalid, reduction='none')
    logprob = logprob * (1.0 - occluded.float())
    
    if reduction_axes:
        logprob = torch.mean(logprob, dim=reduction_axes)
        
    return logprob
    
def tapnet_loss(points, occlusion, target_points, target_occ, shape, mask=None, expected_dist=None,
                position_loss_weight=0.05, expected_dist_thresh=6.0, huber_loss_delta=4.0, 
                rebalance_factor=None, occlusion_loss_mask=None):
    """TAPNet loss."""
    
    if mask is None:
        mask = torch.tensor(1.0)

    points = convert_grid_coordinates(points, shape[3:1:-1], (256, 256), coordinate_format='xy')
    target_points = convert_grid_coordinates(target_points, shape[3:1:-1], (256, 256), coordinate_format='xy')

    loss_huber = huber_loss(points, target_points, target_occ, delta=huber_loss_delta, reduction_axes=None) * mask
    loss_huber = torch.mean(loss_huber) * position_loss_weight

    if expected_dist is None:
        loss_prob = torch.tensor(0.0)
    else:
        loss_prob = prob_loss(points.detach(), expected_dist, target_points, target_occ, expected_dist_thresh, reduction_axes=None) * mask
        loss_prob = torch.mean(loss_prob)

    target_occ = target_occ.to(dtype=occlusion.dtype)
    loss_occ = F.binary_cross_entropy_with_logits(occlusion, target_occ, reduction='none') * mask

    if rebalance_factor is not None:
        loss_occ = loss_occ * ((1 + rebalance_factor) - rebalance_factor * target_occ)
        
    if occlusion_loss_mask is not None:
        loss_occ = loss_occ * occlusion_loss_mask

    loss_occ = torch.mean(loss_occ)

    return loss_huber, loss_occ, loss_prob