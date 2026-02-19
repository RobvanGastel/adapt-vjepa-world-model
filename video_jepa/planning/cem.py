import torch
import torch.nn as nn


class CEM:
    def __init__(
        self,
        wm: nn.Module,
        horizon: int,
        num_samples: int,
        topk: int,
        opt_steps: int,
        var_scale: float = 2.0,
    ):
        self.wm = wm
        self.topk = topk
        self.horizon = horizon
        self.opt_steps = opt_steps
        self.var_scale = var_scale
        self.num_samples = num_samples

        self.action_dim = wm.action_dim
        self.device = next(wm.parameters()).device

    @torch.inference_mode()
    def plan(
        self,
        src_obs: torch.Tensor,
        src_act: torch.Tensor,
        z_g: torch.Tensor,
        alpha: float = 1.0,
    ):

        # Shape: (H, action_dim)
        mu = torch.zeros((self.horizon, self.action_dim), device=self.device)
        sigma = (
            torch.ones((self.horizon, self.action_dim), device=self.device)
            * self.var_scale
        )

        for _ in range(self.opt_steps):
            # Sample action sequences
            eps = torch.randn(
                (self.num_samples, self.horizon, self.action_dim), device=self.device
            )
            actions = mu + sigma * eps

            actions[0] = mu

            z_obs = self.wm.rollout(src_obs, src_act, actions)

            z_y_hat = z_obs[:, -1]
            z_y = z_g[:, -1]

            dists = torch.mean((z_y_hat - z_y) ** 2, dim=(1, 2, 3))
            _, indices = torch.topk(dists, self.topk, largest=False)
            elites = actions[indices]

            new_mu = elites.mean(dim=0)
            new_sigma = elites.std(dim=0)

            mu = alpha * new_mu + (1 - alpha) * mu
            sigma = alpha * new_sigma + (1 - alpha) * sigma + 1e-6
        return mu[0]
