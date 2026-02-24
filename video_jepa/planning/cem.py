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

        self.act_dim = wm.action_dim
        self.device = next(wm.parameters()).device

    @torch.inference_mode()
    def plan(
        self,
        src_obs: torch.Tensor,
        src_act: torch.Tensor,
        z_g: torch.Tensor,
        alpha: float = 1.0,
        return_features: bool = False,
    ):

        # Shape: (H, action_dim)
        mu = torch.zeros((self.horizon, self.act_dim), device=self.device)
        sigma = (
            torch.ones((self.horizon, self.act_dim), device=self.device)
            * self.var_scale
        )

        for _ in range(self.opt_steps):
            # Sample action sequences
            eps = torch.randn(
                (self.num_samples, self.horizon, self.act_dim), device=self.device
            )
            raw_actions = mu + sigma * eps
            raw_actions[0] = mu
            actions = torch.tanh(raw_actions) * 2.0

            z_y_hat = self.wm.rollout(src_obs, src_act, actions)

            dists = torch.mean((z_y_hat[:, -1] - z_g[:, -1]) ** 2, dim=(1, 2, 3))
            _, indices = torch.topk(dists, self.topk, largest=False)
            elites = raw_actions[indices]

            new_mu = elites.mean(dim=0)
            new_sigma = elites.std(dim=0)

            mu = alpha * new_mu + (1 - alpha) * mu
            sigma = alpha * new_sigma + (1 - alpha) * sigma + 1e-6

        action = torch.tanh(mu[0]) * 2.0
        if return_features:
            return action, z_y_hat[0]
        else:
            return action
