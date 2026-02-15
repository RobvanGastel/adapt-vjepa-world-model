import torch
import torch.nn as nn

class CEM:
    def __init__(
            self,
            wm : nn.Module,
            horizon : int,
            num_samples : int,
            topk : int,
            opt_steps : int,
            var_scale : float = 2.0
        ):
        self.wm = wm
        self.topk = topk
        self.horizon = horizon
        self.opt_steps = opt_steps
        self.var_scale = var_scale
        self.num_samples = num_samples
        
        self.action_dim = wm.action_dim
        self.device = next(wm.parameters()).device

    @torch.no_grad()
    def plan(self, src_obs, src_act, z_g):
        # TODO: Pass observations instead of z_g

        # Shape: (H, action_dim)
        mu = torch.zeros((self.horizon, self.action_dim), device=self.device)
        sigma = torch.ones((self.horizon, self.action_dim), device=self.device) * self.var_scale

        for _ in range(self.opt_steps):
            # Sample action sequences
            eps = torch.randn((self.num_samples, self.horizon, self.action_dim), device=self.device)
            actions = mu + sigma * eps
            actions = torch.clamp(actions, min=-2.0, max=2.0) # TODO: Remove?
            z_obs = self.wm.rollout(src_obs, src_act, actions)

            z_y_hat = z_obs[:, -1]
            z_y = z_g[:, -1]
        
            dists = torch.mean((z_y_hat - z_y)**2, dim=(1, 2, 3))
            values, indices = torch.topk(dists, self.topk, largest=False)
            elites = actions[indices]
        
            mu = elites.mean(dim=0)
            sigma = elites.std(dim=0) + 1e-6
        return mu
