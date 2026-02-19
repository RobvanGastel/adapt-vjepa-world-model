import argparse
import logging

import torch
import imageio
import numpy as np
from PIL import Image
import gymnasium as gym

from video_jepa.planning.cem import CEM
from video_jepa.world_model import WorldModel


def create_goal_latent(model) -> torch.Tensor:
    resized_frames = [
        np.array(Image.fromarray(f).resize(model.input_size))
        for f in np.load("output/x_goal.npy")
    ]
    x = np.stack(resized_frames, axis=0) / 255.0
    x = torch.from_numpy(x[:2]).unsqueeze(0).moveaxis(-1, 1).float().cuda()

    # Goal state z_g
    z_g = model.encoder(x).reshape(1, -1, model.patch_h, model.patch_w, model.embed_dim)
    return z_g


def collect_context_rollout(
    env,
    input_size: tuple[int, int],
    num_steps: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs_stack = []
    act_stack = []

    for _ in range(num_steps):
        action = env.action_space.sample()
        env.step(action)

        obs = np.array(Image.fromarray(env.render()).resize(input_size))
        obs = torch.from_numpy(obs).moveaxis(-1, 0)

        obs_stack.append(obs)
        act_stack.append(torch.from_numpy(action))

    obs_stack = torch.stack(obs_stack, dim=1).unsqueeze(0).cuda()  # [1, C, T, H, W]
    act_stack = torch.stack(act_stack, dim=0).unsqueeze(0).cuda()  # [1, T, A]
    return obs_stack, act_stack


def train_world_model(config: argparse.Namespace):
    video_encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")

    model = WorldModel(
        num_hist=config.pred_n_frames,
        num_pred=config.pred_n_frames,
        video_encoder=video_encoder,
        input_size=config.crop_size,
        action_dim=1,
        action_embed_dim=config.action_embed_dim,
    )

    # TODO: Move to parameters
    planner = CEM(
        wm=model,
        horizon=config.horizon,
        num_samples=config.num_samples,
        topk=config.topk,
        opt_steps=config.opt_steps,
        var_scale=1.0,
    )

    # Load pretrained world model
    model.latent_predictor.load_state_dict(torch.load("output/latent_predictor.pt"))
    model.decoder.load_state_dict(torch.load("output/decoder.pt"))
    model.action_encoder.load_state_dict(torch.load("output/action_emb.pt"))
    model.cuda()

    # Goal to balance the pendulum straight
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []

    z_g = create_goal_latent(model)
    obs_stack, act_stack = collect_context_rollout(env, model.input_size, num_steps=6)

    for t in range(config.environment_steps):
        frame = env.render()
        frames.append(frame)

        # Update the observation context
        obs = np.array(Image.fromarray(frame).resize(model.input_size))
        obs = torch.from_numpy(obs).moveaxis(-1, 0).cuda().unsqueeze(0).unsqueeze(2)
        obs_stack = torch.cat([obs_stack[:, :, 1:], obs], dim=2)

        # Retrieve action from the planner
        action = planner.plan(obs_stack, act_stack, z_g)
        obs, _, done, _, _ = env.step(action.detach().cpu().numpy())

        # Update the action context
        action = action.unsqueeze(0).unsqueeze(-1)
        act_stack = torch.cat([act_stack[:, 1:], action], dim=1)

        if done:
            break

    # Store the performance of the model
    imageio.mimsave("assets/pendulum.gif", frames, fps=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Optimization samples of CEM",
    )
    parser.add_argument(
        "--opt_steps",
        type=int,
        default=25,
        help="Optimization steps of CEM",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=16,
        help="Topk variable of CEM",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=6,
        help="Optimization horizon of CEM",
    )
    parser.add_argument(
        "--crop_size",
        type=tuple,
        default=(128, 128),
        help="Size (H, W) of the video frames",
    )
    parser.add_argument(
        "--pred_n_frames",
        type=int,
        default=3,
        help="N frames to predict, and context",
    )
    parser.add_argument(
        "--action_embed_dim",
        type=int,
        default=96,
        help="The action embedding dimension size",
    )
    parser.add_argument(
        "environment_steps",
        type=int,
        default=200,
        help="Number of steps to take actions in the environment",
    )
    config = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    train_world_model(config)
