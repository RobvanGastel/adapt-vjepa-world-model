import os

import torch
import numpy as np
from PIL import Image
import gymnasium as gym

# Suppress pygame/ALSA audio warnings
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


class PendulumDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            num_episodes : int = 1000,
            seq_len : int = 24,
            input_size : tuple[int, int] = (64, 64),
            include_states : bool = True,
            include_actions : bool = True
        ):
        super().__init__()
        self.num_episodes = num_episodes
        self.seq_len = seq_len
        self.input_size = input_size
        self.include_states = include_states
        self.include_actions = include_actions
        
    def __getitem__(self, index):
        env = gym.make('Pendulum-v1', render_mode='rgb_array')
        state, _ = env.reset()
        
        frames, states, actions = [], [], []
        
        for _ in range(self.seq_len):
            action = np.array([0.0])  # Free swing (or sample random actions)
            next_state, _, terminated, truncated, _ = env.step(action)
            
            # Render and resize frame
            frame = env.render()
            if frame.shape[:2] != self.input_size:
                frame = np.array(Image.fromarray(frame).resize(
                    (self.input_size[1], self.input_size[0])
                ))
            
            frames.append(frame)
            states.append(next_state)
            actions.append(action)
            
            if terminated or truncated:
                break
        
        env.close()
        
        # Pad if episode ended early
        while len(frames) < self.seq_len:
            frames.append(frames[-1])
            states.append(states[-1])
            actions.append(actions[-1])
        
        # Convert to tensors
        video = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float() / 255
        
        result = {"video": video}
        if self.include_states:
            result["states"] = torch.from_numpy(np.array(states)).float()
        if self.include_actions:
            result["actions"] = torch.from_numpy(np.array(actions)).float()
        
        return result

    def __len__(self):
        return self.num_episodes
