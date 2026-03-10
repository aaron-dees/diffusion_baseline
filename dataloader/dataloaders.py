
import sys
sys.path.append('../')

import torch
from torch.utils.data import Dataset
import random
from utils.utilities import lowpass_moving_average


class LatentTextureDataset(Dataset):
    def __init__(self, latent_files, window_size):
        """
        latent_files: list of paths to .pt tensors, each [T_i, D]
        window_size: number of timesteps (e.g. 100 for 10s)
        """
        self.latent_files = latent_files
        self.window_size = window_size

        # Load all latents into memory (recommended if they fit)
        self.latents = [torch.load(f) for f in latent_files]

        # Filter out clips that are too short
        self.latents = [
            x for x in self.latents if x.shape[-1] >= window_size
        ]

    def __len__(self):
        # One sample per clip per epoch
        return len(self.latents)

    def __getitem__(self, idx):
        x = self.latents[idx]      # [1, D, T]
        x_slow = lowpass_moving_average(x.unsqueeze(0), kernel_size=9).squeeze(0)
        T = x.shape[-1]
        T = int(167*2) # set size to 10 for overfitting

        start = random.randint(0, T - self.window_size)
        window = x[:,start:start + self.window_size]
        slow_window = x_slow[:,start:start + self.window_size]


        return window, slow_window   # [window_size, D]
    
class LatentTextureDataset_new(Dataset):
    def __init__(self, latent_files, window_size, stride=5, precompute_slow=True, kernel_size=9):
        self.window_size = window_size
        self.stride = stride
        self.kernel_size = kernel_size

        self.latents = []
        self.slow_latents = []
        self.index = []  # list of (clip_idx, start)

        for f in latent_files:
            # x = torch.load(f)  # expect [D, T] or [T, D]
            x = torch.load(f, map_location='cpu').float()  # expect [D, T] or [T, D]
            if x.dim() == 2 and x.shape[0] != 64 and x.shape[1] == 64:
                # convert [T, D] -> [D, T]
                x = x.t()

            assert x.dim() == 2, f"Expected [D,T], got {x.shape}"
            D, T = x.shape
            if T < window_size:
                continue

            self.latents.append(x)

            if precompute_slow:
                x_slow = lowpass_moving_average(x.unsqueeze(0), kernel_size=kernel_size).squeeze(0)
                self.slow_latents.append(x_slow)
            else:
                self.slow_latents.append(None)

            clip_idx = len(self.latents) - 1
            for start in range(0, T - window_size + 1, stride):
                self.index.append((clip_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        clip_idx, start = self.index[idx]
        x = self.latents[clip_idx]         # [D, T]
        window = x[:, start:start+self.window_size]  # [D, W]

        if self.slow_latents[clip_idx] is None:
            x_slow = lowpass_moving_average(x.unsqueeze(0), kernel_size=self.kernel_size).squeeze(0)
        else:
            x_slow = self.slow_latents[clip_idx]

        slow_window = x_slow[:, start:start+self.window_size]
        return window, slow_window
