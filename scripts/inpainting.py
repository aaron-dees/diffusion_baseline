import sys
sys.path.append('../')

from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import os
import torch.nn as nn
import torch
import torchaudio
from torch.utils.tensorboard import SummaryWriter
import numpy as np


from scripts.config import Config
from dataloader.dataloaders import LatentTextureDataset_new

from music2latent import EncoderDecoder

from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler, VInpainter

def resume_from_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
        else:
            print("The checkpoint file does not contain the required keys. Training will start from scratch.")
            start_epoch = 0
    else:
        start_epoch = 0

    return start_epoch

def train(cfg, latent_files, save_dir):

    writer = SummaryWriter(log_dir="./runs/test_run")

    os.makedirs(save_dir, exist_ok=True)

    sample_len = cfg.window_size

    checkpoint_path = None

    codec = EncoderDecoder()

    # define model
    # base architecture lists (original)
    base_channels = [256, 256, 256, 256, 512, 512, 512, 768, 768]
    base_factors  = [1,   4,   4,   4,   2,   2,   2,   2,   2]
    base_items    = [1,   2,   2,   2,   2,   2,   2,   4,   4]
    base_attns    = [0,   0,   0,   0,   0,   1,   1,   1,   1]

    prod = 1
    max_idx = 0
    for i, f in enumerate(base_factors):
        prod *= f
        if prod <= sample_len:
            max_idx = i + 1
        else:
            break
    max_idx = max(1, max_idx)

    channels = base_channels[:max_idx]
    factors  = base_factors[:max_idx]
    items    = base_items[:max_idx]
    attentions = base_attns[:max_idx]

    print("Using UNet config:")
    print("  depth =", max_idx)
    print("  channels =", channels)
    print("  factors  =", factors, " (product =", int(np.prod(factors)), ")")
    print("  items    =", items)
    print("  attentions=", attentions)

    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=cfg.latent_dim,
        channels=channels,
        factors=factors,
        items=items,
        attentions=attentions,
        attention_heads=12,
        attention_features=64,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
    ).to(cfg.device)



    # define dataloader
    dataset = LatentTextureDataset_new(
        latent_files=latent_files,
        window_size=cfg.window_size,
        stride=cfg.dataset_stride,
        kernel_size=cfg.K   
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )

    train_data_loader = loader

    # net = UNetV0(
    #     dim=1,
    #     in_channels=2,
    #     channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],
    #     factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
    #     items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
    #     attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
    #     attention_heads=8,
    #     attention_features=64,
    # ).to(cfg.device)

    # --------------------------------------------------
    # 2. Load trained checkpoint
    # --------------------------------------------------

    checkpoint_path = "./diffusion_baseline_epoch500.pt"

    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # get trained UNet
    net = model.net

    inpainter = VInpainter(net=net)

    for step, batch in enumerate(train_data_loader):
        # print(len(batch))
        z_orig = batch[0][0].to(cfg.device).unsqueeze(0)  # [B, D, T]
        z_slow = batch[1][0].to(cfg.device).unsqueeze(0)  # [B, D
        # print("Batch Size: ", batch_tensor.shape)
        # print("Scale Size: ", scale_tensor.shape)
        B, C, T = z_orig.shape

        # --------------------------------------------------
        # create mask
        # True  = keep
        # False = regenerate
        # --------------------------------------------------

        mask = torch.ones_like(z_orig, dtype=torch.bool)

        # example: remove middle region
        start = int(T * 0.5)
        end = int(T * 1.0)

        print(f"Masking region from {start} to {end} (out of {T} timesteps)")

        mask[:, :, start:end] = False

        # --------------------------------------------------
        # run diffusion inpainting
        # --------------------------------------------------

        with torch.no_grad():
            latent_out = inpainter(
                source=z_orig,
                mask=mask,
                num_steps=50,
                num_resamples=4,
                show_progress=True,
            )

        # latent_out shape = [B, latent_dim, T]

        print("Original Latent Shape:", z_orig.shape)
        print("Inpainted Latent Shape:", latent_out.shape)

        orig_masked = codec.decode(z_orig.masked_fill(~mask, 0.0).cpu())
        orig = codec.decode(z_orig.cpu())
        inpainted = codec.decode(latent_out.cpu())

        torchaudio.save("original.wav", orig.cpu(), sample_rate=41000)
        torchaudio.save("original_masked.wav", orig_masked.cpu(), sample_rate=41000)
        torchaudio.save("inpainted.wav", inpainted.cpu(), sample_rate=41000)

        break





    print (img)

if __name__ == "__main__":


    data_dir = Path("/Users/adees/Code/multi-scale-rnn-vae/test_audio/latents")
    ext = ".pt"
    train_files = [str(p) for p in sorted(data_dir.glob(f"*{ext}"))]

    if not train_files:
        raise FileNotFoundError(f"No '{ext}' files found in {data_dir}")
    
    save_dir = "./checkpoints"

    cfg = Config()
    train(cfg, train_files, save_dir)
