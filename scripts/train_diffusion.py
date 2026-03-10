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

from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

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

    if checkpoint_path != None:
        print(f"Resuming training from: {checkpoint_path}\n")

    if not cfg.diffusion_finetune:
        ##### TRAIN FROM SCRATCH
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.diffusion_scheduler_steps, gamma=0.99)
        start_epoch = resume_from_checkpoint(checkpoint_path, model, optimizer, scheduler)
    else:
        #### FINETUNE
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5) # Change the learning rate
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.diffusion_scheduler_steps, eta_min=1e-6) # Replace the StepLR scheduler with the CosineAnnealingLR scheduler
        start_epoch = resume_from_checkpoint(checkpoint_path, model, optimizer, scheduler)

    accumulation_steps = cfg.diffusion_accumulation_steps 

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

    for i in range(start_epoch, cfg.diffusion_epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_data_loader):
            # print(len(batch))
            z_orig = batch[0].to(cfg.device)  # [B, D, T]
            z_slow = batch[1].to(cfg.device)  # [B, D
            # print("Batch Size: ", batch_tensor.shape)
            # print("Scale Size: ", scale_tensor.shape)

            loss = model(z_orig)

            train_loss += loss.item()

            if (step + 1) % accumulation_steps == 0:
                loss = loss / accumulation_steps
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        train_loss /= len(train_data_loader)
        print(f"Epoch {i+1}, train loss: {train_loss}")
        writer.add_scalar("Loss/Train", train_loss, i+1)

        # random.shuffle(train_dataset.latent_files)

        with torch.no_grad():
            model.eval()

            val_loss = 0
            for batch in train_data_loader:
                batch_tensor = batch[0].to(cfg.device)
                scale_tensor = batch[1].to(cfg.device)

                loss = model(batch_tensor)

                val_loss += loss.item()

            val_loss /= len(train_data_loader)
            print(f"Epoch {i+1}, validation loss: {val_loss}")
            writer.add_scalar("Loss/Val", val_loss, i+1)
            noise = torch.randn(1, cfg.latent_dim, sample_len).to(cfg.device)
            noise = noise * cfg.diffusion_temperature
            diff = model.sample(noise[:,:,:cfg.window_size], num_steps=cfg.diffusion_scheduler_steps, show_progress=True)

        if (i +1) % 50 == 0:
            decoded = codec.decode(diff[0].cpu())
            orig = codec.decode(batch_tensor[0].cpu())
            torchaudio.save(f"./sample_{i+1}.wav", decoded.cpu(), sample_rate=44100)
            torchaudio.save(f"./orig_{i+1}.wav", orig.cpu(), sample_rate=44100)
            print(decoded.shape)
                    # Save a checkpoint every n epochs
        if i % 100 == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': i
                }
                torch.save(checkpoint, f"./diffusion_baseline_epoch{i}.pt")

    print (img)
    codec = EncoderDecoder()

    for epoch in range(cfg.epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_mu = 0.0
        total_std = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for _, z_slow in pbar:
            z_slow_target = z_slow[:, :, cfg.context_window_size:]  # remove context from main input
            z_slow_ctx = z_slow[:, :, :cfg.context_window_size]

            optimizer.zero_grad()

            if epoch + 1 < 200:
                beta = 0.0
            else:
                beta = min(1.0, epoch / 500) * cfg.beta
            # beta = max(1e-4, min(1.0, epoch / 500)) * cfg.beta
            # beta = 0 

            outputs = model(z_slow_ctx)

            loss, recon, kl = vae_slow_loss(z_slow_target, outputs, beta=beta)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            total_mu += outputs[1].mean().item()
            total_std += torch.exp(0.5 * outputs[2]).mean().item()

            pbar.set_postfix({
                "loss": loss.item(),
                "recon": recon.item(),
                "kl": kl.item()
            })

        avg_loss = total_loss / len(loader)
        avg_recon = total_recon / len(loader)
        avg_kl = total_kl / len(loader)
        avg_mu = total_mu / len(loader)
        avg_std = total_std / len(loader)

        print(
            f"[Epoch {epoch+1}] "
            f"Loss: {avg_loss:.4f} | "
            f"Recon: {avg_recon:.4f} | "
            f"KL: {avg_kl:.4f} | "
            f"Beta: {beta:.4f}"
        )


        if (epoch + 1) % 5 == 0:
            writer.add_scalar("Loss/Train", avg_loss, epoch+1)
            writer.add_scalar("Recon/Train", avg_recon, epoch+1)
            writer.add_scalar("KL/Train", avg_kl, epoch+1)
            writer.add_scalar("Mu/Train", avg_mu, epoch+1)
            writer.add_scalar("Std/Train", avg_std, epoch+1)

        if (epoch + 1) % 25 == 0:
            for _, z_val_slow in loader:
                z_slow_target = z_val_slow[:, :, cfg.context_window_size:]  # remove context from main input
                z_slow_ctx = z_val_slow[:, :, :cfg.context_window_size]
                z_hat_val, _, _ = model(z_slow_ctx)
                break  # take first batch only

            plot_latent_pca(z_slow_target[0:1].detach(), z_hat_val[0:1].detach(), epoch+1)
            if (epoch + 1) % 100 == 0:
                recon = codec.decode(z_hat_val)
                orig = codec.decode(z_slow_target)
                for i in range(recon.shape[0]):
                    writer.add_audio(f"Recon/GenLatent_{i}", recon[i:i+1], sample_rate=44100, global_step=epoch+1)
                    writer.add_audio(f"Recon/Orig_{i}", orig[i:i+1], sample_rate=44100, global_step=epoch+1)
                    torchaudio.save(f"./reconstructions/recon_{i}_{epoch+1}.wav", recon[i:i+1], sample_rate=44100)
                    torchaudio.save(f"./reconstructions/orig_{i}_{epoch+1}.wav", orig[i:i+1], sample_rate=44100)
        if (epoch + 1) % 500 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            )

if __name__ == "__main__":


    data_dir = Path("/Users/adees/Code/multi-scale-rnn-vae/test_audio/latents")
    ext = ".pt"
    train_files = [str(p) for p in sorted(data_dir.glob(f"*{ext}"))]

    if not train_files:
        raise FileNotFoundError(f"No '{ext}' files found in {data_dir}")
    
    save_dir = "./checkpoints"

    cfg = Config()
    train(cfg, train_files, save_dir)
