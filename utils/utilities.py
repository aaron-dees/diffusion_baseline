# PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn.functional as F

def plot_latent_pca(z, z_hat, i):
# Use first batch of validation for plotting
    target_seq = z.permute(0,2,1)[0].cpu().numpy()
    pred_seq = z_hat.permute(0,2,1)[0].cpu().numpy()
    all_seq = np.vstack([target_seq])
    pca = PCA(n_components=3)
    print(all_seq.shape)
    Zp_target = pca.fit_transform(all_seq)
    Zp_pred = pca.transform(pred_seq)
    # Zp_target = Zp[:target_seq.shape[0]]
    # Zp_pred = Zp[target_seq.shape[0]:]
    # compute per-sequence mean vectors (matching how PCA was fit: features are the latent dim)
    # src_vec = z.mean(dim=2)[0].cpu().numpy()          # source sequence mean, shape (dim,)
    # tgt_vec = z_hat.mean(dim=2)[0].cpu().numpy()   # target sequence mean, shape (dim,)

    # transform to PCA coords (reshape to (1, dim) for transform)
    # Zp_src = pca.transform(src_vec[None, :])   # shape (1, n_components)
    # Zp_tgt = pca.transform(tgt_vec[None, :])
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(Zp_target[:, 0], Zp_target[:, 1], '-o', markersize=3, alpha=0.6, label='Target')
    plt.plot(Zp_pred[:, 0], Zp_pred[:, 1], '-o', markersize=3, alpha=0.6, label='Prediction')
    # plt.scatter(Zp_src[0, 0], Zp_src[0, 1], marker='X', s=100, c='black', label='Source mean')
    # plt.scatter(Zp_tgt[0, 0], Zp_tgt[0, 1], marker='D', s=100, c='magenta', label='Target mean')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Latent Trajectories (Top 2 PCA)')
    plt.legend()
    plt.grid(True)
    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.plot(Zp_target[:, 0], Zp_target[:, 1], Zp_target[:, 2], '-o', markersize=3, alpha=0.6, label='Target')
    ax.plot(Zp_pred[:, 0], Zp_pred[:, 1], Zp_pred[:, 2], '-o', markersize=3, alpha=0.6, label='Prediction')
    # ax.scatter(Zp_src[0, 0], Zp_src[0, 1], Zp_src[0, 2], marker='X', s=80, c='black', label='Source mean')
    # ax.scatter(Zp_tgt[0, 0], Zp_tgt[0, 1], Zp_tgt[0, 2], marker='D', s=80, c='magenta', label='Target mean')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Latent Trajectories (Top 3 PCA)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"./latent_predictor_pca_{i}.png")
    plt.close()


@torch.no_grad()
def moving_average_smoothing(z, kernel_size, zero_phase=False):
    """
    Moving average smoothing.

    Args:
        z: [B, D, T]
        kernel_size: odd integer (e.g. 9, 25, 51)
        zero_phase: if True, forward-backward filtering

    Returns:
        z_smoothed: [B, D, T]
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"

    B, D, T = z.shape
    pad = kernel_size // 2

    # Reflect padding avoids boundary clicks
    z_pad = F.pad(z, (pad, pad), mode='reflect')

    # Create box kernel
    kernel = torch.ones(D, 1, kernel_size, device=z.device) / kernel_size

    # Depthwise conv (each latent dim filtered independently)
    z_smooth = F.conv1d(z_pad, kernel, groups=D)

    if zero_phase:
        # Apply again reversed (forward-backward)
        z_rev = torch.flip(z_smooth, dims=[-1])
        z_rev = F.pad(z_rev, (pad, pad), mode='reflect')
        z_rev = F.conv1d(z_rev, kernel, groups=D)
        z_smooth = torch.flip(z_rev, dims=[-1])

    return z_smooth

@torch.no_grad()
def lowpass_moving_average(z, kernel_size):
    with torch.no_grad():
        """
        z: [B, D, T]
        kernel_size: int (e.g. 25, 50, 100)
        """
        B, D, T = z.shape

        # reshape to [B*D, 1, T] for conv1d
        z_ = z.reshape(B * D, 1, T)

        kernel = torch.ones(1, 1, kernel_size, device=z.device) / kernel_size
        padding = kernel_size // 2

        z_lp = F.conv1d(z_, kernel, padding=padding)

        # back to [B, D, T]
        z_lp = z_lp.reshape(B, D, T)

    return z_lp

@torch.no_grad()
def gaussian_smoothing(z, kernel_size, sigma, zero_phase=False):
    """
    Gaussian smoothing.

    Args:
        z: [B, D, T]
        kernel_size: odd integer
        sigma: standard deviation of Gaussian
        zero_phase: apply forward-backward filtering

    Returns:
        z_smoothed: [B, D, T]
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"

    B, D, T = z.shape
    pad = kernel_size // 2

    # Create Gaussian kernel
    t = torch.arange(kernel_size, device=z.device) - pad
    kernel_1d = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel = kernel_1d.view(1, 1, kernel_size).repeat(D, 1, 1)

    # Reflect padding
    z_pad = F.pad(z, (pad, pad), mode='reflect')

    # Depthwise convolution
    z_smooth = F.conv1d(z_pad, kernel, groups=D)

    if zero_phase:
        z_rev = torch.flip(z_smooth, dims=[-1])
        z_rev = F.pad(z_rev, (pad, pad), mode='reflect')
        z_rev = F.conv1d(z_rev, kernel, groups=D)
        z_smooth = torch.flip(z_rev, dims=[-1])

    return z_smooth
