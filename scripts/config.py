import torch

class Config:
    latent_dim = 64
    e_dim = 64
    # e_dim = 256

    enc_hidden_dim = 512
    slow_hidden_dim = 512
    fast_hidden_dim = 800

    # K = 5                      # slow update rate
    # K = 9                      # slow update rate
    K = 15                      # slow update rate
    window_size = 128          # ~10 seconds
    # window_size = 11          # ~2 seconds
    batch_size = 4
    dataset_stride = 15
    context_encoder_pooling_rate = 1

    target_window_size = 56
    context_window_size = 111   # size of context window
    context_window_hidden_dim = 256

    lr = 1e-4
    # beta = 1e-6                 # KL weight
    beta = 1e-5                 # KL weight
    # beta = 0                 # KL weight
    epochs = 2000

    device = "cuda" if torch.cuda.is_available() else "cpu"


    diffusion_save_out_path = "./checkpoints"
    diffusion_save_name = "test_run"
    diffusion_finetune = False
    diffusion_scheduler_steps = 100
    diffusion_accumulation_steps = 2
    diffusion_epochs = 25000
    diffusion_save_interval = 500
    diffusion_temperature = 1.0
    diffusion_view_analysis_interval = 50
    diffusion_waveform_model_path = "/Users/adees/Downloads/encodec_model.pt"
    diffusion_scale_required = False