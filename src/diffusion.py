import torch

class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps= timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.device = device
        self.alphas= self.alphas.to(device)
        self.betas = self.betas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self
    
    def q_sample(self, z_start, t, noise=None):
        if noise is None:
            noise=torch.randn_like(z_start)

        alpha_bar_t = self.alpha_bars[t]

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t).view(-1,1,1,1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t).view(-1, 1, 1, 1)
        z_noisy = sqrt_alpha_bar_t * z_start + sqrt_one_minus_alpha_bar_t * noise
        return z_noisy, noise