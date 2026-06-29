import math
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*4),
            nn.SiLU(),
            nn.Linear(embedding_dim*4, embedding_dim)
        )

    def forward(self, t):
        half_dim = self.embedding_dim // 2
        device = t.device

        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / (half_dim - 1)
        )
        angles = t[:, None].float() * frequencies[None, :]
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return self.mlp(embedding)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        time_bias = self.time_proj(t_emb)
        h = h + time_bias[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.skip(x)
    

class LatentUNet(nn.Module):
    def __init__(self, latent_channels=4, base_channels=64, time_dim = 128):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_dim)
        self.input_conv = nn.Conv2d(latent_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = ResBlock(base_channels, base_channels, time_dim)
        self.downsample = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.middle = ResBlock(base_channels * 2, base_channels * 2, time_dim)

        self.upsample = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up1 = ResBlock(base_channels * 2, base_channels, time_dim)

        self.output_conv = nn.Conv2d(base_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, z, t, cond=None):
        t_emb = self.time_embedding(t)

        x = self.input_conv(z)

        skip = self.down1(x, t_emb)
        x = self.downsample(skip)

        x = self.middle(x, t_emb)

        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.up1 (x, t_emb)

        return self.output_conv(x)
    