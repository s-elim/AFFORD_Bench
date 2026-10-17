"""Policy architectures for the AFFORD-Bench baselines (behavior cloning and a DDPM action-chunk diffusion policy).

`proprio_dim` is the length of the low-dim input vector (see afford_pro.learning.features.input_dim). Visual models
also take a 3 x 80 x 80 workcell image. Both predict `chunk_size` future actions in [-1, 1].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """Behavioral Cloning Policy supporting both State and Visuomotor inputs."""

    def __init__(self, action_dim: int = 7, chunk_size: int = 16, proprio_dim: int = 22, modality: str = "state"):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.modality = modality

        if modality == "visual":
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            in_dim = 64 * 4 * 4 + proprio_dim
        else:
            in_dim = proprio_dim

        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, chunk_size * action_dim),
        )

    def forward(self, proprio: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
        if self.modality == "visual" and image is not None:
            v_feat = self.conv(image)
            feat = torch.cat([v_feat, proprio], dim=-1)
        else:
            feat = proprio
        out = self.head(feat)
        return out.view(-1, self.chunk_size, self.action_dim)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DiffusionPolicy(nn.Module):
    """1D Action Diffusion Policy with DDPM variance schedule."""

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 16,
        proprio_dim: int = 22,
        modality: str = "state",
        diffusion_steps: int = 50,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.modality = modality
        self.diffusion_steps = diffusion_steps

        if modality == "visual":
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            cond_in = 64 * 4 * 4 + proprio_dim
        else:
            cond_in = proprio_dim

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        flat_act_dim = chunk_size * action_dim
        self.denoise_net = nn.Sequential(
            nn.Linear(flat_act_dim + 128 + 128, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, flat_act_dim),
        )

        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        proprio: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modality == "visual" and image is not None:
            v_feat = self.conv(image)
            cond_inp = torch.cat([v_feat, proprio], dim=-1)
        else:
            cond_inp = proprio

        cond = self.cond_mlp(cond_inp)
        t_emb = self.time_mlp(timesteps)

        B = noisy_actions.size(0)
        flat_act = noisy_actions.view(B, -1)
        x = torch.cat([flat_act, cond, t_emb], dim=-1)
        noise_pred = self.denoise_net(x)
        return noise_pred.view(B, self.chunk_size, self.action_dim)

    @torch.no_grad()
    def sample_actions(self, proprio: torch.Tensor, image: torch.Tensor | None = None) -> torch.Tensor:
        """Reverse diffusion sampling from pure Gaussian noise."""
        B = proprio.size(0)
        device = proprio.device
        act = torch.randn(B, self.chunk_size, self.action_dim, device=device)

        for step in reversed(range(self.diffusion_steps)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            pred_noise = self.forward(act, t, proprio, image)

            alpha = self.alphas[step]
            alpha_cumprod = self.alphas_cumprod[step]
            beta = self.betas[step]

            noise = torch.randn_like(act) if step > 0 else 0.0
            act = (1 / torch.sqrt(alpha)) * (act - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise)
            if step > 0:
                act = act + torch.sqrt(beta) * noise

        return torch.clamp(act, -1.0, 1.0)
