"""
Conditional Generator: noise + text conditioning -> 64x64 RGB image.

Uses Upsample + Conv2d instead of ConvTranspose2d to reduce checkerboard artifacts.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def weights_init(m: nn.Module) -> None:
    """DCGAN-style weight initialization."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def _upsample_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Nearest-neighbor upsample + conv (avoids checkerboard from ConvTranspose2d)."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ConditionalGenerator(nn.Module):
    """
    Maps (z, text_embedding) to a 64x64 image.

    Architecture: FC -> 4x upsample blocks (64x64 output).
    """

    def __init__(
        self,
        z_dim: int = 100,
        cond_dim: int = 256,
        ngf: int = 64,
        nc: int = 3,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim

        self.fc = nn.Sequential(
            nn.Linear(z_dim + cond_dim, ngf * 8 * 4 * 4),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.ReLU(True),
        )

        self.conv_blocks = nn.Sequential(
            _upsample_block(ngf * 8, ngf * 4),  # 4 -> 8
            _upsample_block(ngf * 4, ngf * 2),  # 8 -> 16
            _upsample_block(ngf * 2, ngf),      # 16 -> 32
            _upsample_block(ngf, ngf),          # 32 -> 64
            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, cond], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        return self.conv_blocks(x)
