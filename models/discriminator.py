"""
Conditional Discriminator: image + text conditioning -> real/fake logit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .generator import weights_init


class ConditionalDiscriminator(nn.Module):
    """
    Classifies images as real/fake while conditioned on text embedding.

    Architecture: 4x Conv2d -> global pool -> concat text -> FC logit.
    """

    def __init__(
        self,
        cond_dim: int = 256,
        ndf: int = 64,
        nc: int = 3,
    ):
        super().__init__()
        self.cond_dim = cond_dim

        self.conv_blocks = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 -> 16x16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 4x4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ndf*8 * 4 * 4 = 8192 for ndf=64
        self.feature_dim = ndf * 8 * 4 * 4
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + cond_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

        self.apply(weights_init)

    def forward(
        self,
        image: torch.Tensor,
        cond: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (B, 3, 64, 64)
            cond: (B, cond_dim)
        Returns:
            (B, 1) logits, or (logits, flattened conv features) if return_features=True
        """
        features = self.conv_blocks(image)
        flat = features.view(features.size(0), -1)
        x = torch.cat([flat, cond], dim=1)
        out = self.classifier(x)
        if return_features:
            return out, flat
        return out
