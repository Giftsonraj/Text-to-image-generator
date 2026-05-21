"""
Text encoder: DistilBERT (frozen) + trainable projection to conditioning vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "distilbert-base-uncased"
DEFAULT_COND_DIM = 256


class TextEncoder(nn.Module):
    """Encode text captions into a fixed-size conditioning vector for the GAN."""

    def __init__(
        self,
        cond_dim: int = DEFAULT_COND_DIM,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 64,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.max_length = max_length
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        # Freeze pretrained weights — only projection is trained with the GAN.
        for param in self.bert.parameters():
            param.requires_grad = False

        hidden_size = self.bert.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, cond_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return conditioning vectors of shape (batch, cond_dim)."""
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over token embeddings (masked).
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.projection(pooled)

    def tokenize(self, texts: list[str], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoded.items()}

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device | str = "cpu") -> torch.Tensor:
        """Encode a list of strings into conditioning vectors."""
        self.eval()
        batch = self.tokenize(texts, device=device)
        return self.forward(batch["input_ids"], batch["attention_mask"])
