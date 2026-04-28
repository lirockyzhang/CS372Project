"""Mini transformer policy-value network for AlphaZero on UTTT.

Input:
    obs: (B, 6, 9, 9) float32

Output:
    policy_logits: (B, 9, 9)
    value:         (B,)

Design:
    - 81 cell tokens (one per board cell)
    - each token sees the 6 observation planes at that cell
    - learned positional embeddings
    - small Transformer encoder
    - policy head predicts one logit per cell token
    - value head reads from a learned CLS token
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniTransformerBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class AlphaZeroTransformerNet(nn.Module):
    """Small transformer policy-value network for UTTT."""

    def __init__(
        self,
        embed_dim:     int   = 128,
        depth:         int   = 4,
        num_heads:     int   = 4,
        mlp_ratio:     float = 4.0,
        dropout:       float = 0.0,
        value_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_tokens = 81
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(6, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.Sequential(
            *[
                MiniTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        self.policy_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )

        # Dropout (when value_dropout > 0) sits between the GELU and the final
        # projection -- standard FC-bottleneck regularization for the value head.
        value_drop_layer: nn.Module = (
            nn.Dropout(value_dropout) if value_dropout > 0 else nn.Identity()
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            value_drop_layer,
            nn.Linear(embed_dim, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.input_proj:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _tokenize(self, obs: torch.Tensor) -> torch.Tensor:
        """Convert (B, 6, 9, 9) -> (B, 81, 6)."""
        return obs.permute(0, 2, 3, 1).reshape(obs.size(0), 81, 6)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = obs.size(0)

        tokens = self._tokenize(obs)           # (B, 81, 6)
        x = self.input_proj(tokens)            # (B, 81, D)
        x = x + self.pos_embed                 # (B, 81, D)

        cls = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        x = torch.cat([cls, x], dim=1)         # (B, 82, D)

        x = self.blocks(x)
        x = self.final_norm(x)

        cls_out = x[:, 0]                      # (B, D)
        cell_out = x[:, 1:]                    # (B, 81, D)

        policy_logits = self.policy_head(cell_out).squeeze(-1).view(B, 9, 9)
        value = torch.tanh(self.value_head(cls_out).squeeze(-1))

        return policy_logits, value


