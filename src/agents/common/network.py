"""Shared CNN trunk + policy / value heads used by every neural agent.

PPO's ``UTTTNet`` and AlphaZero's ``AlphaZeroNet`` were two byte-identical
copies of the same network — they are now one class, with ``UTTTNet`` exported
as an alias so existing PPO checkpoints and call sites keep working.

Three masking helpers live alongside because callers want different downstream
shapes:

    * ``masked_policy_probs`` → softmax over legal moves (used by MCTS to pick
      child priors)
    * ``masked_log_probs``    → log-softmax over legal moves (used by PPO for
      log-prob lookups inside the clipped surrogate)
    * ``masked_softmax``      → like ``masked_policy_probs`` but operates on
      tensors flattened to ``(B, 81)`` internally; used inside the MCTS leaf
      evaluators
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F


@runtime_checkable
class PolicyValueNet(Protocol):
    """Structural type of any AlphaToe-compatible policy/value network.

    Both ``AlphaZeroNet`` (CNN, defined below) and
    ``AlphaZeroTransformerNet`` (in ``agents/alphazero/transformer.py``)
    satisfy this protocol, so the search algorithms (``AlphaZeroMCTS``,
    ``AlphaGumbelMCTS``) can be wired to any backbone interchangeably.
    """

    def __call__(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: ``(B, 6, 9, 9)`` obs -> ``((B, 9, 9) logits, (B,) value)``."""

    def eval(self) -> "PolicyValueNet": ...
    def train(self, mode: bool = True) -> "PolicyValueNet": ...
    def to(self, device: torch.device | str) -> "PolicyValueNet": ...


class ResBlock(nn.Module):
    """Pre-activation residual block: Conv-BN-ReLU-Conv-BN + skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class AlphaZeroNet(nn.Module):
    """ResNet trunk with separate policy and value heads.

    Input  : (B, 6, 9, 9) float32 observation (from ``env.observation.observe``)
    Output : ``policy_logits`` (B, 9, 9) raw logits, ``value`` (B,) in [-1, 1]
    """

    def __init__(
        self,
        channels:      int   = 64,
        num_blocks:    int   = 3,
        value_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Shared trunk
        self.input_conv = nn.Conv2d(6, channels, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        # Policy head: 2 x 1x1 conv -> flatten -> FC -> (9, 9) logits
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 9 * 9, 81)

        # Value head: 1 x 1x1 conv -> flatten -> FC -> FC -> tanh
        # Dropout (when value_dropout > 0) applied AFTER the hidden ReLU --
        # the standard place to regularize the FC bottleneck.
        self.value_conv    = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn      = nn.BatchNorm2d(1)
        self.value_fc1     = nn.Linear(1 * 9 * 9, 64)
        self.value_dropout = nn.Dropout(value_dropout) if value_dropout > 0 else nn.Identity()
        self.value_fc2     = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.input_bn(self.input_conv(obs)))
        x = self.res_blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_fc(p.view(p.size(0), -1))
        policy_logits = p.view(-1, 9, 9)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = F.relu(self.value_fc1(v.view(v.size(0), -1)))
        v = self.value_dropout(v)
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return policy_logits, value


# Backwards-compatible alias for PPO call sites and checkpoints.
UTTTNet = AlphaZeroNet


# ----------------------------------------------------------------------
# Masking helpers — three shapes for three downstream callers
# ----------------------------------------------------------------------


def masked_policy_probs(
    logits:     torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Softmax over legal moves (illegal moves get probability 0).

    Args:
        logits:     (9, 9) or (B, 9, 9) raw policy logits.
        legal_mask: same shape bool tensor (True = legal).

    Returns:
        probs: same shape float32, sums to 1 over legal moves.
    """
    flat_logits = logits.reshape(-1, 81)
    flat_mask   = legal_mask.reshape(-1, 81)
    flat_logits = flat_logits.masked_fill(~flat_mask, -1e9)
    probs = F.softmax(flat_logits, dim=-1)
    return probs.reshape(logits.shape)


def masked_log_probs(
    logits:     torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Log-softmax over legal moves, returned flat as ``(B, 81)``.

    PPO uses this directly inside the surrogate-loss minibatch loop; the flat
    shape is convenient for ``Categorical(logits=...)``.
    """
    logits_flat = logits.reshape(-1, 81)
    mask_flat   = legal_mask.reshape(-1, 81)
    logits_flat = logits_flat.masked_fill(~mask_flat, -1e8)
    return F.log_softmax(logits_flat, dim=-1)


def masked_softmax(
    logits: torch.Tensor,
    mask:   torch.Tensor,
) -> torch.Tensor:
    """Batched softmax over legal moves preserving the input shape.

    Used by the MCTS leaf evaluators which receive ``(B, 9, 9)`` logits and
    expect ``(B, 9, 9)`` priors back.
    """
    flat_logits = logits.reshape(logits.size(0), -1)
    flat_mask   = mask.reshape(mask.size(0), -1).bool()
    flat_logits = flat_logits.masked_fill(~flat_mask, -1e9)
    probs = F.softmax(flat_logits, dim=-1)
    return probs.reshape_as(logits)
