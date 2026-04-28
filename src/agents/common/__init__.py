"""Shared building blocks used by every neural agent (PPO, AlphaZero, AlphaGumbel).

The CNN trunk + policy / value heads are functionally identical across all
three trainers, so they live here once.  The Transformer backbone in
``agents/alphazero/transformer.py`` is genuinely different and stays separate
but reuses ``masked_policy_probs`` from this package.
"""

from agents.common.network import (
    AlphaZeroNet,
    PolicyValueNet,
    ResBlock,
    UTTTNet,
    masked_log_probs,
    masked_policy_probs,
    masked_softmax,
)
from .replay_buffer import ReplayBuffer

__all__ = [
    "AlphaZeroNet",
    "PolicyValueNet",
    "ResBlock",
    "UTTTNet",
    "ReplayBuffer",
    "masked_log_probs",
    "masked_policy_probs",
    "masked_softmax",
]
