from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from agents.common.network import UTTTNet, masked_log_probs

DEFAULT_CHECKPOINT = Path(__file__).parent / "model" / "ppo.pt"


class PPOAgent:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: str = "cpu",
        channels: int = 64,
        num_blocks: int = 3,
    ) -> None:
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_CHECKPOINT
        self.device = torch.device(device)
        self.net = UTTTNet(channels=channels, num_blocks=num_blocks).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.net.load_state_dict(checkpoint["model"])
        else:
            self.net.load_state_dict(checkpoint)
        self.net.eval()

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        legal_mask: np.ndarray,
        *,
        greedy: bool = True,
        temperature: float = 1.0,
    ) -> tuple[int, int]:
        """Select an action given observation and legal mask.

        Args:
            obs: (6, 9, 9) float32 observation.
            legal_mask: (9, 9) bool legal action mask.
            greedy: If True, pick the highest-probability legal action.
            temperature: Softmax temperature when not greedy (lower = sharper).

        Returns:
            (board_idx, cell_idx) tuple.
        """
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(legal_mask).unsqueeze(0).to(self.device)

        logits, _ = self.net(obs_t)

        if not greedy:
            logits = logits / temperature

        log_probs = masked_log_probs(logits, mask_t)

        if greedy:
            flat_action = log_probs.argmax(dim=-1).item()
        else:
            flat_action = torch.multinomial(log_probs.exp(), 1).item()

        return flat_action // 9, flat_action % 9
