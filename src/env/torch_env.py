"""PyTorch adapter around the NumPy VectorUTTTEnv.

Keeps game simulation in NumPy for simplicity/speed, but exposes a
tensor-friendly API for PPO / AlphaZero training code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .vector_env import VectorUTTTEnv


class TorchUTTTEnv:
    """Thin PyTorch wrapper for VectorUTTTEnv.

    API:
        reset() -> (obs, legal_mask)
        step(actions) -> (obs, rewards, terminated, legal_mask)

    Shapes:
        obs:         (N, 6, 9, 9) float32
        legal_mask:  (N, 9, 9) bool
        rewards:     (N,) float32
        terminated:  (N,) bool
        actions:     (N, 2) int64, where [:, 0] = board_idx, [:, 1] = cell_idx
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cpu",
        non_blocking: bool = False,
    ) -> None:
        self.env = VectorUTTTEnv(num_envs)
        self.device = torch.device(device)
        self.non_blocking = non_blocking

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def _to_tensor(
        self,
        array: np.ndarray,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Convert NumPy array to torch tensor on target device."""
        tensor = torch.from_numpy(array)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor.to(self.device, non_blocking=self.non_blocking)

    def _actions_to_numpy(self, actions: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert action batch to NumPy int64 array of shape (N, 2)."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().to("cpu").numpy()

        actions = np.asarray(actions, dtype=np.int64)

        if actions.shape != (self.num_envs, 2):
            raise ValueError(
                f"Expected actions shape {(self.num_envs, 2)}, got {actions.shape}"
            )

        return actions

    def reset(
        self,
        mask: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset all envs or a masked subset.

        Args:
            mask: Optional (N,) boolean mask. True entries are reset.

        Returns:
            obs: (N, 6, 9, 9) float32 tensor
            legal_mask: (N, 9, 9) bool tensor
        """
        mask_np: np.ndarray | None
        if mask is None:
            mask_np = None
        elif isinstance(mask, torch.Tensor):
            mask_np = mask.detach().to("cpu").numpy().astype(np.bool_)
        else:
            mask_np = np.asarray(mask, dtype=np.bool_)

        self.env.reset(mask=mask_np)

        obs_np = self.env.observe()
        legal_mask_np = self.env.legal_action_mask()

        obs = self._to_tensor(obs_np, dtype=torch.float32)
        legal_mask = self._to_tensor(legal_mask_np)

        return obs, legal_mask

    def observe(self) -> torch.Tensor:
        """Return current observations as torch tensor."""
        return self._to_tensor(self.env.observe(), dtype=torch.float32)

    def legal_action_mask(self) -> torch.Tensor:
        """Return current legal action masks as torch tensor."""
        return self._to_tensor(self.env.legal_action_mask())

    def step(
        self,
        actions: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step all environments once.

        Args:
            actions: (N, 2) tensor/array of (board_idx, cell_idx).

        Returns:
            obs: (N, 6, 9, 9) float32 tensor
            rewards: (N,) float32 tensor
            terminated: (N,) bool tensor
            legal_mask: (N, 9, 9) bool tensor
        """
        actions_np = self._actions_to_numpy(actions)

        rewards_np, terminated_np, legal_mask_np = self.env.step(actions_np)
        obs_np = self.env.observe()

        obs = self._to_tensor(obs_np, dtype=torch.float32)
        rewards = self._to_tensor(rewards_np, dtype=torch.float32)
        terminated = self._to_tensor(terminated_np)
        legal_mask = self._to_tensor(legal_mask_np)

        return obs, rewards, terminated, legal_mask

    def sample_random_actions(self) -> torch.Tensor:
        """Sample one legal random action per environment.

        Useful for smoke tests / baselines.
        Returns:
            actions: (N, 2) int64 tensor
        """
        mask = self.env.legal_action_mask() # (N, 9, 9)
        actions = np.zeros((self.num_envs, 2), dtype=np.int64)

        for i in range(self.num_envs):
            legal = np.argwhere(mask[i])
            if len(legal) == 0:
                raise RuntimeError(f"No legal actions available in env {i}")
            choice = legal[np.random.randint(len(legal))]
            actions[i] = choice  # [board_idx, cell_idx]

        return self._to_tensor(actions, dtype=torch.int64)

    def get_state(self) -> dict[str, Any]:
        """Optional helper for debugging / inspection."""
        return {
            "boards_x": self.env.boards_x.copy(),
            "boards_o": self.env.boards_o.copy(),
            "meta_x": self.env.meta_x.copy(),
            "meta_o": self.env.meta_o.copy(),
            "meta_draw": self.env.meta_draw.copy(),
            "current_player": self.env.current_player.copy(),
            "active_board": self.env.active_board.copy(),
            "terminated": self.env.terminated.copy(),
            "winner": self.env.winner.copy(),
        }