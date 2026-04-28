"""Circular replay buffer shared by AlphaZero and AlphaGumbel training.

Stores (obs, policy, value) triples from self-play games. Oldest entries are
silently overwritten once the buffer is full. All arrays are preallocated up
front to avoid repeated reallocations during long runs.

Typical sizing: 500k positions ~ 16 iterations of history at 1000 games/iter
and ~30 moves/game.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Preallocated circular buffer for AlphaZero-style training samples."""

    def __init__(self, max_size: int = 500_000) -> None:
        self.max_size = max_size
        self._obs    = np.zeros((max_size, 6, 9, 9), dtype=np.float32)
        self._policy = np.zeros((max_size, 9, 9),    dtype=np.float32)
        self._value  = np.zeros(max_size,             dtype=np.float32)
        self._ptr    = 0   # next write position
        self._size   = 0   # current number of valid entries

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_game(
        self,
        obs:    np.ndarray,  # (T, 6, 9, 9)
        policy: np.ndarray,  # (T, 9, 9)
        value:  np.ndarray,  # (T,)
    ) -> None:
        """Add all positions from one game to the buffer.

        Wraps around if adding T positions would exceed ``max_size``.
        """
        T = len(value)
        if T == 0:
            return

        space_left = self.max_size - self._ptr
        if T <= space_left:
            self._obs   [self._ptr : self._ptr + T] = obs
            self._policy[self._ptr : self._ptr + T] = policy
            self._value [self._ptr : self._ptr + T] = value
        else:
            # Write wraps around the end -> split into two slices.
            self._obs   [self._ptr :] = obs   [:space_left]
            self._policy[self._ptr :] = policy[:space_left]
            self._value [self._ptr :] = value [:space_left]
            remainder = T - space_left
            self._obs   [:remainder] = obs   [space_left:]
            self._policy[:remainder] = policy[space_left:]
            self._value [:remainder] = value [space_left:]

        self._ptr  = (self._ptr + T) % self.max_size
        self._size = min(self._size + T, self.max_size)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a random minibatch of (obs, policy, value)."""
        if batch_size > self._size:
            raise ValueError(
                f"Requested {batch_size} samples but buffer only has {self._size}"
            )
        idx = np.random.randint(0, self._size, size=batch_size)
        return self._obs[idx], self._policy[idx], self._value[idx]

    # ------------------------------------------------------------------
    # Introspection / loading
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of valid entries currently in the buffer."""
        return self._size

    @property
    def is_ready(self) -> bool:
        """True once there is at least one sample available."""
        return self._size > 0

    def load_from_npz(self, path: str, max_positions: int | None = None) -> int:
        """Load a dataset.npz file directly into the buffer.

        The .npz file must have keys: ``obs`` (N, 6, 9, 9), ``policy`` (N, 9, 9),
        ``value`` (N,). Returns the number of positions loaded.
        """
        data   = np.load(path)
        obs    = data["obs"].astype(np.float32)
        policy = data["policy"].astype(np.float32)
        value  = data["value"].astype(np.float32)

        # Apply caller-specified cap first, then hard buffer-size cap.
        cap = min(
            max_positions if max_positions is not None else len(value),
            self.max_size,
        )
        if len(value) > cap:
            obs    = obs   [-cap:]
            policy = policy[-cap:]
            value  = value [-cap:]

        self.add_game(obs, policy, value)
        return len(value)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={self._size:,}/{self.max_size:,}, "
            f"ptr={self._ptr})"
        )
