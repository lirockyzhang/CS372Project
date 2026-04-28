"""Gymnasium-style single-game environment for UTTT."""

from __future__ import annotations

import numpy as np

from .logic import legal_action_mask, legal_actions, step
from .observation import observe
from .state import UTTTState, initial_state


class UTTTEnv:
    """Single-game Ultimate Tic-Tac-Toe environment.

    Actions are (board_idx, cell_idx) tuples, each 0-8.
    Observations are (6, 9, 9) float32 arrays from the current player's POV.
    """

    def __init__(self) -> None:
        self._state: UTTTState = initial_state()

    @property
    def state(self) -> UTTTState:
        return self._state

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset to a fresh game. Returns (observation, info)."""
        self._state = initial_state()
        obs = observe(self._state)
        info = {
            "legal_action_mask": legal_action_mask(self._state),
            "current_player": self._state.current_player,
        }
        return obs, info

    def step(
        self, action: tuple[int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply action. Returns (obs, reward, terminated, truncated, info).

        Reward is +1 for a win, -1 for a loss, 0 otherwise,
        from the perspective of the player who just moved.
        """
        player_before = self._state.current_player
        step(self._state, action)

        reward = 0.0
        if self._state.terminated:
            if self._state.winner == player_before:
                reward = 1.0
            elif self._state.winner != -1:
                reward = -1.0

        obs = observe(self._state)
        info = {
            "legal_action_mask": legal_action_mask(self._state),
            "current_player": self._state.current_player,
        }
        return obs, reward, self._state.terminated, False, info

    def legal_action_mask(self) -> np.ndarray:
        """Return (9, 9) bool mask."""
        return legal_action_mask(self._state)

    def legal_actions(self) -> list[tuple[int, int]]:
        """Return list of legal (board_idx, cell_idx)."""
        return legal_actions(self._state)

    def clone(self) -> UTTTEnv:
        """Deep copy for MCTS tree search."""
        new = UTTTEnv.__new__(UTTTEnv)
        new._state = self._state.clone()
        return new
