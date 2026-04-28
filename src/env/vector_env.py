"""Vectorized (batched) UTTT environment using NumPy for PPO self-play."""

from __future__ import annotations

import numpy as np

from .state import FULL_BOARD, WIN_MASKS

# Pre-compute win masks as a NumPy array for vectorized checks.
_WIN_MASKS_NP = np.array(WIN_MASKS, dtype=np.uint16)  # (8,)


def _check_win_vec(bits: np.ndarray) -> np.ndarray:
    """Vectorized win check: bits shape (...,), returns bool (...)."""
    # bits[..., None] & masks[None, ...] == masks[None, ...]
    expanded = bits[..., np.newaxis]  # (..., 1)
    masks = _WIN_MASKS_NP  # (8,)
    return np.any((expanded & masks) == masks, axis=-1)

class VectorUTTTEnv:
    """Batched UTTT environment → N parallel games with NumPy-vectorized step.

    Actions: (N, 2) int array where [:, 0] = board_idx, [:, 1] = cell_idx.
    Observations: (N, 6, 9, 9) float32.
    """

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self._allocate()

    def _allocate(self) -> None:
        N = self.num_envs
        self.boards_x = np.zeros((N, 9), dtype=np.uint16)
        self.boards_o = np.zeros((N, 9), dtype=np.uint16)
        self.meta_x = np.zeros(N, dtype=np.uint16)
        self.meta_o = np.zeros(N, dtype=np.uint16)
        self.meta_draw = np.zeros(N, dtype=np.uint16)
        self.current_player = np.zeros(N, dtype=np.int8)
        self.active_board = np.full(N, -1, dtype=np.int8)
        self.terminated = np.zeros(N, dtype=np.bool_)
        self.winner = np.full(N, -1, dtype=np.int8)

    def reset(self, mask: np.ndarray | None = None) -> None:
        """Reset all envs (or those indicated by *mask*)."""
        if mask is None:
            self._allocate()
            return

        idx = np.where(mask)[0]
        if len(idx) == 0:
            return

        self.boards_x[idx] = 0
        self.boards_o[idx] = 0
        self.meta_x[idx] = 0
        self.meta_o[idx] = 0
        self.meta_draw[idx] = 0
        self.current_player[idx] = 0
        self.active_board[idx] = -1
        self.terminated[idx] = False
        self.winner[idx] = -1

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched step.

        Args:
            actions: (N, 2) int array → [:, 0] = board_idx, [:, 1] = cell_idx.

        Returns:
            rewards: (N,) float32 → from perspective of the player who just moved.
            terminated: (N,) bool.
            info_mask: (N, 9, 9) bool → legal action masks for the next player.

        Terminated games are auto-reset.
        """
        N = self.num_envs
        board_idx = actions[:, 0].astype(np.int32)
        cell_idx = actions[:, 1].astype(np.int32)
        env_idx = np.arange(N)
        cell_bit = np.uint16(1) << cell_idx.astype(np.uint16)

        player_before = self.current_player.copy()

        # --- place marks ---
        is_x = self.current_player == 0
        # Gather the sub-board for each env
        # boards_x[env_idx, board_idx] |= cell_bit  (where is_x)
        cur_x = self.boards_x[env_idx, board_idx]
        cur_o = self.boards_o[env_idx, board_idx]

        self.boards_x[env_idx, board_idx] = np.where(is_x, cur_x | cell_bit, cur_x)
        self.boards_o[env_idx, board_idx] = np.where(~is_x, cur_o | cell_bit, cur_o)

        # --- check sub-board wins ---
        player_bits = np.where(
            is_x,
            self.boards_x[env_idx, board_idx],
            self.boards_o[env_idx, board_idx],
        )
        sub_won = _check_win_vec(player_bits)  # (N,)

        board_bit = np.uint16(1) << board_idx.astype(np.uint16)

        # Update meta boards for wins
        self.meta_x = np.where(sub_won & is_x, self.meta_x | board_bit, self.meta_x)
        self.meta_o = np.where(sub_won & ~is_x, self.meta_o | board_bit, self.meta_o)

        # Check sub-board draws (full but not won)
        occupied = self.boards_x[env_idx, board_idx] | self.boards_o[env_idx, board_idx]
        sub_full = occupied == FULL_BOARD
        sub_draw = sub_full & ~sub_won
        self.meta_draw = np.where(sub_draw, self.meta_draw | board_bit, self.meta_draw)

        # --- check game termination ---
        meta_current = np.where(is_x, self.meta_x, self.meta_o)
        game_won = _check_win_vec(meta_current)

        all_resolved = (self.meta_x | self.meta_o | self.meta_draw) == FULL_BOARD
        game_draw = all_resolved & ~game_won

        just_terminated = (game_won | game_draw) & ~self.terminated
        self.terminated = self.terminated | just_terminated
        self.winner = np.where(
            game_won & just_terminated,
            player_before,
            self.winner,
        ).astype(np.int8)

        # --- determine next active board ---
        target_bit = np.uint16(1) << cell_idx.astype(np.uint16)
        target_resolved = (
            (self.meta_x | self.meta_o | self.meta_draw) & target_bit
        ).astype(bool)

        self.active_board = np.where(
            self.terminated,
            self.active_board,
            np.where(target_resolved, np.int8(-1), cell_idx.astype(np.int8)),
        )

        # --- switch player ---
        self.current_player = (self.current_player ^ 1).astype(np.int8)

        # --- compute rewards (from perspective of player who just moved) ---
        rewards = np.zeros(N, dtype=np.float32)
        rewards = np.where(game_won & just_terminated, 1.0, rewards)
        # (draws get 0)

        # --- auto-reset terminated games ---
        term = self.terminated.copy()
        self.reset(mask=term)

        # --- legal action mask for next state ---
        info_mask = self._legal_action_mask_batch()

        return rewards, term, info_mask

    def legal_action_mask(self) -> np.ndarray:
        """Return (N, 9, 9) bool legal action masks."""
        return self._legal_action_mask_batch()

    def _legal_action_mask_batch(self) -> np.ndarray:
        """Compute (N, 9, 9) bool legal action masks."""
        N = self.num_envs
        mask = np.zeros((N, 9, 9), dtype=np.bool_)

        resolved = self.meta_x | self.meta_o | self.meta_draw  # (N,)

        for b in range(9):
            b_bit = np.uint16(1 << b)
            board_available = ~((resolved & b_bit).astype(bool))  # (N,)

            # Is this the active board (or free choice)?
            is_target = (self.active_board == b) | (self.active_board == -1)
            can_play = board_available & is_target & ~self.terminated  # (N,)

            if not np.any(can_play):
                continue

            occupied = self.boards_x[:, b] | self.boards_o[:, b]  # (N,)
            for c in range(9):
                c_bit = np.uint16(1 << c)
                cell_empty = ~((occupied & c_bit).astype(bool))  # (N,)
                mask[:, b, c] = can_play & cell_empty

        return mask

    def observe(self) -> np.ndarray:
        """Return (N, 6, 9, 9) float32 observations from current player's POV."""
        N = self.num_envs
        obs = np.zeros((N, 6, 9, 9), dtype=np.float32)

        is_x = (self.current_player == 0)[:, np.newaxis]  # (N, 1)

        for b in range(9):
            my_bits = np.where(is_x[:, 0], self.boards_x[:, b], self.boards_o[:, b])
            opp_bits = np.where(is_x[:, 0], self.boards_o[:, b], self.boards_x[:, b])
            for c in range(9):
                c_bit = np.uint16(1 << c)
                obs[:, 0, b, c] = ((my_bits & c_bit) != 0).astype(np.float32)
                obs[:, 1, b, c] = ((opp_bits & c_bit) != 0).astype(np.float32)

        # Plane 2: legal mask
        obs[:, 2] = self._legal_action_mask_batch().astype(np.float32)

        # Planes 3-5: meta-board expanded
        my_meta = np.where(self.current_player == 0, self.meta_x, self.meta_o)
        opp_meta = np.where(self.current_player == 0, self.meta_o, self.meta_x)

        for b in range(9):
            b_bit = np.uint16(1 << b)
            obs[:, 3, b, :] = ((my_meta & b_bit) != 0).astype(np.float32)[:, np.newaxis]
            obs[:, 4, b, :] = ((opp_meta & b_bit) != 0).astype(np.float32)[:, np.newaxis]
            obs[:, 5, b, :] = ((self.meta_draw & b_bit) != 0).astype(np.float32)[:, np.newaxis]

        return obs
