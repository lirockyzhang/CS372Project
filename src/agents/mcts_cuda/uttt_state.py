"""Host-side UTTT state wrapper for the vendored MCTSNC engine.

``MCTSNC`` consumes states via three methods:

    state.get_board()        -> int8[9, 9]
    state.get_extra_info()   -> int8[16]
    state.get_turn()         -> {-1, +1}

and three class-level static methods (board shape, extra-info size, max
actions). Internally we own an ``env.state.UTTTState`` (the canonical UTTT
representation, bitboards) and project it into the (board, extra_info)
shape that the device functions in ``uttt_mechanics.py`` expect.

Doing this projection on demand — rather than mirroring every move into a
separate (9, 9) array — keeps a single source of truth (``env.logic.step``)
for the actual UTTT rules.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from env.logic import legal_actions, step
from env.state import UTTTState as _EnvUTTTState
from env.state import initial_state as _initial_state

from ._state_base import State


class UTTTState(State):
    """UTTT state shaped for the MCTSNC engine.

    Use the static getters when constructing the engine:

        ai = MCTSNC(
            UTTTState.get_board_shape(),
            UTTTState.get_extra_info_memory(),
            UTTTState.get_max_actions(),
            ...
        )

    Use the per-instance getters when calling ``ai.run(...)``:

        best = ai.run(state.get_board(), state.get_extra_info(), state.get_turn())
    """

    BOARD_SHAPE       = (9, 9)
    EXTRA_INFO_MEMORY = 16
    MAX_ACTIONS       = 81

    def __init__(self, parent=None):
        super().__init__(parent)
        # The canonical state lives in env.state.UTTTState (bitboards).
        self._inner: _EnvUTTTState = _initial_state()
        # Pklesk's State.turn is +1 / -1; env's current_player is 0 (X) / 1 (O).
        self.turn = self._turn_from_inner()

    # ------------------------------------------------------------------
    # Required by MCTSNC (static class info)
    # ------------------------------------------------------------------

    @staticmethod
    def class_repr() -> str:
        return "UTTT_9x9"

    @staticmethod
    def get_board_shape() -> tuple[int, int]:
        return UTTTState.BOARD_SHAPE

    @staticmethod
    def get_extra_info_memory() -> int:
        return UTTTState.EXTRA_INFO_MEMORY

    @staticmethod
    def get_max_actions() -> int:
        return UTTTState.MAX_ACTIONS

    # ------------------------------------------------------------------
    # Required by MCTSNC (per-instance projections)
    # ------------------------------------------------------------------

    def get_board(self) -> np.ndarray:
        """Project the bitboard state into a ``(9, 9) int8`` grid.

        Cell value: 0 = empty, +1 = X, -1 = O. Layout matches the action
        decoding used by ``uttt_mechanics`` — sub-board ``b`` lives at
        meta-row ``b // 3`` and meta-col ``b % 3``.
        """
        board = np.zeros((9, 9), dtype=np.int8)
        for b in range(9):
            meta_r = b // 3
            meta_c = b % 3
            x_bits = int(self._inner.boards_x[b])
            o_bits = int(self._inner.boards_o[b])
            for cell in range(9):
                cell_r = cell // 3
                cell_c = cell % 3
                r = meta_r * 3 + cell_r
                c = meta_c * 3 + cell_c
                bit = 1 << cell
                if x_bits & bit:
                    board[r, c] = 1
                elif o_bits & bit:
                    board[r, c] = -1
        return board

    def get_extra_info(self) -> np.ndarray:
        """Pack ``active_board`` and the three meta bitfields into ``int8[16]``."""
        info = np.zeros(16, dtype=np.int8)
        info[0] = self._inner.active_board   # -1 or 0..8 (fits int8)

        for idx_lo, value in (
            (1, self._inner.meta_x),
            (3, self._inner.meta_o),
            (5, self._inner.meta_draw),
        ):
            v  = int(value) & 0x1FF          # 9 bits
            lo = v & 0xFF                     # 0..255
            # NumPy doesn't auto-wrap 128..255 into signed int8 (-128..-1)
            # the way Numba's int8() cast does; do the two's-complement
            # conversion explicitly so the engine sees the same bit pattern.
            info[idx_lo]     = np.int8(lo if lo < 128 else lo - 256)
            info[idx_lo + 1] = np.int8((v >> 8) & 1)

        return info

    def get_turn(self) -> int:
        return self.turn

    # ------------------------------------------------------------------
    # Driver-loop API — used by self_play.play_games_cuda between
    # successive MCTSNC.run() calls to advance the game.
    # ------------------------------------------------------------------

    def take_action_job(self, action_index: int) -> bool:
        """Apply action ``action_index`` (0..80) in place. Returns False if illegal."""
        if self._inner.terminated:
            return False
        b = action_index // 9
        c = action_index %  9
        try:
            step(self._inner, (b, c))
        except ValueError:
            return False
        self.turn = self._turn_from_inner()
        # Reset cached outcome so compute_outcome re-checks after the move.
        self.outcome_computed = False
        self.outcome = None
        self.last_action_index = action_index
        return True

    def compute_outcome_job(self) -> Optional[int]:
        """Return -1 (O wins), 0 (draw), +1 (X wins), or None (ongoing)."""
        if not self._inner.terminated:
            return None
        w = self._inner.winner
        if w == -1:
            return 0
        if w == 0:
            return 1     # X won  → max player
        if w == 1:
            return -1    # O won  → min player
        return None

    # ------------------------------------------------------------------
    # Convenience helpers for the driver loop
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self._inner.terminated

    @property
    def current_player_zero_one(self) -> int:
        """The env-style current_player: 0 = X, 1 = O. Useful when stitching
        outcomes into the (obs, policy, value) sample buffer."""
        return self._inner.current_player

    def legal_action_indices(self) -> list[int]:
        """Return the legal actions as a flat list of indices in [0, 81)."""
        return [b * 9 + c for (b, c) in legal_actions(self._inner)]

    @property
    def env_state(self) -> _EnvUTTTState:
        """Expose the underlying env state for ``observe()`` and similar callers."""
        return self._inner

    def __str__(self) -> str:
        from env.render import render
        return render(self._inner)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _turn_from_inner(self) -> int:
        # env: current_player 0 = X (max), 1 = O (min) → pklesk: +1 / -1
        return 1 if self._inner.current_player == 0 else -1
