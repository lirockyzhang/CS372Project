"""UTTT game state using bitboard representation."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

# 8 winning patterns for a 3x3 grid (rows, cols, diags) as 9-bit masks.
# Bit layout for a 3x3 board (row-major):
#   0 1 2
#   3 4 5
#   6 7 8
WIN_MASKS: tuple[int, ...] = (
    0b000_000_111,  # row 0
    0b000_111_000,  # row 1
    0b111_000_000,  # row 2
    0b001_001_001,  # col 0
    0b010_010_010,  # col 1
    0b100_100_100,  # col 2
    0b100_010_001,  # diag  (top-left to bottom-right)
    0b001_010_100,  # anti-diag (top-right to bottom-left)
)

FULL_BOARD = 0b111_111_111  # all 9 bits set


@dataclass(slots=True)
class UTTTState:
    """Mutable Ultimate Tic-Tac-Toe state with bitboard representation.

    Sub-board and cell indices use row-major 0-8 mapping:
        0 1 2
        3 4 5
        6 7 8

    The 3x3 meta-board of sub-boards follows the same layout.
    """

    # Per-sub-board bitboards (9 cells each, bits 0-8).
    boards_x: np.ndarray  # uint16[9]
    boards_o: np.ndarray  # uint16[9]

    # Meta-board: which sub-boards are won/drawn (bits 0-8).
    meta_x: int
    meta_o: int
    meta_draw: int

    current_player: int  # 0 = X, 1 = O
    active_board: int  # 0-8 = forced sub-board, -1 = free choice

    terminated: bool
    winner: int  # -1 = ongoing or draw, 0 = X won, 1 = O won

    def clone(self) -> UTTTState:
        """Fast shallow copy (arrays are copied, ints are immutable)."""
        return UTTTState(
            boards_x=self.boards_x.copy(),
            boards_o=self.boards_o.copy(),
            meta_x=self.meta_x,
            meta_o=self.meta_o,
            meta_draw=self.meta_draw,
            current_player=self.current_player,
            active_board=self.active_board,
            terminated=self.terminated,
            winner=self.winner,
        )


def initial_state() -> UTTTState:
    """Create a fresh game state (empty board, X to move)."""
    return UTTTState(
        boards_x=np.zeros(9, dtype=np.uint16),
        boards_o=np.zeros(9, dtype=np.uint16),
        meta_x=0,
        meta_o=0,
        meta_draw=0,
        current_player=0,
        active_board=-1,
        terminated=False,
        winner=-1,
    )
