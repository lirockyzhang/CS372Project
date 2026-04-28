"""Pure-Python mirror of ``uttt_mechanics.py`` for parity testing.

Each function below is a verbatim Python translation of the corresponding
``@cuda.jit(device=True)`` function in ``uttt_mechanics.py`` — same indices,
same constants, same control flow, just without the Numba decorators and
typed casts. Importing this module does **not** require numba.

Used by ``tests/test_uttt_parity.py`` to compare the algorithm against
``env.logic`` (the bitboard-based source of truth) on millions of random
moves. If you change ``uttt_mechanics.py``, change this file too — they
must stay in lockstep.

DO NOT USE THIS IN PRODUCTION CODE. It exists for testing only.
"""

from __future__ import annotations

import numpy as np

# Mirror of the 8 win-masks in uttt_mechanics.py (rows, cols, diags on a 3x3).
_W = (
    0b000000111,  # row 0
    0b000111000,  # row 1
    0b111000000,  # row 2
    0b001001001,  # col 0
    0b010010010,  # col 1
    0b100100100,  # col 2
    0b100010001,  # diag
    0b001010100,  # anti-diag
)
_FULL_9 = 0b111111111   # 511 — all 9 sub-boards resolved


# ----------------------------------------------------------------------
# Bitfield helpers (mirror of uttt_mechanics._read_meta / _set_meta_bit)
# ----------------------------------------------------------------------

def _read_meta(extra_info: np.ndarray, idx_lo: int) -> int:
    lo = int(extra_info[idx_lo]) & 0xff
    hi = int(extra_info[idx_lo + 1]) & 1
    return lo | (hi << 8)


# Bit b of a uint8 written as a signed int8: bit 7 = 128 wraps to -128 in
# two's-complement. Numba's ``int8(1 << 7)`` does this implicitly; NumPy's
# ``np.int8(128)`` raises OverflowError, so we use a precomputed table.
_BIT_AS_INT8 = np.array([1, 2, 4, 8, 16, 32, 64, -128], dtype=np.int8)


def _set_meta_bit(extra_info: np.ndarray, idx_lo: int, b: int) -> None:
    if b < 8:
        extra_info[idx_lo] = extra_info[idx_lo] | _BIT_AS_INT8[b]
    else:
        extra_info[idx_lo + 1] = extra_info[idx_lo + 1] | np.int8(1)


def _check_9bit_win(bits: int) -> bool:
    for m in _W:
        if (bits & m) == m:
            return True
    return False


def _sub_board_bits(board: np.ndarray, b: int, token: int) -> int:
    meta_r = b // 3
    meta_c = b %  3
    bits = 0
    for cell in range(9):
        cell_r = cell // 3
        cell_c = cell %  3
        r = meta_r * 3 + cell_r
        c = meta_c * 3 + cell_c
        if int(board[r, c]) == token:
            bits |= (1 << cell)
    return bits


def _sub_board_occupied(board: np.ndarray, b: int) -> int:
    meta_r = b // 3
    meta_c = b %  3
    bits = 0
    for cell in range(9):
        cell_r = cell // 3
        cell_c = cell %  3
        r = meta_r * 3 + cell_r
        c = meta_c * 3 + cell_c
        if int(board[r, c]) != 0:
            bits |= (1 << cell)
    return bits


# ----------------------------------------------------------------------
# Public mirrors of the 5 device functions
# ----------------------------------------------------------------------

def is_action_legal_mirror(board: np.ndarray, extra_info: np.ndarray,
                           action: int) -> bool:
    b = action // 9
    cell = action %  9
    meta_r = b // 3
    meta_c = b %  3
    cell_r = cell // 3
    cell_c = cell %  3
    r = meta_r * 3 + cell_r
    c = meta_c * 3 + cell_c

    active = int(extra_info[0])

    if int(board[r, c]) != 0:
        return False
    if active != -1 and active != b:
        return False
    resolved = _read_meta(extra_info, 1) | _read_meta(extra_info, 3) | _read_meta(extra_info, 5)
    if (resolved >> b) & 1:
        return False
    return True


def take_action_mirror(board: np.ndarray, extra_info: np.ndarray,
                       turn: int, action: int) -> None:
    b = action // 9
    cell = action %  9
    meta_r = b // 3
    meta_c = b %  3
    cell_r = cell // 3
    cell_c = cell %  3
    r = meta_r * 3 + cell_r
    c = meta_c * 3 + cell_c

    board[r, c] = np.int8(turn)

    bits_mover = _sub_board_bits(board, b, turn)
    if _check_9bit_win(bits_mover):
        if turn == 1:
            _set_meta_bit(extra_info, 1, b)
        else:
            _set_meta_bit(extra_info, 3, b)
    elif _sub_board_occupied(board, b) == _FULL_9:
        _set_meta_bit(extra_info, 5, b)

    resolved = _read_meta(extra_info, 1) | _read_meta(extra_info, 3) | _read_meta(extra_info, 5)
    if (resolved >> cell) & 1:
        extra_info[0] = np.int8(-1)
    else:
        extra_info[0] = np.int8(cell)


def legal_actions_playout_mirror(board: np.ndarray, extra_info: np.ndarray) -> list[int]:
    active = int(extra_info[0])
    resolved = _read_meta(extra_info, 1) | _read_meta(extra_info, 3) | _read_meta(extra_info, 5)
    out: list[int] = []
    for action in range(81):
        b = action // 9
        cell = action %  9
        if (resolved >> b) & 1:
            continue
        if active != -1 and active != b:
            continue
        meta_r = b // 3
        meta_c = b %  3
        cell_r = cell // 3
        cell_c = cell %  3
        r = meta_r * 3 + cell_r
        c = meta_c * 3 + cell_c
        if int(board[r, c]) != 0:
            continue
        out.append(action)
    return out


def compute_outcome_mirror(board: np.ndarray, extra_info: np.ndarray,
                           turn: int, last_action: int) -> int:
    last_token = -turn
    if last_token == 1:
        meta_player = _read_meta(extra_info, 1)
    else:
        meta_player = _read_meta(extra_info, 3)

    if _check_9bit_win(meta_player):
        return last_token

    meta_x = _read_meta(extra_info, 1)
    meta_o = _read_meta(extra_info, 3)
    meta_d = _read_meta(extra_info, 5)
    if (meta_x | meta_o | meta_d) == _FULL_9:
        return 0
    return 2  # ongoing
