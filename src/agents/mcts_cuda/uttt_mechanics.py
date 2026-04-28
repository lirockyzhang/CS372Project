"""Ultimate Tic-Tac-Toe game mechanics for the vendored MCTSNC engine.

Implements the five ``@cuda.jit(device=True)`` functions that ``mctsnc.py``
imports — ``is_action_legal``, ``take_action``, ``legal_actions_playout``,
``take_action_playout``, ``compute_outcome`` — using the (m, n, board,
extra_info, turn, ...) interface defined in pklesk's ``mctsnc_game_mechanics``.

State representation
--------------------
``board`` is the engine's ``int8[9, 9]`` grid. Each cell holds 0 (empty),
+1 (X = maximizing player), or -1 (O = minimizing player).

``extra_info`` is an ``int8[16]`` array holding info that is *not* derivable
from the board alone:

    [0]      active sub-board (-1 = any, 0..8 = specific board the next move
             must be played on; matches src/env/state.UTTTState.active_board)
    [1..2]   meta_x bitfield (9 bits, packed little-endian)
             bit b set <=> X won sub-board b
    [3..4]   meta_o bitfield (same encoding)
    [5..6]   meta_draw bitfield (same encoding)
    [7..15]  reserved / padding

Action encoding
---------------
``action`` is an int in [0, 81). Decoding:

    sub_board = action // 9        (0..8)
    cell      = action %  9        (0..8 within the sub-board)
    meta_row  = sub_board // 3
    meta_col  = sub_board %  3
    cell_row  = cell      // 3
    cell_col  = cell      %  3
    r = meta_row * 3 + cell_row    (0..8 row on the 9x9 board)
    c = meta_col * 3 + cell_col    (0..8 col on the 9x9 board)

Turn convention (must match pklesk)
-----------------------------------
- ``take_action`` / ``take_action_playout`` are called with ``turn`` = the
  player about to move; the placed mark equals ``turn`` (+1 or -1).
- ``compute_outcome`` is called *after* ``take_action`` AND after the engine
  has flipped ``turn``. The just-moved player is therefore ``-turn``.
- "Ongoing" outcomes are signalled by returning any value not in {-1, 0, 1}
  — we use ``2`` to match the convention of the upstream Connect 4 example.
"""

from __future__ import annotations

from numba import cuda, int8, int16, int32


# ----------------------------------------------------------------------
# 9-bit win masks — same 8 patterns used by src/env/state.py.
# Inlined as int constants because Numba can't pull module-level Python
# tuples into device kernels at compile time.
# ----------------------------------------------------------------------
# rows 0..2, cols 0..2, two diagonals
_W0 = 0b000000111   # 7
_W1 = 0b000111000   # 56
_W2 = 0b111000000   # 448
_W3 = 0b001001001   # 73
_W4 = 0b010010010   # 146
_W5 = 0b100100100   # 292
_W6 = 0b100010001   # 273
_W7 = 0b001010100   # 84
_FULL_9 = 0b111111111  # 511 — all 9 sub-boards resolved


# ----------------------------------------------------------------------
# Device-side bitfield helpers (operate on the meta_x / meta_o / meta_draw
# pairs of bytes inside extra_info). All return int32 for safe shifting.
# ----------------------------------------------------------------------

@cuda.jit(device=True)
def _read_meta(extra_info, idx_lo):
    """Read the 9-bit packed meta bitfield starting at ``extra_info[idx_lo]``."""
    lo = int32(extra_info[idx_lo]) & 0xff
    hi = int32(extra_info[idx_lo + 1]) & 1
    return lo | (hi << 8)


@cuda.jit(device=True)
def _set_meta_bit(extra_info, idx_lo, b):
    """Set bit ``b`` (0..8) of the 9-bit meta bitfield at ``extra_info[idx_lo]``."""
    if b < 8:
        extra_info[idx_lo] = int8(int32(extra_info[idx_lo]) | (1 << b))
    else:
        extra_info[idx_lo + 1] = int8(int32(extra_info[idx_lo + 1]) | 1)


@cuda.jit(device=True)
def _check_9bit_win(bits):
    """True iff any of the 8 tic-tac-toe win-masks is fully covered by ``bits``."""
    if (bits & _W0) == _W0: return True
    if (bits & _W1) == _W1: return True
    if (bits & _W2) == _W2: return True
    if (bits & _W3) == _W3: return True
    if (bits & _W4) == _W4: return True
    if (bits & _W5) == _W5: return True
    if (bits & _W6) == _W6: return True
    if (bits & _W7) == _W7: return True
    return False


# ----------------------------------------------------------------------
# Sub-board occupancy probe — read the 9 cells of sub-board ``b`` and pack
# the cells matching ``token`` into a 9-bit value.
# ----------------------------------------------------------------------

@cuda.jit(device=True)
def _sub_board_bits(board, b, token):
    """Pack the cells of sub-board ``b`` matching ``token`` into a 9-bit int."""
    meta_r = b // 3
    meta_c = b % 3
    bits = int32(0)
    # Numba unrolls these constant-bounds loops at JIT time.
    for cell in range(9):
        cell_r = cell // 3
        cell_c = cell % 3
        r = meta_r * 3 + cell_r
        c = meta_c * 3 + cell_c
        if board[r, c] == token:
            bits |= (1 << cell)
    return bits


@cuda.jit(device=True)
def _sub_board_occupied(board, b):
    """Pack non-empty cells of sub-board ``b`` into a 9-bit int (X | O)."""
    meta_r = b // 3
    meta_c = b % 3
    bits = int32(0)
    for cell in range(9):
        cell_r = cell // 3
        cell_c = cell % 3
        r = meta_r * 3 + cell_r
        c = meta_c * 3 + cell_c
        if board[r, c] != 0:
            bits |= (1 << cell)
    return bits


# ----------------------------------------------------------------------
# Public 5-function interface required by mctsnc.py
# ----------------------------------------------------------------------

@cuda.jit(device=True)
def is_action_legal(m, n, board, extra_info, turn, action, legal_actions):
    """Set ``legal_actions[action]`` to True iff this action is legal in this state."""
    b = action // 9
    cell = action - b * 9
    meta_r = b // 3
    meta_c = b - meta_r * 3
    cell_r = cell // 3
    cell_c = cell - cell_r * 3
    r = meta_r * 3 + cell_r
    c = meta_c * 3 + cell_c

    active = extra_info[0]   # -1 or 0..8

    # Cell empty?
    if board[r, c] != 0:
        legal_actions[action] = False
        return

    # Active-board constraint?
    if active != -1 and active != b:
        legal_actions[action] = False
        return

    # Sub-board not yet resolved?
    resolved = _read_meta(extra_info, 1) | _read_meta(extra_info, 3) | _read_meta(extra_info, 5)
    if (resolved >> b) & 1:
        legal_actions[action] = False
        return

    legal_actions[action] = True


@cuda.jit(device=True)
def take_action(m, n, board, extra_info, turn, action):
    """Apply ``action`` to ``board`` / ``extra_info``. Caller passes the *moving* player as ``turn``."""
    b = action // 9
    cell = action - b * 9
    meta_r = b // 3
    meta_c = b - meta_r * 3
    cell_r = cell // 3
    cell_c = cell - cell_r * 3
    r = meta_r * 3 + cell_r
    c = meta_c * 3 + cell_c

    # Place the mark.
    board[r, c] = int8(turn)

    # Did the moving player just complete sub-board b?
    bits_mover = _sub_board_bits(board, b, turn)
    if _check_9bit_win(bits_mover):
        if turn == 1:
            _set_meta_bit(extra_info, 1, b)   # meta_x
        else:
            _set_meta_bit(extra_info, 3, b)   # meta_o
    else:
        # Maybe the sub-board just became full (drawn).
        if _sub_board_occupied(board, b) == _FULL_9:
            _set_meta_bit(extra_info, 5, b)   # meta_draw

    # Determine next active sub-board.
    # The next move must be played in sub-board 'cell'. If that sub-board is
    # already resolved, the next mover gets a free choice (active = -1).
    resolved = _read_meta(extra_info, 1) | _read_meta(extra_info, 3) | _read_meta(extra_info, 5)
    if (resolved >> cell) & 1:
        extra_info[0] = int8(-1)
    else:
        extra_info[0] = int8(cell)


@cuda.jit(device=True)
def legal_actions_playout(m, n, board, extra_info, turn, legal_actions_with_count):
    """Populate ``legal_actions_with_count[0..count-1]`` with legal action indices."""
    active = extra_info[0]
    resolved = _read_meta(extra_info, 1) | _read_meta(extra_info, 3) | _read_meta(extra_info, 5)
    count = int16(0)

    for action in range(81):
        b = action // 9
        cell = action - b * 9

        # Sub-board must not be resolved.
        if (resolved >> b) & 1:
            continue
        # Active-board constraint.
        if active != -1 and active != b:
            continue
        # Cell must be empty.
        meta_r = b // 3
        meta_c = b - meta_r * 3
        cell_r = cell // 3
        cell_c = cell - cell_r * 3
        r = meta_r * 3 + cell_r
        c = meta_c * 3 + cell_c
        if board[r, c] != 0:
            continue

        legal_actions_with_count[count] = int16(action)
        count += int16(1)

    legal_actions_with_count[-1] = count


@cuda.jit(device=True)
def take_action_playout(m, n, board, extra_info, turn, action, action_ord, legal_actions_with_count):
    """Same as ``take_action``. We re-derive ``legal_actions_with_count`` each
    playout step (UTTT's active-board switching makes incremental maintenance
    expensive — recomputing in ``legal_actions_playout`` is simpler and
    correct, mirroring the upstream Gomoku approach)."""
    take_action(m, n, board, extra_info, turn, action)


@cuda.jit(device=True)
def compute_outcome(m, n, board, extra_info, turn, last_action):
    """Return -1/0/+1 for terminal states or any other value (we use 2) for ongoing.

    ``turn`` is the player about to move *next*; the just-moved player is
    therefore ``-turn``. We only need to inspect the meta bitfield for
    ``-turn`` because no one but ``-turn`` modified state since the last call.
    """
    last_token = -turn

    if last_token == 1:
        meta_player = _read_meta(extra_info, 1)   # meta_x
    else:
        meta_player = _read_meta(extra_info, 3)   # meta_o

    if _check_9bit_win(meta_player):
        return last_token

    meta_x_v    = _read_meta(extra_info, 1)
    meta_o_v    = _read_meta(extra_info, 3)
    meta_draw_v = _read_meta(extra_info, 5)
    if (meta_x_v | meta_o_v | meta_draw_v) == _FULL_9:
        return 0      # all 9 sub-boards resolved without a meta-line winner

    return 2          # ongoing
