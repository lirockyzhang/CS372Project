"""``State`` base class vendored from https://github.com/pklesk/mcts_numba_cuda.

Vendored at commit c9bb9eff20feadf3a5c64632aa2d1a463b640e47 (2026-03-20).
Copyright (c) Przemysław Klęsk; licensed under CC-BY 4.0.
See ATTRIBUTION.md for the full attribution notice.

Source: ``src/mcts.py`` of the upstream repository — only the ``State`` class
is vendored here; the upstream ``MCTS`` reference implementation (CPU,
single-threaded) is intentionally not copied because we already have our own
CPU MCTS at ``src/agents/mcts/mcts_agent.py``.

Local modifications: the upstream ``from utils import dict_to_str`` import
was removed because ``State`` does not actually use ``dict_to_str``.
"""

from __future__ import annotations


class State:
    """Arbitrary abstract state of some game or sequential decision problem.

    Meant to be inherited and extended to subclasses for use with searches
    conducted by the ``MCTSNC`` class.

    When searches using ``MCTSNC`` are planned, the programmer (while
    inheriting from ``State``) must provide implementations for the following
    non-static methods:

        - ``get_board``
        - ``get_extra_info``

    and the following static ones:

        - ``get_board_shape``
        - ``get_extra_info_memory``
        - ``get_max_actions``
    """

    def __init__(self, parent=None):
        """Constructor of ``State`` instances.

        Should be called as the first line of every subclass constructor:
        ``super().__init__(parent)``.
        """
        self.win_flag = False
        self.n = 0
        self.n_wins = 0
        self.parent = parent
        self.children: dict = {}
        self.outcome_computed = False
        self.outcome = None
        self.turn = 1 if self.parent is None else self.parent.turn
        self.last_action_index = None

    def __str__(self):
        """[To be implemented in subclasses.] String representation of this state."""
        pass

    @staticmethod
    def class_repr():
        """[To be implemented in subclasses.] String repr of this state class."""
        pass

    def _subtree_size(self):
        size = 1
        for key in self.children:
            size += self.children[key]._subtree_size()
        return size

    def _subtree_max_depth(self):
        d = 0
        for key in self.children:
            temp_d = self.children[key]._subtree_max_depth()
            if 1 + temp_d > d:
                d = 1 + temp_d
        return d

    def _subtree_depths(self, d=0, depths=None):
        if depths is None:
            depths = []
        depths.append(d)
        for key in self.children:
            self.children[key]._subtree_depths(d + 1, depths)
        return depths

    def get_turn(self):
        """{-1, 1} indicating whose turn it is."""
        return self.turn

    def take_action(self, action_index):
        """Take action ``action_index`` and return the implied child state."""
        if action_index in self.children:
            return self.children[action_index]
        child = type(self)(self)  # copying constructor
        action_legal = child.take_action_job(action_index)
        if not action_legal:
            return None
        child.last_action_index = action_index
        self.children[action_index] = child
        return child

    def take_action_job(self, action_index):
        """[To be implemented in subclasses.] Returns True iff action legal."""
        pass

    def compute_outcome(self):
        """Compute (or return cached) outcome: -1, 0, +1, or None (ongoing)."""
        if self.outcome_computed:
            return self.outcome
        if self.last_action_index is None:
            return None
        self.outcome = self.compute_outcome_job()
        self.outcome_computed = True
        if self.outcome == -self.turn:
            self.win_flag = True
        return self.outcome

    def compute_outcome_job(self):
        """[To be implemented in subclasses.]"""
        pass

    def get_board(self):
        """[Required for MCTSNC.] (m, n) int8 board representation of this state."""
        pass

    def get_extra_info(self):
        """[Required for MCTSNC.] Optional 1-D int8 array of extra state info."""
        return None

    def expand(self):
        """Generate all children of this state via ``take_action``."""
        if len(self.children) == 0 and self.compute_outcome() is None:
            for action_index in range(self.__class__.get_max_actions()):
                self.take_action(action_index)

    def take_random_action_playout(self):
        """[To be implemented in subclasses.] Random rollout step."""
        pass

    @staticmethod
    def action_name_to_index(action_name):
        """[Optional in subclasses.] Map a string action name to its index."""
        pass

    @staticmethod
    def action_index_to_name(action_index):
        """[Optional in subclasses.] Map an action index to its string name."""
        pass

    @staticmethod
    def get_board_shape():
        """[Required for MCTSNC.] Tuple (rows, cols) of the board shape."""
        pass

    @staticmethod
    def get_extra_info_memory():
        """[Required for MCTSNC.] Number of bytes for the extra-info array."""
        pass

    @staticmethod
    def get_max_actions():
        """[Required for MCTSNC.] Largest branching factor in the game."""
        pass
