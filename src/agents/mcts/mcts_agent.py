"""Monte Carlo Tree Search agent for Ultimate Tic-Tac-Toe.

Works directly on UTTTState using step / step_clone from env.logic —
no Gymnasium wrapper overhead, no NumPy obs tensors.

Public API
----------
MCTSAgent.act(state)              -> (board_idx, cell_idx)
MCTSAgent.act_with_policy(state)  -> ((board_idx, cell_idx), policy_9x9)

act_with_policy() returns a (9, 9) float32 visit-count policy suitable
as a training target for AlphaZero-style networks.
"""

from __future__ import annotations

import math
import random

import numpy as np

from env.logic import check_win, legal_actions, step, step_clone
from env.state import UTTTState

class _Node:
    """MCTS tree node.

    __slots__ cuts per-node memory and speeds up attribute access —matters
    because a single search can create tens of thousands of nodes.
    """

    __slots__ = (
        "state",
        "parent",
        "action",
        "children",
        "visits",
        "value_sum",
        "untried_actions",
    )

    def __init__(
        self,
        state: UTTTState,
        parent: _Node | None = None,
        action: tuple[int, int] | None = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.children: list[_Node] = []
        self.visits: int = 0
        self.value_sum: float = 0.0
        # Shuffled once so expansion order has no directional bias.
        self.untried_actions: list[tuple[int, int]] = (
            [] if state.terminated else legal_actions(state)
        )
        random.shuffle(self.untried_actions)

    def ucb1(self, parent_visits: int, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        return self.value_sum / self.visits + c * math.sqrt(
            math.log(parent_visits) / self.visits
        )


class MCTSAgent:
    """Pure MCTS agent for two-player zero-sum Ultimate Tic-Tac-Toe.

    Parameters
    ----------
    num_simulations:
        Number of select->expand->rollout->backprop cycles per move.
        800 is a reasonable default; raise for stronger play.
    exploration_constant:
        The 'c' in UCB1.  sqrt(2) ~= 1.414 is the theoretical default.
    """

    def __init__(
        self,
        num_simulations: int = 1000,
        exploration_constant: float = math.sqrt(2),
    ) -> None:
        self.num_simulations = num_simulations
        self.c = exploration_constant

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, state: UTTTState) -> tuple[int, int]:
        """Return the best action from *state* (robust-child: most visits)."""
        root = self._build_tree(state)
        best = max(root.children, key=lambda n: n.visits)
        assert best.action is not None
        return best.action

    def act_with_policy(
        self, state: UTTTState
    ) -> tuple[tuple[int, int], np.ndarray]:
        """Return *(action, policy)*.

        *policy* is a (9, 9) float32 array where each entry is the
        fraction of visits that went to that (board, cell).  Entries for
        actions that were never expanded are 0.  Suitable as a
        AlphaZero-style MCTS policy target.
        """
        root = self._build_tree(state)

        policy = np.zeros((9, 9), dtype=np.float32)
        total = sum(child.visits for child in root.children)
        for child in root.children:
            b, c_idx = child.action  # type: ignore[misc]
            policy[b, c_idx] = child.visits / total

        best = max(root.children, key=lambda n: n.visits)
        assert best.action is not None
        return best.action, policy

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def _build_tree(self, state: UTTTState) -> _Node:
        """Run *num_simulations* full MCTS iterations from a cloned root."""
        root = _Node(state.clone())
        for _ in range(self.num_simulations):
            leaf = self._select(root)
            winner = self._rollout(leaf.state)
            self._backpropagate(leaf, winner)
        return root

    def _select(self, node: _Node) -> _Node:
        """Walk down the tree via UCB1 until we reach an expandable node."""
        while not node.state.terminated:
            if node.untried_actions:
                return self._expand(node)
            # Fully expanded: descend to the highest-UCB1 child.
            # Capture parent_visits in the default arg to avoid closure issues.
            pv = node.visits
            node = max(node.children, key=lambda n, p=pv: n.ucb1(p, self.c))
        return node

    def _expand(self, node: _Node) -> _Node:
        """Create one new child for an untried action."""
        action = node.untried_actions.pop()
        child_state = step_clone(node.state, action)
        child = _Node(child_state, parent=node, action=action)
        node.children.append(child)
        return child

    def _rollout(self, state: UTTTState) -> int:
        """Random playout from *state* to a terminal.  Returns winner (-1/0/1).

        Mutates a throw-away clone rather than calling step_clone() each
        move —one allocation for the whole rollout instead of one per step.
        """
        if state.terminated:
            return state.winner
        s = state.clone()
        while not s.terminated:
            actions = legal_actions(s)
            action = random.choice(actions)
            step(s, action)
        return s.winner

    def _backpropagate(self, node: _Node | None, winner: int) -> None:
        """Propagate the rollout result back up to the root.

        value_sum at each node is accumulated from the perspective of the
        player who moved TO reach that node (i.e. 1 - current_player,
        because step() already switched the turn before we stored state).
        """
        while node is not None:
            node.visits += 1
            if winner != -1:
                just_moved = 1 - node.state.current_player
                node.value_sum += 1.0 if winner == just_moved else -1.0
            # draws contribute 0 —no explicit branch needed
            node = node.parent
