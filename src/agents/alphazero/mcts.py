"""AlphaZero MCTS with PUCT selection and batched leaf evaluation.

Differences from vanilla MCTSAgent:
  - No random rollouts —network evaluates each new leaf.
  - PUCT replaces UCB1, using the network policy prior P(s,a).
  - Virtual losses allow K leaves to be selected before any network call,
    so observations can be batched into a single forward pass.
  - Dirichlet noise is mixed into the root prior during self-play.

Public API
----------
  AlphaZeroMCTS(network, num_simulations, batch_size, device)
  agent.act(state)                    -> (board_idx, cell_idx)
  agent.act_with_policy(state)        -> ((board_idx, cell_idx), policy_9x9)
  agent.act_with_policy(state, noise) -> same, with Dirichlet at root
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from agents.common.network import masked_softmax
from env.logic import check_win, legal_actions, legal_action_mask, step, step_clone
from env.observation import observe
from env.state import UTTTState

if TYPE_CHECKING:
    from agents.common.network import PolicyValueNet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C_PUCT        = 1.0    # exploration constant in PUCT formula
DIRICHLET_EPS = 0.25   # weight of noise mixed into root prior
DIRICHLET_ALPHA = 0.12 # concentration parameter

# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class _AZNode:
    """MCTS node for AlphaZero search.

    __slots__ keeps per-node memory tight —searches can create tens of
    thousands of nodes per move.
    """

    __slots__ = (
        "state",
        "parent",
        "action",
        "children",
        "visits",
        "value_sum",
        "virtual_loss",
        "prior",
        "is_expanded",
    )

    def __init__(
        self,
        state:  UTTTState,
        parent: _AZNode | None = None,
        action: tuple[int, int] | None = None,
        prior:  float = 0.0,
    ) -> None:
        self.state        = state
        self.parent       = parent
        self.action       = action
        self.prior        = prior          # P(s, a) from network policy
        self.children:    list[_AZNode] = []
        self.visits:      int   = 0
        self.value_sum:   float = 0.0
        self.virtual_loss: int  = 0        # temporarily subtracted during batching
        self.is_expanded: bool  = False

    @property
    def q_value(self) -> float:
        """Mean value estimate, accounting for virtual losses."""
        total = self.visits + self.virtual_loss
        if total == 0:
            return 0.0
        return (self.value_sum - self.virtual_loss) / total

    def puct_score(self, parent_visits: int) -> float:
        """PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))."""
        u = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visits + self.virtual_loss)
        return self.q_value + u


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AlphaZeroMCTS:
    """AlphaZero MCTS agent (PUCT selection).

    Network-agnostic: works with any backbone implementing the
    ``agents.common.network.PolicyValueNet`` protocol — currently
    ``AlphaZeroNet`` (CNN), ``UTTTNet`` (PPO weights, same architecture), and
    ``AlphaZeroTransformerNet`` (Transformer).

    Parameters
    ----------
    network:         PolicyValueNet in eval mode.
    num_simulations: Total MCTS simulations per move.
    batch_size:      Leaves to accumulate before each network call.
    device:          torch device for network inference.
    """

    def __init__(
        self,
        network:         "PolicyValueNet",
        num_simulations: int = 200,
        batch_size:      int = 8,
        device:          torch.device | str = "cpu",
    ) -> None:
        self.network         = network
        self.num_simulations = num_simulations
        self.batch_size      = batch_size
        self.device          = torch.device(device)
        self.network.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, state: UTTTState) -> tuple[int, int]:
        """Return the most-visited action from *state*."""
        root = self._build_tree(state, add_noise=False)
        return max(root.children, key=lambda n: n.visits).action  # type: ignore[return-value]

    def act_with_policy(
        self,
        state:     UTTTState,
        add_noise: bool = False,
    ) -> tuple[tuple[int, int], np.ndarray]:
        """Return *(action, policy_9x9)*.

        *policy* is the normalised visit-count distribution —a (9, 9)
        float32 array suitable as an AlphaZero training target.

        Args:
            state:     Current game state (not mutated).
            add_noise: If True, mix Dirichlet noise into the root prior.
                       Set True during self-play, False during evaluation.
        """
        root = self._build_tree(state, add_noise=add_noise)

        policy = np.zeros((9, 9), dtype=np.float32)
        total  = sum(c.visits for c in root.children)
        for child in root.children:
            b, c = child.action  # type: ignore[misc]
            policy[b, c] = child.visits / total

        best_action = max(root.children, key=lambda n: n.visits).action
        return best_action, policy  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self, state: UTTTState, add_noise: bool) -> _AZNode:
        """Run *num_simulations* iterations and return the root node."""
        root = _AZNode(state.clone())
        self._expand(root)
        if add_noise:
            self._add_dirichlet_noise(root)

        sims_done = 0
        while sims_done < self.num_simulations:
            batch_n = min(self.batch_size, self.num_simulations - sims_done)

            # --- select a batch of leaves ------------------------------------
            leaves: list[_AZNode] = []
            for _ in range(batch_n):
                leaf = self._select(root)
                leaf.virtual_loss += 1
                leaves.append(leaf)

            # --- separate terminal from non-terminal leaves ------------------
            to_evaluate: list[_AZNode] = []
            for leaf in leaves:
                if leaf.state.terminated:
                    # terminal value known —no network call needed
                    value = self._terminal_value(leaf)
                    self._backpropagate(leaf, value)
                    leaf.virtual_loss -= 1
                else:
                    to_evaluate.append(leaf)

            # --- batch network inference for non-terminal leaves -------------
            if to_evaluate:
                obs_np = np.stack([observe(n.state) for n in to_evaluate])
                obs_t  = torch.from_numpy(obs_np).to(self.device)

                with torch.no_grad():
                    logits, values = self.network(obs_t)

                masks = torch.from_numpy(
                    np.stack([legal_action_mask(n.state) for n in to_evaluate])
                ).to(self.device)

                probs_t = masked_softmax(logits, masks)  # (K, 9, 9)
                probs_np = probs_t.cpu().numpy()
                values_np = values.cpu().numpy()

                for i, leaf in enumerate(to_evaluate):
                    self._expand_with_prior(leaf, probs_np[i])
                    self._backpropagate(leaf, -float(values_np[i]))
                    leaf.virtual_loss -= 1

            sims_done += batch_n

        return root

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def _expand(self, node: _AZNode) -> None:
        """Expand root on first visit using network —required before noise."""
        if node.is_expanded or node.state.terminated:
            return
        obs_t = torch.from_numpy(observe(node.state)).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(
            legal_action_mask(node.state).astype(np.float32)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.network(obs_t)

        probs = masked_softmax(logits, mask_t.bool()).squeeze(0).cpu().numpy()
        self._expand_with_prior(node, probs)

    def _expand_with_prior(self, node: _AZNode, policy: np.ndarray) -> None:
        """Create children for all legal actions, seeded with *policy* priors."""
        if node.is_expanded or node.state.terminated:
            return
        for b, c in legal_actions(node.state):
            child = _AZNode(
                state  = step_clone(node.state, (b, c)),
                parent = node,
                action = (b, c),
                prior  = float(policy[b, c]),
            )
            node.children.append(child)
        node.is_expanded = True

    def _select(self, root: _AZNode) -> _AZNode:
        """Walk down the tree via PUCT until an unexpanded or terminal node."""
        node = root
        while node.is_expanded and not node.state.terminated:
            pv   = node.visits + node.virtual_loss
            node = max(node.children, key=lambda n, p=pv: n.puct_score(p))
        return node

    def _backpropagate(self, node: _AZNode | None, value: float) -> None:
        """Propagate *value* from *node* up to the root.

        Value is from the perspective of the player who just moved to reach
        each node, so it flips sign at each level.
        """
        while node is not None:
            node.visits    += 1
            node.value_sum += value
            value           = -value
            node            = node.parent

    @staticmethod
    def _terminal_value(node: _AZNode) -> float:
        """Return the game outcome from the perspective of the node's mover."""
        winner = node.state.winner
        if winner == -1:
            return 0.0
        # The player who moved TO reach this node is 1 - current_player
        # (step() already switched the turn).
        just_moved = 1 - node.state.current_player
        return 1.0 if winner == just_moved else -1.0

    @staticmethod
    def _add_dirichlet_noise(root: _AZNode) -> None:
        """Mix Dirichlet noise into root children priors for exploration."""
        if not root.children:
            return
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root.children))
        for child, eta in zip(root.children, noise):
            child.prior = (1 - DIRICHLET_EPS) * child.prior + DIRICHLET_EPS * eta


