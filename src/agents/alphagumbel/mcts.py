"""AlphaGumbel MCTS for Ultimate Tic-Tac-Toe.

Simplified practical variant:
- Gumbel top-k sampling at the root
- Sequential Halving at the root
- standard PUCT at non-root nodes
- network value instead of rollouts
- improved policy target from completed Q-values

This is not the full paper-faithful Gumbel AlphaZero, but it is a coherent
root-Gumbel variant suitable for small-scale experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from agents.common.network import masked_softmax
from env.logic import legal_action_mask, legal_actions, step_clone
from env.observation import observe
from env.state import UTTTState

if TYPE_CHECKING:
    from agents.common.network import PolicyValueNet


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

C_PUCT = 1.25
ROOT_C_VISIT = 50.0
ROOT_C_SCALE = 1.0


# ---------------------------------------------------------------------------
# Search output
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SearchOutput:
    action: tuple[int, int]
    improved_policy: np.ndarray  # (9, 9)
    visit_policy: np.ndarray     # (9, 9)


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = (
        "state",
        "parent",
        "action",
        "prior",
        "children",
        "visits",
        "value_sum",
        "is_expanded",
    )

    def __init__(
        self,
        state: UTTTState,
        parent: _Node | None = None,
        action: tuple[int, int] | None = None,
        prior: float = 0.0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: list[_Node] = []
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def puct_score(self, parent_visits: int, c_puct: float) -> float:
        u = c_puct * self.prior * math.sqrt(max(parent_visits, 1)) / (1 + self.visits)
        return self.q_value + u


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AlphaGumbelMCTS:
    """Simplified root-Gumbel AlphaZero-style MCTS.

    Network-agnostic: works with any backbone implementing the
    ``agents.common.network.PolicyValueNet`` protocol — currently
    ``AlphaZeroNet`` (CNN), ``UTTTNet`` (PPO weights, same architecture), and
    ``AlphaZeroTransformerNet`` (Transformer). Only the search algorithm is
    Gumbel-specific; the network just needs to emit
    ``(B,9,9) policy_logits`` and ``(B,) value`` from a ``(B,6,9,9)`` obs.
    """

    def __init__(
        self,
        network: "PolicyValueNet",
        *,
        num_simulations: int = 200,
        max_root_actions: int = 16,
        device: torch.device | str = "cpu",
        c_puct: float = C_PUCT,
        c_visit: float = ROOT_C_VISIT,
        c_scale: float = ROOT_C_SCALE,
    ) -> None:
        if num_simulations <= 0:
            raise ValueError("num_simulations must be > 0")
        if max_root_actions <= 0:
            raise ValueError("max_root_actions must be > 0")

        self.network = network
        self.num_simulations = num_simulations
        self.max_root_actions = max_root_actions
        self.device = torch.device(device)
        self.c_puct = c_puct
        self.c_visit = c_visit
        self.c_scale = c_scale

        # Root of the most recent ``search()`` call. Exposed for inspection
        # tools (e.g. the web UI's per-move Q readout); not part of the
        # training-loop contract.
        self._last_root: _Node | None = None

        self.network.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, state: UTTTState) -> tuple[int, int]:
        if state.terminated:
            raise ValueError("Cannot act on a terminal state")
        out = self.search(state, add_gumbel_noise=False)
        return out.action

    def act_with_policy(
        self,
        state: UTTTState,
        *,
        add_gumbel_noise: bool = True,
    ) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
        if state.terminated:
            raise ValueError("Cannot act on a terminal state")
        out = self.search(state, add_gumbel_noise=add_gumbel_noise)
        return out.action, out.improved_policy, out.visit_policy

    def search(
        self,
        state: UTTTState,
        *,
        add_gumbel_noise: bool,
    ) -> SearchOutput:
        root = _Node(state.clone())

        root_logits, root_value, root_probs = self._expand_root(root)

        if root.state.terminated or not root.children:
            raise ValueError("Cannot search from a terminal root state")

        legal_mask = legal_action_mask(root.state)
        legal_flat = np.flatnonzero(legal_mask.reshape(-1))
        num_legal = len(legal_flat)

        m = min(self.max_root_actions, self.num_simulations, num_legal)
        if m <= 0:
            raise RuntimeError("No legal root actions available")

        # Gumbel top-k over legal actions only.
        flat_logits = root_logits.reshape(-1)
        legal_logits = flat_logits[legal_flat]

        if add_gumbel_noise:
            gumbels = np.random.gumbel(size=len(legal_flat)).astype(np.float32)
        else:
            gumbels = np.zeros(len(legal_flat), dtype=np.float32)

        top_order = np.argsort(-(gumbels + legal_logits))[:m]
        candidate_flat = legal_flat[top_order]
        candidate_actions = [(idx // 9, idx % 9) for idx in candidate_flat]
        gumbel_by_action = {
            (idx // 9, idx % 9): float(gumbels[top_order[i]])
            for i, idx in enumerate(candidate_flat)
        }

        child_map: dict[tuple[int, int], _Node] = {
            child.action: child
            for child in root.children
            if child.action is not None
        }
        candidates = [child_map[a] for a in candidate_actions]

        self._run_root_sequential_halving(
            root=root,
            candidates=candidates,
            root_logits=root_logits,
            gumbel_by_action=gumbel_by_action,
        )

        best_child = max(
            candidates,
            key=lambda child: self._root_score(
                root=root,
                action=child.action,  # type: ignore[arg-type]
                child=child,
                root_logits=root_logits,
                gumbel_by_action=gumbel_by_action,
            ),
        )
        best_action = best_child.action
        if best_action is None:
            raise RuntimeError("Best root child has no action")

        visit_policy = self._root_visit_policy(root)
        improved_policy = self._root_improved_policy(
            root=root,
            root_logits=root_logits,
            root_value=root_value,
            root_probs=root_probs,
            legal_mask=legal_mask,
        )

        if (not np.isfinite(improved_policy).all()) or improved_policy.sum() <= 0:
            improved_policy = visit_policy.copy()

        self._last_root = root

        return SearchOutput(
            action=best_action,
            improved_policy=improved_policy,
            visit_policy=visit_policy,
        )

    # ------------------------------------------------------------------
    # Root logic
    # ------------------------------------------------------------------

    def _run_root_sequential_halving(
        self,
        *,
        root: _Node,
        candidates: list[_Node],
        root_logits: np.ndarray,
        gumbel_by_action: dict[tuple[int, int], float],
    ) -> None:
        """Allocate root search budget via simplified Sequential Halving."""
        remaining = list(candidates)
        budget = self.num_simulations

        while budget > 0 and len(remaining) > 1:
            num_actions = len(remaining)
            num_rounds_left = max(1, math.ceil(math.log2(num_actions)))
            sims_per_action = max(1, budget // (num_actions * num_rounds_left))

            used = 0
            for child in remaining:
                reps = min(sims_per_action, budget - used)
                for _ in range(reps):
                    self._simulate_from_child(child)
                used += reps
                if used >= budget:
                    break

            budget -= used
            if budget <= 0:
                break

            # Important: keep using the same Gumbel + logits across rounds.
            remaining.sort(
                key=lambda child: self._root_score(
                    root=root,
                    action=child.action,  # type: ignore[arg-type]
                    child=child,
                    root_logits=root_logits,
                    gumbel_by_action=gumbel_by_action,
                ),
                reverse=True,
            )
            keep = max(1, (len(remaining) + 1) // 2)
            remaining = remaining[:keep]

        # Spend leftover budget round-robin over survivors.
        i = 0
        while budget > 0 and remaining:
            self._simulate_from_child(remaining[i % len(remaining)])
            i += 1
            budget -= 1

    def _root_score(
        self,
        *,
        root: _Node,
        action: tuple[int, int],
        child: _Node,
        root_logits: np.ndarray,
        gumbel_by_action: dict[tuple[int, int], float],
    ) -> float:
        """Root score = gumbel + prior logit + scaled Q."""
        b, c = action
        root_max_visits = max((ch.visits for ch in root.children), default=1)
        sigma_q = (self.c_visit + root_max_visits) * self.c_scale * child.q_value
        return float(gumbel_by_action[action]) + float(root_logits[b, c]) + sigma_q

    def _root_visit_policy(self, root: _Node) -> np.ndarray:
        policy = np.zeros((9, 9), dtype=np.float32)
        total = sum(child.visits for child in root.children)

        if total <= 0:
            prior_total = sum(child.prior for child in root.children)
            if prior_total > 0:
                for child in root.children:
                    if child.action is None:
                        continue
                    b, c = child.action
                    policy[b, c] = child.prior / prior_total
            return policy

        for child in root.children:
            if child.action is None:
                continue
            b, c = child.action
            policy[b, c] = child.visits / total

        return policy

    def _root_improved_policy(
        self,
        *,
        root: _Node,
        root_logits: np.ndarray,
        root_value: float,
        root_probs: np.ndarray,
        legal_mask: np.ndarray,
    ) -> np.ndarray:
        """Construct completed-Q improved policy target."""
        improved = np.zeros((9, 9), dtype=np.float32)

        visited_actions: list[tuple[int, int]] = []
        visited_qs: list[float] = []
        total_visits = 0
        root_max_visits = max((child.visits for child in root.children), default=1)

        for child in root.children:
            if child.action is None:
                continue
            if child.visits > 0:
                visited_actions.append(child.action)
                visited_qs.append(child.q_value)
                total_visits += child.visits

        flat_probs = root_probs.reshape(-1)

        if visited_actions:
            visited_idx = np.array(
                [b * 9 + c for (b, c) in visited_actions],
                dtype=np.int64,
            )
            visited_prior_mass = float(flat_probs[visited_idx].sum())

            if visited_prior_mass > 1e-8:
                weighted_q_sum = float(
                    (flat_probs[visited_idx] * np.asarray(visited_qs, dtype=np.float32)).sum()
                )
                visited_weighted_avg = weighted_q_sum / visited_prior_mass
            else:
                visited_weighted_avg = root_value

            v_mix = (root_value + total_visits * visited_weighted_avg) / (1.0 + total_visits)
        else:
            v_mix = root_value

        completed_q = np.full((9, 9), v_mix, dtype=np.float32)
        for (b, c), q in zip(visited_actions, visited_qs):
            completed_q[b, c] = q

        q_scale = (self.c_visit + root_max_visits) * self.c_scale
        improved_logits = root_logits + q_scale * completed_q

        logits_t = torch.from_numpy(improved_logits).unsqueeze(0)
        mask_t = torch.from_numpy(legal_mask).unsqueeze(0)
        probs = masked_softmax(logits_t, mask_t).squeeze(0).cpu().numpy().astype(np.float32)

        improved[:] = probs
        return improved

    # ------------------------------------------------------------------
    # Tree expansion / simulation
    # ------------------------------------------------------------------

    def _expand_root(self, root: _Node) -> tuple[np.ndarray, float, np.ndarray]:
        logits, values = self._evaluate_batch([root.state])
        root_logits = logits[0]
        root_value = float(values[0])

        mask = legal_action_mask(root.state)
        probs = masked_softmax(
            torch.from_numpy(root_logits).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        ).squeeze(0).cpu().numpy().astype(np.float32)

        self._expand_with_policy(root, probs)
        return root_logits, root_value, probs

    def _simulate_from_child(self, child: _Node) -> None:
        """One simulation starting from a fixed root child."""
        node = child

        while node.is_expanded and not node.state.terminated:
            parent_visits = max(node.visits, 1)
            node = max(
                node.children,
                key=lambda n: n.puct_score(parent_visits, self.c_puct),
            )

        if node.state.terminated:
            value = self._terminal_value(node)
            self._backpropagate(node, value)
            return

        logits, values = self._evaluate_batch([node.state])
        value = float(values[0])

        mask = legal_action_mask(node.state)
        probs = masked_softmax(
            torch.from_numpy(logits[0]).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        ).squeeze(0).cpu().numpy().astype(np.float32)

        self._expand_with_policy(node, probs)

        # Network value is from current-player-to-move perspective.
        # Node backup is from the player who just moved to reach the node.
        self._backpropagate(node, -value)

    def _expand_with_policy(self, node: _Node, policy: np.ndarray) -> None:
        if node.is_expanded or node.state.terminated:
            return

        for action in legal_actions(node.state):
            b, c = action
            child = _Node(
                state=step_clone(node.state, action),
                parent=node,
                action=action,
                prior=float(policy[b, c]),
            )
            node.children.append(child)

        node.is_expanded = True

    def _evaluate_batch(
        self,
        states: list[UTTTState],
    ) -> tuple[np.ndarray, np.ndarray]:
        obs_np = np.stack([observe(s) for s in states], axis=0)
        obs_t = torch.from_numpy(obs_np).to(self.device)

        with torch.no_grad():
            logits_t, values_t = self.network(obs_t)

        logits = logits_t.cpu().numpy().astype(np.float32)
        values = values_t.cpu().numpy().astype(np.float32)
        return logits, values

    def _backpropagate(self, node: _Node | None, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.value_sum += value
            value = -value
            node = node.parent

    @staticmethod
    def _terminal_value(node: _Node) -> float:
        winner = node.state.winner
        if winner == -1:
            return 0.0
        just_moved = 1 - node.state.current_player
        return 1.0 if winner == just_moved else -1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

