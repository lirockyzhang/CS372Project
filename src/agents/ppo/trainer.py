"""PPO trainer: rollout collection, GAE, clipped PPO updates, and the
end-to-end training loop with periodic strength evaluation.

``PPOTrainer.update()`` does one rollout-plus-update cycle (the unit of work).
``PPOTrainer.run(...)`` is the high-level loop the CLI script calls — it
handles checkpointing, CSV/TensorBoard/console logging via the universal
``TrainingLogger``, periodic 4-metric strength evaluation (vs random, vs
previous snapshot, vs MCTS@1000, plus rolling Elo), and resume support.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.common.evaluation import (
    evaluate_vs_mcts,
    evaluate_vs_previous,
    evaluate_vs_random,
)
from agents.common.network import UTTTNet, masked_log_probs
from env.logic import legal_action_mask
from env.observation import observe
from env.state import UTTTState
from env.vector_env import VectorUTTTEnv
from utils.elo import DEFAULT_K, INITIAL_ELO, update_rating
from utils.training_logger import TrainingLogger


# ---------------------------------------------------------------------------
# Rollout container + GAE
# ---------------------------------------------------------------------------

@dataclass
class RolloutData:
    """Collected rollout data from self-play."""

    obs:         np.ndarray  # (T, N, 6, 9, 9)
    actions:     np.ndarray  # (T, N, 2)
    log_probs:   np.ndarray  # (T, N)
    values:      np.ndarray  # (T, N)
    rewards:     np.ndarray  # (T, N)
    terminated:  np.ndarray  # (T, N) bool
    legal_masks: np.ndarray  # (T, N, 9, 9)
    last_values: np.ndarray  # (N,)


def compute_gae(
    rewards:     np.ndarray,
    values:      np.ndarray,
    terminated:  np.ndarray,
    last_values: np.ndarray,
    gamma:       float = 0.99,
    lam:         float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns for two-player zero-sum self-play.

    Consecutive steps alternate players in canonical observations, so
    ``V(s_{t+1})`` is from the opponent's perspective — negate it to get the
    current player's expected future return:

        delta_t = r_t + gamma * (-V(s_{t+1})) * (1 - done_t) - V(s_t)
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae   = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        next_values       = last_values if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - terminated[t].astype(np.float32)
        delta    = rewards[t] - gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * (-last_gae)
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Adapter: PPO net -> .act(state) interface used by the common eval helpers
# ---------------------------------------------------------------------------


class _NetGreedyAgent:
    """Wraps a ``UTTTNet`` so it speaks the ``act(state)`` protocol the
    ``agents.common.evaluation`` helpers expect.
    """

    def __init__(self, net: UTTTNet, device: torch.device) -> None:
        self.net    = net
        self.device = device

    @torch.no_grad()
    def act(self, state: UTTTState) -> tuple[int, int]:
        obs    = observe(state)
        mask   = legal_action_mask(state)
        obs_t  = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        logits, _ = self.net(obs_t)
        log_probs = masked_log_probs(logits, mask_t)
        flat = int(log_probs.argmax(dim=-1).item())
        return flat // 9, flat % 9


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """PPO self-play trainer using ``VectorUTTTEnv``."""

    def __init__(
        self,
        net: UTTTNet,
        *,
        num_envs:           int   = 512,
        steps_per_rollout:  int   = 128,
        num_epochs:         int   = 4,
        num_minibatches:    int   = 4,
        lr:                 float = 3e-4,
        gamma:              float = 0.99,
        gae_lambda:         float = 0.95,
        clip_eps:           float = 0.2,
        vf_coef:            float = 0.5,
        ent_coef:           float = 0.01,
        max_grad_norm:      float = 0.5,
        device:             str   = "cpu",
    ) -> None:
        self.net               = net
        self.num_envs          = num_envs
        self.steps_per_rollout = steps_per_rollout
        self.num_epochs        = num_epochs
        self.num_minibatches   = num_minibatches
        self.gamma             = gamma
        self.gae_lambda        = gae_lambda
        self.clip_eps          = clip_eps
        self.vf_coef           = vf_coef
        self.ent_coef          = ent_coef
        self.max_grad_norm     = max_grad_norm
        self.device            = torch.device(device)

        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        self.env       = VectorUTTTEnv(num_envs=num_envs)

        # Episode tracking (reset per update)
        self._ep_rewards        = np.zeros(num_envs, dtype=np.float32)
        self._ep_lengths        = np.zeros(num_envs, dtype=np.int32)
        self._completed_rewards: list[float] = []
        self._completed_lengths: list[int]   = []

        # Elo state — used by run() but kept on the trainer so it survives
        # checkpoint resume.
        self.elo          = INITIAL_ELO
        self.opponent_elo = INITIAL_ELO

    # ------------------------------------------------------------------
    # Per-update primitives
    # ------------------------------------------------------------------

    def collect_rollout(self) -> RolloutData:
        """Run self-play for ``steps_per_rollout`` steps and collect transitions."""
        T, N = self.steps_per_rollout, self.num_envs

        obs_buf         = np.zeros((T, N, 6, 9, 9), dtype=np.float32)
        actions_buf     = np.zeros((T, N, 2),       dtype=np.int32)
        log_probs_buf   = np.zeros((T, N),           dtype=np.float32)
        values_buf      = np.zeros((T, N),           dtype=np.float32)
        rewards_buf     = np.zeros((T, N),           dtype=np.float32)
        terminated_buf  = np.zeros((T, N),           dtype=np.bool_)
        legal_masks_buf = np.zeros((T, N, 9, 9),    dtype=np.bool_)

        obs        = self.env.observe()
        legal_mask = self.env._legal_action_mask_batch()

        self.net.eval()
        for t in range(T):
            obs_buf[t]         = obs
            legal_masks_buf[t] = legal_mask

            with torch.no_grad():
                obs_t  = torch.from_numpy(obs).to(self.device)
                mask_t = torch.from_numpy(legal_mask).to(self.device)
                logits, values = self.net(obs_t)
                log_probs_all  = masked_log_probs(logits, mask_t)
                dist           = Categorical(logits=log_probs_all)
                flat_actions   = dist.sample()
                log_prob       = dist.log_prob(flat_actions)

            board_idx = flat_actions.cpu().numpy() // 9
            cell_idx  = flat_actions.cpu().numpy() % 9
            actions   = np.stack([board_idx, cell_idx], axis=1).astype(np.int32)

            actions_buf[t]   = actions
            log_probs_buf[t] = log_prob.cpu().numpy()
            values_buf[t]    = values.cpu().numpy()

            rewards, terminated, next_legal_mask = self.env.step(actions)
            rewards_buf[t]    = rewards
            terminated_buf[t] = terminated

            self._ep_lengths += 1
            self._ep_rewards += rewards
            done_idx = np.where(terminated)[0]
            for i in done_idx:
                self._completed_rewards.append(self._ep_rewards[i])
                self._completed_lengths.append(self._ep_lengths[i])
                self._ep_rewards[i] = 0.0
                self._ep_lengths[i] = 0

            obs        = self.env.observe()
            legal_mask = next_legal_mask

        with torch.no_grad():
            _, last_values = self.net(torch.from_numpy(obs).to(self.device))
        last_values = last_values.cpu().numpy()

        self.net.train()
        return RolloutData(
            obs         = obs_buf,
            actions     = actions_buf,
            log_probs   = log_probs_buf,
            values      = values_buf,
            rewards     = rewards_buf,
            terminated  = terminated_buf,
            legal_masks = legal_masks_buf,
            last_values = last_values,
        )

    def update(self) -> dict[str, float]:
        """Collect a rollout and run PPO update. Returns training stats."""
        self._completed_rewards.clear()
        self._completed_lengths.clear()

        rollout = self.collect_rollout()
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.values,
            rollout.terminated,
            rollout.last_values,
            gamma=self.gamma,
            lam=self.gae_lambda,
        )

        T, N       = self.steps_per_rollout, self.num_envs
        batch_size = T * N

        flat_obs            = torch.from_numpy(rollout.obs.reshape(batch_size, 6, 9, 9)).to(self.device)
        flat_actions        = torch.from_numpy(rollout.actions.reshape(batch_size, 2)).to(self.device)
        flat_old_log_probs  = torch.from_numpy(rollout.log_probs.reshape(batch_size)).to(self.device)
        flat_advantages     = torch.from_numpy(advantages.reshape(batch_size)).to(self.device)
        flat_returns        = torch.from_numpy(returns.reshape(batch_size)).to(self.device)
        flat_legal_masks    = torch.from_numpy(rollout.legal_masks.reshape(batch_size, 9, 9)).to(self.device)

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        flat_action_idx = (flat_actions[:, 0] * 9 + flat_actions[:, 1]).long()

        minibatch_size    = batch_size // self.num_minibatches
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        num_updates       = 0

        for _ in range(self.num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx     = indices[start : start + minibatch_size]
                mb_obs     = flat_obs[mb_idx]
                mb_legal   = flat_legal_masks[mb_idx]
                mb_actions = flat_action_idx[mb_idx]
                mb_old_lp  = flat_old_log_probs[mb_idx]
                mb_adv     = flat_advantages[mb_idx]
                mb_ret     = flat_returns[mb_idx]

                logits, values = self.net(mb_obs)
                log_probs_all  = masked_log_probs(logits, mb_legal)
                new_log_prob   = log_probs_all.gather(1, mb_actions.unsqueeze(1)).squeeze(1)

                ratio       = torch.exp(new_log_prob - mb_old_lp)
                surr1       = ratio * mb_adv
                surr2       = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_ret)
                entropy    = Categorical(logits=log_probs_all).entropy().mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()
                num_updates       += 1

        n_ep = len(self._completed_rewards) if self._completed_rewards else 1
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss":  total_value_loss  / num_updates,
            "entropy":     total_entropy     / num_updates,
            "win_rate":    sum(r > 0 for r in self._completed_rewards) / max(n_ep, 1),
            "mean_ep_len": float(np.mean(self._completed_lengths)) if self._completed_lengths else 0.0,
            "episodes":    len(self._completed_rewards),
        }

    # ------------------------------------------------------------------
    # End-to-end training loop
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        run_dir:           str | Path,
        num_updates:       int,
        save_every:        int                = 50,
        log_every:         int                = 10,
        eval_every:        int                = 50,
        eval_random_games: int                = 20,
        eval_mcts_games:   int                = 20,
        eval_mcts_sims:    int                = 1_000,
        elo_k:             float              = DEFAULT_K,
        resume_from:       str | Path | None  = None,
        # Comparison-study knob: fire the strength panel when cumulative NN
        # forwards crosses each milestone instead of every ``eval_every``
        # updates. None = legacy ``eval_every``-based cadence.
        eval_at_forwards:  list[int] | None   = None,
    ) -> None:
        """Train for ``num_updates`` updates with periodic eval + structured logging."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        start_update     = 0
        previous_snapshot: UTTTNet | None = None
        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            # Accept three checkpoint flavors:
            #   1. PPO native:   {"model": state_dict, "update": ..., "elo": ...}
            #   2. AlphaZero / Transformer pretrain: {"network_state": state_dict, ...}
            #      -- weights only; PPO-specific state (update, elo) starts fresh
            #   3. Raw state_dict
            if isinstance(ckpt, dict) and "model" in ckpt:
                self.net.load_state_dict(ckpt["model"])
                start_update = int(ckpt.get("update", 0))
                self.elo          = float(ckpt.get("elo",          INITIAL_ELO))
                self.opponent_elo = float(ckpt.get("opponent_elo", INITIAL_ELO))
                print(f"Resumed PPO checkpoint from {resume_from} "
                      f"(update {start_update}, elo {self.elo:.1f})")
            elif isinstance(ckpt, dict) and "network_state" in ckpt:
                self.net.load_state_dict(ckpt["network_state"])
                src = ckpt.get("model_type", "?")
                print(f"Warm-started from pretrain checkpoint {resume_from} "
                      f"(arch={src}, weights only -- PPO state fresh)")
            else:
                self.net.load_state_dict(ckpt)
                print(f"Loaded raw state_dict from {resume_from}")

        self.net = self.net.to(self.device)

        print(f"\n{'='*65}")
        print("  PPO Training")
        print(
            f"  device={self.device}  num_envs={self.num_envs}  "
            f"steps_per_rollout={self.steps_per_rollout}"
        )
        if eval_at_forwards:
            print(f"  panel-eval milestones (NN forwards): {eval_at_forwards}")
        print(f"  run dir: {run_dir}")
        print(f"{'='*65}\n")

        # Cumulative comparison-study counters. Reset each ``run()`` call --
        # not persisted across resume by design (documented).
        run_start              = time.time()
        cum_nn_forwards        = 0
        forwards_per_update    = self.num_envs * self.steps_per_rollout
        eval_milestones        = (
            sorted(int(x) for x in eval_at_forwards) if eval_at_forwards else None
        )
        next_milestone_idx     = 0

        with TrainingLogger(
            run_dir,
            headline_keys=[
                "iter", "policy_loss", "value_loss", "entropy",
                "selfplay_win_rate", "win_rate_random", "win_rate_mcts", "elo",
                "cumulative_nn_forwards", "cumulative_time_s",
            ],
            resume=(start_update > 0),
        ) as logger:
            try:
                for i in range(1, num_updates + 1):
                    stats = self.update()
                    update_num = start_update + i

                    # 1 NN forward per env step during rollout (action selection).
                    cum_nn_forwards += forwards_per_update

                    metrics: dict[str, Any] = {
                        "policy_loss":       stats["policy_loss"],
                        "value_loss":        stats["value_loss"],
                        "entropy":           stats["entropy"],
                        "selfplay_win_rate": stats["win_rate"],
                        "mean_ep_len":       stats["mean_ep_len"],
                        "episodes":          stats["episodes"],
                        "cumulative_nn_forwards":        cum_nn_forwards,
                        "cumulative_selfplay_positions": cum_nn_forwards,
                        "cumulative_time_s":             time.time() - run_start,
                        # Eval-panel keys preallocated as None so the CSV
                        # header is locked with all columns on update 1 even
                        # when the first milestone hasn't been crossed yet.
                        "win_rate_previous": None,
                        "score_previous":    None,
                        "draws_previous":    None,
                        "win_rate_random":   None,
                        "win_rate_mcts":     None,
                        "elo":               None,
                    }

                    if i % log_every == 0:
                        print(
                            f"  [update {update_num:>5}]  "
                            f"pi={stats['policy_loss']:.4f}  "
                            f"vf={stats['value_loss']:.4f}  "
                            f"ent={stats['entropy']:.3f}  "
                            f"sp_win={stats['win_rate']:.3f}  "
                            f"len={stats['mean_ep_len']:.1f}"
                        )

                    # Decide whether to fire the strength-panel eval this update.
                    if eval_milestones is not None:
                        do_eval = (
                            next_milestone_idx < len(eval_milestones)
                            and cum_nn_forwards
                                >= eval_milestones[next_milestone_idx]
                        )
                        if do_eval:
                            next_milestone_idx += 1
                    else:
                        do_eval = (i % eval_every == 0)

                    if do_eval:
                        eval_metrics = self._evaluate(
                            previous_snapshot,
                            eval_random_games=eval_random_games,
                            eval_mcts_games=eval_mcts_games,
                            eval_mcts_sims=eval_mcts_sims,
                            elo_k=elo_k,
                        )
                        metrics.update(eval_metrics)
                        # Snapshot *after* eval so the next eval compares to the
                        # network as it was at this checkpoint.
                        previous_snapshot = copy.deepcopy(self.net).eval()

                    if i % save_every == 0:
                        ckpt_path = run_dir / f"ppo_{update_num}.pt"
                        torch.save(
                            {
                                "model":        self.net.state_dict(),
                                "update":       update_num,
                                "elo":          self.elo,
                                "opponent_elo": self.opponent_elo,
                            },
                            ckpt_path,
                        )
                        print(f"  -> saved {ckpt_path.name}")

                    logger.log_iteration(update_num, metrics)

            except KeyboardInterrupt:
                print("\nInterrupted.")

        # Always save a final checkpoint.
        final_path = run_dir / "ppo.pt"
        torch.save(
            {
                "model":        self.net.state_dict(),
                "update":       start_update + num_updates,
                "elo":          self.elo,
                "opponent_elo": self.opponent_elo,
            },
            final_path,
        )
        print(f"Final model saved to {final_path}")

    # ------------------------------------------------------------------
    # Evaluation: 4 strength metrics
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        previous_snapshot: UTTTNet | None,
        *,
        eval_random_games: int,
        eval_mcts_games:   int,
        eval_mcts_sims:    int,
        elo_k:             float,
    ) -> dict[str, Any]:
        """Run the 3 strength matches and update Elo from the gating result."""
        self.net.eval()
        agent = _NetGreedyAgent(self.net, self.device)

        rand = evaluate_vs_random(agent, n_games=eval_random_games)
        mcts = evaluate_vs_mcts(  agent,
                                  mcts_sims=eval_mcts_sims,
                                  n_games=eval_mcts_games)

        if previous_snapshot is None:
            # First eval has no previous snapshot to gate against; report a draw
            # so the Elo update is a no-op.
            prev_score    = 0.5
            prev_win_rate = 0.5
            prev_draws    = 0
        else:
            prev_agent = _NetGreedyAgent(previous_snapshot, self.device)
            prev = evaluate_vs_previous(agent, prev_agent, n_games=eval_random_games)
            prev_score    = prev["score"]
            prev_win_rate = prev["win_rate"]
            prev_draws    = prev["draws"]

        self.elo = update_rating(
            self.elo,
            self.opponent_elo,
            score=prev_score,
            n_games=eval_random_games,
            k=elo_k,
        )
        # PPO has no accept/reject step, so the opponent rating tracks the
        # current rating after each eval — Elo is a smoothed strength signal.
        self.opponent_elo = self.elo

        self.net.train()
        return {
            "win_rate_previous": prev_win_rate,
            "score_previous":    prev_score,
            "draws_previous":    prev_draws,
            "win_rate_random":   rand["win_rate"],
            "win_rate_mcts":     mcts["win_rate"],
            "elo":               self.elo,
        }
