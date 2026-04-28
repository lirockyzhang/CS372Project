"""AlphaZero training loop using the Transformer policy/value backbone.

Structurally identical to ``AlphaZeroTrainer`` (same self-play / train /
evaluate / accept-or-reject cycle, same metrics, same ``TrainingLogger``).
The only differences are:

  * The network is ``AlphaZeroTransformerNet`` instead of ``AlphaZeroNet``.
  * The transformer-specific config is exposed via the constructor and saved
    in ``model_config`` on every checkpoint so eval scripts can reconstruct
    the exact backbone.

Kept as a separate class (rather than parameterising ``AlphaZeroTrainer``)
because the two backbones may want to diverge on optimiser config or
attention-specific tweaks later.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.alphagumbel.mcts import AlphaGumbelMCTS
from agents.alphagumbel.self_play import play_game as gumbel_play_game
from agents.alphazero.mcts import AlphaZeroMCTS
from agents.alphazero.self_play import play_game as puct_play_game
from agents.alphazero.transformer import AlphaZeroTransformerNet
from agents.common.evaluation import (
    evaluate_vs_mcts,
    evaluate_vs_previous,
    evaluate_vs_random,
)
from agents.common.network import PolicyValueNet
from agents.common.replay_buffer import ReplayBuffer
from utils.elo import DEFAULT_K, INITIAL_ELO, update_rating
from utils.training_logger import TrainingLogger


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    network:    AlphaZeroTransformerNet,
    optimizer:  optim.Optimizer,
    buffer:     ReplayBuffer,
    batch_size: int,
    device:     torch.device,
) -> tuple[float, float]:
    """One gradient step. Returns (policy_loss, value_loss)."""
    obs_np, policy_np, value_np = buffer.sample(batch_size)

    obs_t    = torch.from_numpy(obs_np).to(device)
    policy_t = torch.from_numpy(policy_np).to(device)
    value_t  = torch.from_numpy(value_np).to(device)

    logits, value_pred = network(obs_t)

    log_probs   = torch.log_softmax(logits.reshape(-1, 81), dim=-1)
    policy_flat = policy_t.reshape(-1, 81)
    policy_loss = -(policy_flat * log_probs).sum(dim=-1).mean()
    value_loss  = nn.functional.mse_loss(value_pred, value_t)

    loss = 0.5 * policy_loss + 2.0 * value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TransformerTrainer:
    """AlphaZero-style trainer for the Transformer policy/value network."""

    def __init__(
        self,
        run_dir:                Path,
        games_per_iter:         int   = 1_000,
        train_steps:            int   = 500,
        sims:                   int   = 200,
        eval_sims:              int   = 100,
        eval_games:             int   = 40,
        win_threshold:          float = 0.50,
        batch_size:             int   = 256,
        leaf_batch_size:        int   = 8,
        eval_leaf_batch_size:   int   = 8,
        buffer_size:            int   = 500_000,
        min_buffer:             int   = 10_000,
        temp_threshold:         int   = 10,
        lr_init:                float = 1e-3,
        lr_min:                 float = 1e-4,
        total_iters:            int   = 75,
        embed_dim:              int   = 128,
        depth:                  int   = 4,
        num_heads:              int   = 4,
        mlp_ratio:              float = 4.0,
        dropout:                float = 0.0,
        device:                 torch.device | str = "cpu",
        mcts_dataset_path:      str | None = None,
        mcts_refresh_positions: int = 5_000,
        eval_random_games:      int = 20,
        eval_mcts_games:        int = 20,
        eval_mcts_sims:         int = 1_000,
        elo_k:                  float = DEFAULT_K,
        # Search algorithm: "puct" (vanilla AlphaZero) or "gumbel" (Gumbel root)
        search:                 str = "puct",
        max_root_actions:       int = 16,
        # Comparison-study knob -- see AlphaZeroTrainer for full notes.
        eval_at_forwards:       list[int] | None = None,
    ) -> None:
        if search not in ("puct", "gumbel"):
            raise ValueError(f"search must be 'puct' or 'gumbel', got {search!r}")
        self.search           = search
        self.max_root_actions = max_root_actions
        self.eval_at_forwards = (
            sorted(int(x) for x in eval_at_forwards) if eval_at_forwards else None
        )
        self.run_dir                = Path(run_dir)
        self.games_per_iter         = games_per_iter
        self.train_steps            = train_steps
        self.sims                   = sims
        self.eval_sims              = eval_sims
        self.eval_games             = eval_games
        self.win_threshold          = win_threshold
        self.batch_size             = batch_size
        self.leaf_batch_size        = leaf_batch_size
        self.eval_leaf_batch_size   = eval_leaf_batch_size
        self.min_buffer             = min_buffer
        self.temp_threshold         = temp_threshold
        self.lr_init                = lr_init
        self.lr_min                 = lr_min
        self.total_iters            = total_iters
        self.device                 = torch.device(device)

        self.embed_dim = embed_dim
        self.depth     = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout   = dropout

        self.mcts_refresh_positions = mcts_refresh_positions

        self.eval_random_games = eval_random_games
        self.eval_mcts_games   = eval_mcts_games
        self.eval_mcts_sims    = eval_mcts_sims
        self.elo_k             = elo_k

        self._mcts_obs:    np.ndarray | None = None
        self._mcts_policy: np.ndarray | None = None
        self._mcts_value:  np.ndarray | None = None
        if mcts_dataset_path is not None:
            _d = np.load(mcts_dataset_path)
            self._mcts_obs    = _d["obs"].astype(np.float32)
            self._mcts_policy = _d["policy"].astype(np.float32)
            self._mcts_value  = _d["value"].astype(np.float32)
            print(f"MCTS anchor dataset loaded: {len(self._mcts_value):,} positions from {mcts_dataset_path}")

        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.network = AlphaZeroTransformerNet(
            embed_dim = embed_dim,
            depth     = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            dropout   = dropout,
        ).to(self.device)

        self.buffer    = ReplayBuffer(max_size=buffer_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr_init)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_iters, eta_min=lr_min
        )

        self.elo          = INITIAL_ELO
        self.opponent_elo = INITIAL_ELO

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def load_mcts_dataset(
        self,
        path:           str,
        pretrain_steps: int = 0,
        max_positions:  int | None = None,
    ) -> None:
        """Bootstrap the buffer from an MCTS dataset; optionally pre-train."""
        if self._mcts_obs is not None:
            n_avail = len(self._mcts_value)
            n = min(max_positions, n_avail) if max_positions is not None else n_avail
            self.buffer.add_game(
                self._mcts_obs[:n],
                self._mcts_policy[:n],
                self._mcts_value[:n],
            )
        else:
            n = self.buffer.load_from_npz(path, max_positions=max_positions)
        print(f"Loaded {n:,} positions into buffer  (buffer={self.buffer.size:,})")

        if pretrain_steps > 0 and self.buffer.size >= self.batch_size:
            print(f"Pre-training for {pretrain_steps} steps on bootstrap data ...")
            self.network.train()
            policy_losses: list[float] = []
            value_losses:  list[float] = []
            for _ in range(pretrain_steps):
                pl, vl = train_step(
                    self.network, self.optimizer, self.buffer,
                    self.batch_size, self.device,
                )
                policy_losses.append(pl)
                value_losses.append(vl)
            print(
                f"Pre-training done  "
                f"policy_loss={np.mean(policy_losses):.4f}  "
                f"value_loss={np.mean(value_losses):.4f}"
            )

    # ------------------------------------------------------------------
    # Search-agent factories — let callers swap PUCT for Gumbel cleanly
    # ------------------------------------------------------------------

    def _make_selfplay_agent(self, network: PolicyValueNet):
        if self.search == "gumbel":
            return AlphaGumbelMCTS(
                network,
                num_simulations  = self.sims,
                max_root_actions = self.max_root_actions,
                device           = self.device,
            )
        return AlphaZeroMCTS(
            network,
            num_simulations = self.sims,
            batch_size      = self.leaf_batch_size,
            device          = self.device,
        )

    def _make_eval_agent(self, network: PolicyValueNet):
        if self.search == "gumbel":
            return AlphaGumbelMCTS(
                network,
                num_simulations  = self.eval_sims,
                max_root_actions = self.max_root_actions,
                device           = self.device,
            )
        return AlphaZeroMCTS(
            network,
            num_simulations = self.eval_sims,
            batch_size      = self.eval_leaf_batch_size,
            device          = self.device,
        )

    def run(self, start_iter: int = 0) -> None:
        """Run the training loop from *start_iter*, logging metrics each iteration."""
        print(f"\n{'=' * 72}")
        print(f"  AlphaZero Transformer Training  (search={self.search})")
        print(
            f"  device={self.device}  sims={self.sims}  "
            f"games/iter={self.games_per_iter}  steps/iter={self.train_steps}"
        )
        print(
            f"  transformer: embed_dim={self.embed_dim} depth={self.depth} "
            f"heads={self.num_heads} mlp_ratio={self.mlp_ratio} dropout={self.dropout}"
        )
        if self.eval_at_forwards:
            print(f"  panel-eval milestones (NN forwards): {self.eval_at_forwards}")
        print(f"  run dir: {self.run_dir}")
        print(f"{'=' * 72}\n")

        # Cumulatives for the comparison study -- see AZ trainer for notes.
        self._run_start              = time.time()
        self._cum_nn_forwards        = 0
        self._cum_selfplay_positions = 0
        self._next_milestone_idx     = 0

        with TrainingLogger(
            self.run_dir,
            headline_keys=[
                "iter", "policy_loss", "value_loss",
                "win_rate_previous", "win_rate_random", "win_rate_mcts", "elo",
                "cumulative_nn_forwards", "cumulative_time_s",
            ],
            resume=(start_iter > 0),
        ) as logger:
            for iteration in range(start_iter, self.total_iters):
                metrics = self._run_one_iteration(iteration)
                logger.log_iteration(iteration + 1, metrics)

    # ------------------------------------------------------------------
    # Iteration phases
    # ------------------------------------------------------------------

    def _run_one_iteration(self, iteration: int) -> dict[str, Any]:
        iter_start = time.time()
        print(f"--- Iteration {iteration + 1}/{self.total_iters} ---")

        refresh_positions = self._refresh_anchor_data()
        sp_stats          = self._run_self_play()

        # Update cumulative NN-forward + position counters (see AZ trainer).
        iter_forwards = int(
            sp_stats["selfplay_games"]
            * sp_stats["avg_game_length"]
            * self.sims
        )
        self._cum_nn_forwards        += iter_forwards
        self._cum_selfplay_positions += int(sp_stats["selfplay_positions"])

        if self.buffer.size < self.min_buffer:
            print(f"  buffer too small ({self.buffer.size} < {self.min_buffer}), skipping training")
            self._save_checkpoint(iteration, label="selfplay_only")
            return {
                "phase":             "selfplay_only",
                "buffer_size":       self.buffer.size,
                "refresh_positions": refresh_positions,
                **sp_stats,
                "iter_time_min":     (time.time() - iter_start) / 60,
                "cumulative_nn_forwards":        self._cum_nn_forwards,
                "cumulative_selfplay_positions": self._cum_selfplay_positions,
                "cumulative_time_s":             time.time() - self._run_start,
                "sims_per_move":                 self.sims,
            }

        old_network = copy.deepcopy(self.network)
        old_network.eval()

        train_stats = self._run_training()

        # Milestone gating for the strength panel (vs random + vs MCTS@N).
        run_panel = True
        if self.eval_at_forwards is not None:
            if (
                self._next_milestone_idx < len(self.eval_at_forwards)
                and self._cum_nn_forwards
                    >= self.eval_at_forwards[self._next_milestone_idx]
            ):
                run_panel = True
                self._next_milestone_idx += 1
            else:
                run_panel = False
        eval_stats = self._run_evaluation(old_network, run_panel=run_panel)

        accepted = eval_stats["win_rate_previous"] >= self.win_threshold
        if accepted:
            print(f"  new network accepted "
                  f"(win rate {eval_stats['win_rate_previous']:.1%} "
                  f">= {self.win_threshold:.1%})")
            label = "accepted"
        else:
            print(f"  new network rejected "
                  f"(win rate {eval_stats['win_rate_previous']:.1%} "
                  f"< {self.win_threshold:.1%}), reverting")
            self.network.load_state_dict(old_network.state_dict())
            label = "rejected"

        self.elo = update_rating(
            self.elo,
            self.opponent_elo,
            score=eval_stats["score_previous"],
            n_games=self.eval_games,
            k=self.elo_k,
        )
        if accepted:
            self.opponent_elo = self.elo

        self._save_checkpoint(iteration, label=label)

        iter_time = time.time() - iter_start
        print(f"  iteration {iteration + 1} complete in {iter_time/60:.1f} min\n")

        return {
            "phase":             label,
            "buffer_size":       self.buffer.size,
            "refresh_positions": refresh_positions,
            **sp_stats,
            **train_stats,
            **eval_stats,
            "elo":               self.elo,
            "opponent_elo":      self.opponent_elo,
            "iter_time_min":     iter_time / 60,
            "cumulative_nn_forwards":        self._cum_nn_forwards,
            "cumulative_selfplay_positions": self._cum_selfplay_positions,
            "cumulative_time_s":             time.time() - self._run_start,
            "sims_per_move":                 self.sims,
        }

    def _refresh_anchor_data(self) -> int:
        if self._mcts_obs is None:
            return 0
        n_avail = len(self._mcts_value)
        k = min(self.mcts_refresh_positions, n_avail)
        idx = np.random.randint(0, n_avail, size=k)
        self.buffer.add_game(
            self._mcts_obs[idx],
            self._mcts_policy[idx],
            self._mcts_value[idx],
        )
        print(f"  re-injected {k:,} MCTS positions (random sample)  buffer={self.buffer.size:,}")
        return k

    def _run_self_play(self) -> dict[str, Any]:
        self.network.eval()
        agent = self._make_selfplay_agent(self.network)
        play  = gumbel_play_game if self.search == "gumbel" else puct_play_game

        sp_start = time.time()
        positions_added = 0
        for g in range(self.games_per_iter):
            samples = play(agent, temp_threshold=self.temp_threshold)
            obs    = np.stack([s[0] for s in samples])
            policy = np.stack([s[1] for s in samples])
            value  = np.array([s[2] for s in samples], dtype=np.float32)
            self.buffer.add_game(obs, policy, value)
            positions_added += len(samples)

            if (g + 1) % 100 == 0:
                elapsed = time.time() - sp_start
                print(
                    f"  self-play {g+1}/{self.games_per_iter}"
                    f"  positions={positions_added:,}"
                    f"  buffer={self.buffer.size:,}"
                    f"  {(g+1)/elapsed:.2f} games/s"
                )

        sp_time = time.time() - sp_start
        avg_game_len = positions_added / max(self.games_per_iter, 1)
        print(f"  self-play done in {sp_time/60:.1f} min  "
              f"+{positions_added:,} positions  buffer={self.buffer.size:,}")

        return {
            "selfplay_games":         self.games_per_iter,
            "selfplay_positions":     positions_added,
            "selfplay_time_s":        sp_time,
            "selfplay_games_per_sec": self.games_per_iter / max(sp_time, 1e-9),
            "avg_game_length":        avg_game_len,
        }

    def _run_training(self) -> dict[str, Any]:
        one_epoch    = self.buffer.size // self.batch_size
        actual_steps = min(self.train_steps, one_epoch)

        self.network.train()
        train_start = time.time()
        policy_losses: list[float] = []
        value_losses:  list[float] = []
        for _ in range(actual_steps):
            pl, vl = train_step(
                self.network, self.optimizer, self.buffer,
                self.batch_size, self.device,
            )
            policy_losses.append(pl)
            value_losses.append(vl)

        self.scheduler.step()
        lr_now = self.scheduler.get_last_lr()[0]
        train_time = time.time() - train_start

        policy_loss_mean = float(np.mean(policy_losses)) if policy_losses else 0.0
        value_loss_mean  = float(np.mean(value_losses))  if value_losses  else 0.0
        print(
            f"  training done in {train_time:.1f}s"
            f"  steps={actual_steps}/{self.train_steps}"
            f"  policy_loss={policy_loss_mean:.4f}"
            f"  value_loss={value_loss_mean:.4f}"
            f"  lr={lr_now:.2e}"
        )

        return {
            "train_steps_actual": actual_steps,
            "train_time_s":       train_time,
            "policy_loss":        policy_loss_mean,
            "value_loss":         value_loss_mean,
            "lr":                 lr_now,
        }

    def _run_evaluation(
        self,
        old_network: AlphaZeroTransformerNet,
        run_panel:   bool = True,
    ) -> dict[str, Any]:
        """Same semantics as ``AlphaZeroTrainer._run_evaluation``."""
        self.network.eval()
        eval_start = time.time()

        new_agent = self._make_eval_agent(self.network)
        old_agent = self._make_eval_agent(old_network)

        prev = evaluate_vs_previous(new_agent, old_agent, n_games=self.eval_games)

        if run_panel:
            rand = evaluate_vs_random(new_agent, n_games=self.eval_random_games)
            mcts = evaluate_vs_mcts(  new_agent,
                                      mcts_sims=self.eval_mcts_sims,
                                      n_games=self.eval_mcts_games)
            rand_win = rand["win_rate"]
            mcts_win = mcts["win_rate"]
        else:
            rand_win = None
            mcts_win = None

        eval_time = time.time() - eval_start
        if run_panel:
            print(
                f"  evaluation done in {eval_time:.1f}s  "
                f"vs_prev={prev['win_rate']:.1%}  "
                f"vs_random={rand_win:.1%}  "
                f"vs_mcts({self.eval_mcts_sims})={mcts_win:.1%}"
            )
        else:
            print(
                f"  gating eval done in {eval_time:.1f}s  "
                f"vs_prev={prev['win_rate']:.1%}  (panel skipped)"
            )

        return {
            "eval_time_s":       eval_time,
            "win_rate_previous": prev["win_rate"],
            "score_previous":    prev["score"],
            "draws_previous":    prev["draws"],
            "win_rate_random":   rand_win,
            "win_rate_mcts":     mcts_win,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, iteration: int, label: str = "") -> None:
        name = f"iter_{iteration + 1:03d}"
        if label:
            name += f"_{label}"
        path = self.run_dir / f"{name}.pt"
        torch.save(
            {
                "iteration":       iteration + 1,
                "network_state":   self.network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "buffer_size":     self.buffer.size,
                "elo":             self.elo,
                "opponent_elo":    self.opponent_elo,
                "model_type":      "transformer",
                "model_config": {
                    "embed_dim": self.embed_dim,
                    "depth":     self.depth,
                    "num_heads": self.num_heads,
                    "mlp_ratio": self.mlp_ratio,
                    "dropout":   self.dropout,
                },
            },
            path,
        )
        print(f"  checkpoint saved -> {path}")

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint. Optimizer / scheduler state optional so a
        weights-only pretrain checkpoint can be used as a warm-start."""
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network_state"])
        full_resume = "optimizer_state" in ckpt and "scheduler_state" in ckpt
        if full_resume:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.elo          = float(ckpt.get("elo",          INITIAL_ELO))
        self.opponent_elo = float(ckpt.get("opponent_elo", INITIAL_ELO))
        start_iter = int(ckpt.get("iteration", 0))
        if full_resume:
            print(f"Resumed from {path}  (iteration {start_iter}, elo {self.elo:.1f})")
        else:
            print(f"Warm-started (weights only) from {path}")
        return start_iter
