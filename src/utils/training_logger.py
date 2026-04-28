"""Universal training metrics logger.

Every trainer (AlphaZero, AlphaGumbel, Transformer, PPO) drives the same
``TrainingLogger`` so logs are uniformly shaped across runs.

Per call to ``log_iteration(iter_num, metrics)`` the logger writes:

    * One row to ``<run_dir>/train_log.csv`` (flat, one column per metric key)
    * One scalar per metric to ``<run_dir>/tensorboard/`` (if torch's
      tensorboard support is importable)
    * One human-readable line to stdout summarising the headline metrics

The CSV header is locked on the first row, so trainers must log the same
schema every iteration.  Resume mode appends to the existing file without
re-writing the header.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except Exception:  # tensorboard pkg not installed, torch.utils.tensorboard import fails
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False


class TrainingLogger:
    """CSV + TensorBoard + console metrics sink driven by trainers."""

    def __init__(
        self,
        run_dir:        str | Path,
        *,
        csv_name:       str        = "train_log.csv",
        tb_subdir:      str        = "tensorboard",
        headline_keys:  list[str] | None = None,
        resume:         bool       = False,
    ) -> None:
        self.run_dir       = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path      = self.run_dir / csv_name
        self.headline_keys = headline_keys or ["iter"]
        self._resume       = resume

        # CSV is opened lazily on the first log call so we capture every key
        # the trainer emits in the header row.
        self._csv_file:        Any         = None
        self._csv_writer:      Any         = None
        self._csv_fieldnames:  list[str] | None = None

        self._tb: Any = None
        if _TB_AVAILABLE:
            self._tb = SummaryWriter(log_dir=str(self.run_dir / tb_subdir))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_iteration(self, iter_num: int, metrics: Mapping[str, Any]) -> None:
        """Append one iteration's metrics to every sink."""
        row = {"iter": iter_num, **{k: _format_csv(v) for k, v in metrics.items()}}
        self._write_csv(row)
        self._write_tb(iter_num, metrics)
        self._print_console(row)

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if self._tb is not None:
            self._tb.close()
            self._tb = None

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    # ------------------------------------------------------------------
    # Sinks
    # ------------------------------------------------------------------

    def _write_csv(self, row: dict[str, Any]) -> None:
        if self._csv_writer is None:
            self._csv_fieldnames = list(row.keys())
            mode = "a" if (self._resume and self.csv_path.exists()) else "w"
            self._csv_file = open(self.csv_path, mode, newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fieldnames)
            if mode == "w":
                self._csv_writer.writeheader()
        # Pad missing keys, drop unknown ones, so the schema stays stable.
        safe = {k: row.get(k, "") for k in (self._csv_fieldnames or [])}
        self._csv_writer.writerow(safe)
        self._csv_file.flush()

    def _write_tb(self, iter_num: int, metrics: Mapping[str, Any]) -> None:
        if self._tb is None:
            return
        for k, v in metrics.items():
            if isinstance(v, bool):
                self._tb.add_scalar(k, int(v), iter_num)
            elif isinstance(v, (int, float)):
                self._tb.add_scalar(k, v, iter_num)

    def _print_console(self, row: dict[str, Any]) -> None:
        bits: list[str] = []
        for k in self.headline_keys:
            if k not in row:
                continue
            v = row[k]
            if isinstance(v, float):
                bits.append(f"{k}={v:.3f}")
            else:
                bits.append(f"{k}={v}")
        print("[" + " | ".join(bits) + "]", file=sys.stdout, flush=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _format_csv(v: Any) -> Any:
    """Pretty-format floats so the CSV stays readable; pass other types through."""
    if isinstance(v, float):
        return f"{v:.6g}"
    return v
