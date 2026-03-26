from __future__ import annotations

from pathlib import Path
from typing import Any


class TensorboardLogger:
    """Best-effort TensorBoard scalar logger.

    If tensorboardX is not installed, logging is disabled without crashing runtime.
    """

    def __init__(self, log_dir: str | None) -> None:
        self._enabled = False
        self._writer: Any | None = None
        self._log_dir = log_dir

        if not log_dir:
            return

        try:
            from tensorboardX import SummaryWriter  # type: ignore
        except Exception:
            return

        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(logdir=str(path))
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if not self._enabled or self._writer is None:
            return
        self._writer.add_scalar(tag, float(value), int(step))

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
        self._enabled = False

