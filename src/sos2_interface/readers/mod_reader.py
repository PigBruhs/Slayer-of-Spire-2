from __future__ import annotations

import threading
import time

from sos2_interface.contracts.state import GameStateSnapshot, PlayerState
from sos2_interface.readers.base import GameReader


class ModReader(GameReader):
    """Accepts externally pushed snapshots from a game mod bridge."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_pushed_ms = None
        self._state = self._empty_state(frame_id=0)

    def read_state(self) -> GameStateSnapshot:
        with self._lock:
            return self._state.model_copy(deep=True)

    def ingest_state(self, snapshot: GameStateSnapshot) -> GameStateSnapshot:
        with self._lock:
            normalized = snapshot.model_copy(deep=True)
            normalized.source = "mod"
            if normalized.frame_id <= self._state.frame_id:
                normalized.frame_id = self._state.frame_id + 1
            if normalized.timestamp_ms <= 0:
                normalized.timestamp_ms = int(time.time() * 1000)
            self._state = normalized
            self._last_pushed_ms = normalized.timestamp_ms
            return normalized.model_copy(deep=True)

    def status(self) -> dict[str, str | bool | int | None]:
        with self._lock:
            return {
                "ok": True,
                "mode": "mod",
                "ingest_enabled": True,
                "latest_frame": self._state.frame_id,
                "last_push_ms": self._last_pushed_ms,
            }

    @staticmethod
    def _empty_state(frame_id: int) -> GameStateSnapshot:
        return GameStateSnapshot(
            source="mod",
            frame_id=frame_id,
            timestamp_ms=int(time.time() * 1000),
            in_combat=False,
            in_event=False,
            player=PlayerState(hp=0, max_hp=0, energy=0),
            warnings=["waiting for mod state push"],
        )

