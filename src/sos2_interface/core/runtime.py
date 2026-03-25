from __future__ import annotations

import threading
import time

from sos2_interface.actions.noop_executor import ActionExecutor
from sos2_interface.contracts.action import ActionCommand, ActionResult
from sos2_interface.contracts.state import GameStateSnapshot
from sos2_interface.readers.base import GameReader


class Runtime:
    def __init__(self, reader: GameReader, executor: ActionExecutor, interval_ms: int = 200) -> None:
        self._reader = reader
        self._executor = executor
        self._interval_ms = max(interval_ms, 50)
        self._latest_state: GameStateSnapshot | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_latest_state(self) -> GameStateSnapshot:
        with self._lock:
            if self._latest_state is None:
                self._latest_state = self._reader.read_state()
            return self._latest_state

    def submit_action(self, action: ActionCommand) -> ActionResult:
        return self._executor.execute(action)

    def can_ingest_state(self) -> bool:
        return callable(getattr(self._reader, "ingest_state", None))

    def ingest_state(self, state: GameStateSnapshot) -> GameStateSnapshot:
        ingest = getattr(self._reader, "ingest_state", None)
        if not callable(ingest):
            raise RuntimeError("Current reader does not support external state ingest")

        accepted = ingest(state)
        with self._lock:
            self._latest_state = accepted
        return accepted

    def reader_status(self) -> dict[str, str | bool]:
        return self._reader.status()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            state = self._reader.read_state()
            with self._lock:
                self._latest_state = state
            time.sleep(self._interval_ms / 1000)

