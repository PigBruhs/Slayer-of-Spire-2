from __future__ import annotations

import json
import time
from pathlib import Path

from sos2_interface.actions.noop_executor import ActionExecutor
from sos2_interface.contracts.action import ActionCommand, ActionResult


class DryRunActionExecutor(ActionExecutor):
    """Persists intended actions as JSONL without touching game input."""

    def __init__(self, out_file: str = "runtime/planner_actions.jsonl") -> None:
        self._out_file = Path(out_file)
        self._out_file.parent.mkdir(parents=True, exist_ok=True)

    def execute(self, action: ActionCommand) -> ActionResult:
        payload = {
            "timestamp_ms": int(time.time() * 1000),
            "action": action.model_dump(),
        }
        with self._out_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

        return ActionResult(
            accepted=True,
            executor="dry_run",
            action_id=action.action_id,
            message="Action planned only; no input was sent to the game.",
        )

