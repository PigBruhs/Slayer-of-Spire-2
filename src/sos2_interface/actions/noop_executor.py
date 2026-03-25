from __future__ import annotations

import json
from pathlib import Path

from sos2_interface.contracts.action import ActionCommand, ActionResult


class ActionExecutor:
    def execute(self, action: ActionCommand) -> ActionResult:
        raise NotImplementedError


class NoopActionExecutor(ActionExecutor):
    """Writes action commands to disk so you can inspect what the policy would do."""

    def __init__(self, out_file: str = "runtime/actions.log") -> None:
        self._out_file = Path(out_file)
        self._out_file.parent.mkdir(parents=True, exist_ok=True)

    def execute(self, action: ActionCommand) -> ActionResult:
        with self._out_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(action.model_dump(), ensure_ascii=True) + "\n")
        return ActionResult(
            accepted=True,
            executor="noop",
            action_id=action.action_id,
            message="Action recorded. Plug in SendInput/memory writer later.",
        )

