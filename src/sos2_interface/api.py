from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi import Header, HTTPException

from sos2_interface.contracts.action import ActionCommand, ActionResult
from sos2_interface.contracts.state import GameStateSnapshot
from sos2_interface.core.runtime import Runtime
from sos2_interface.policy.rule_assistant import suggest_actions


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="Slayer of Spire 2 Interface Layer", version="0.1.0")
    ingest_token = os.getenv("SOS2_INGEST_TOKEN")

    @app.get("/health")
    def health() -> dict[str, str | bool | int | None]:
        status = runtime.reader_status()
        status["ingest_enabled"] = runtime.can_ingest_state()
        return status

    @app.get("/state", response_model=GameStateSnapshot)
    def state() -> GameStateSnapshot:
        return runtime.get_latest_state()

    @app.get("/suggestions", response_model=list[ActionCommand])
    def suggestions() -> list[ActionCommand]:
        return suggest_actions(runtime.get_latest_state())

    @app.post("/action", response_model=ActionResult)
    def action(command: ActionCommand) -> ActionResult:
        return runtime.submit_action(command)

    @app.post("/ingest/state")
    def ingest_state(
        snapshot: GameStateSnapshot,
        x_sos2_token: str | None = Header(default=None, alias="X-SOS2-Token"),
    ) -> dict[str, int | bool | str]:
        if ingest_token and x_sos2_token != ingest_token:
            raise HTTPException(status_code=401, detail="invalid ingest token")
        if not runtime.can_ingest_state():
            raise HTTPException(status_code=409, detail="ingest is disabled for current reader")

        accepted = runtime.ingest_state(snapshot)
        return {
            "ok": True,
            "source": accepted.source,
            "frame_id": accepted.frame_id,
            "timestamp_ms": accepted.timestamp_ms,
        }

    return app

