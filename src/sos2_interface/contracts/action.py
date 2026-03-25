from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


ActionType = Literal[
    "play_card",
    "use_potion",
    "end_turn",
    "undo_end_turn",
    "combat_select_card",
    "combat_confirm_selection",
    "claim_reward",
    "select_card_reward",
    "skip_card_reward",
    "proceed",
    "event_choose",
    "map_choose",
    "choose_rest_option",
    "shop_purchase",
    "select_card",
    "confirm_selection",
    "cancel_selection",
    "select_relic",
    "skip_relic_selection",
    "claim_treasure_relic",
    "advance_dialogue",
    "noop",
]


class ActionCommand(BaseModel):
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    action_type: ActionType
    card_id: str | None = None
    target_id: str | None = None
    option_index: int | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ActionResult(BaseModel):
    accepted: bool
    executor: str
    action_id: str
    message: str
    emitted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

