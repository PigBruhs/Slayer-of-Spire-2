from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


ActionType = Literal[
    # Core combat actions.
    "play_card",
    "use_potion",
    "end_turn",
    "undo_end_turn",
    "combat_select_card",
    "combat_confirm_selection",
    # Rewards / transitions.
    "claim_reward",
    "rewards_claim",
    "select_card_reward",
    "rewards_pick_card",
    "skip_card_reward",
    "rewards_skip_card",
    "proceed",
    "proceed_to_map",
    # Event / map aliases.
    "event_choose",
    "choose_event_option",
    "map_choose",
    "choose_map_node",
    "choose_rest_option",
    "rest_choose_option",
    "shop_purchase",
    "deck_select_card",
    "select_card",
    "deck_confirm_selection",
    "confirm_selection",
    "deck_cancel_selection",
    "cancel_selection",
    "relic_select",
    "select_relic",
    "relic_skip",
    "skip_relic_selection",
    "treasure_claim_relic",
    "claim_treasure_relic",
    "event_advance_dialogue",
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

