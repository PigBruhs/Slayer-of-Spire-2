from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


IntentType = Literal["attack", "defend", "buff", "debuff", "multi_attack", "unknown"]


class EnemyIntent(BaseModel):
    enemy_id: str
    intent_type: IntentType = "unknown"
    amount: int | None = None
    hits: int | None = None
    min_amount: int | None = None
    max_amount: int | None = None
    probability: float = 1.0
    is_random: bool = False
    intent_text: str | None = None


class EnemyState(BaseModel):
    enemy_id: str
    hp: int
    max_hp: int
    block: int = 0
    intents: list[EnemyIntent] = Field(default_factory=list)


class PlayerState(BaseModel):
    hp: int
    max_hp: int
    block: int = 0
    energy: int = 0
    hand: list[str] = Field(default_factory=list)
    draw_pile_count: int = 0
    discard_pile_count: int = 0


class EventState(BaseModel):
    event_id: str
    title: str
    options: list[str] = Field(default_factory=list)


class GameStateSnapshot(BaseModel):
    source: Literal["mock", "memory", "screen", "hybrid", "mod", "mcp_api"]
    frame_id: int
    timestamp_ms: int
    in_combat: bool = False
    in_event: bool = False
    turn: int | None = None
    player: PlayerState
    enemies: list[EnemyState] = Field(default_factory=list)
    event: EventState | None = None
    warnings: list[str] = Field(default_factory=list)
    # MCP screen type, e.g. monster/map/shop/event/menu.
    state_type: str | None = None
    # MCP mode metadata (mainly for multiplayer endpoint polling).
    game_mode: str | None = None
    net_type: str | None = None
    player_count: int | None = None
    local_player_slot: int | None = None
    # Full raw state payload from MCP endpoint for downstream debugging/screen-specific logic.
    raw_state: dict[str, Any] = Field(default_factory=dict)
