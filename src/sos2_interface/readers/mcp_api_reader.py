from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error, parse, request

from sos2_interface.contracts.state import EnemyIntent, EnemyState, EventState, GameStateSnapshot, PlayerState
from sos2_interface.readers.base import GameReader


@dataclass
class McpApiReaderConfig:
    host: str = "127.0.0.1"
    port: int = 15526
    mode: str = "singleplayer"
    timeout_seconds: float = 2.0

    @classmethod
    def from_json(cls, path: str | Path) -> "McpApiReaderConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            host=str(data.get("host", "127.0.0.1")),
            port=int(data.get("port", 15526)),
            mode=_normalize_mode(data.get("mode", "singleplayer")),
            timeout_seconds=float(data.get("timeout_seconds", 2.0)),
        )


class McpApiReader(GameReader):
    """Polls STS2 MCP mod REST API and normalizes key fields into GameStateSnapshot."""

    def __init__(self, config: McpApiReaderConfig) -> None:
        self._config = config
        self._frame = 0
        self._last_state_type: str | None = None
        self._last_error: str | None = None
        self._last_http_code: int | None = None

    def read_state(self) -> GameStateSnapshot:
        self._frame += 1
        warnings: list[str] = []

        payload = self._fetch_payload(warnings)
        timestamp_ms = int(time.time() * 1000)
        if payload is None:
            player = PlayerState(hp=0, max_hp=0, energy=0)
            return GameStateSnapshot(
                source="mcp_api",
                frame_id=self._frame,
                timestamp_ms=timestamp_ms,
                in_combat=False,
                in_event=False,
                player=player,
                warnings=warnings,
            )

        state_type = str(payload.get("state_type") or "unknown")
        self._last_state_type = state_type

        section = _screen_section(payload, state_type)
        player = _parse_player(section.get("player") if isinstance(section, dict) else None)
        enemies = _parse_enemies(section.get("enemies") if isinstance(section, dict) else None)

        if player.hp <= 0:
            # Some screens expose player summary in top-level or run blocks; use as fallback.
            player = _parse_player(payload.get("player") or _extract_run_player(payload))

        turn_value = (
            _to_int_or_none(payload.get("turn"))
            or _to_int_or_none(section.get("round") if isinstance(section, dict) else None)
        )
        event = _parse_event(payload, state_type)

        in_combat = state_type in {"monster", "elite", "boss", "hand_select"}
        in_event = state_type == "event"

        return GameStateSnapshot(
            source="mcp_api",
            frame_id=self._frame,
            timestamp_ms=timestamp_ms,
            in_combat=in_combat,
            in_event=in_event,
            turn=turn_value,
            player=player,
            enemies=enemies,
            event=event,
            warnings=warnings,
            state_type=state_type,
            raw_state=payload,
        )

    def status(self) -> dict[str, str | bool | int | None]:
        endpoint = self._endpoint_url()
        return {
            "ok": self._last_error is None,
            "mode": "mcp_api",
            "mcp_mode": self._config.mode,
            "endpoint": endpoint,
            "last_state_type": self._last_state_type,
            "last_http_code": self._last_http_code,
            "last_error": self._last_error,
        }

    def _endpoint_url(self) -> str:
        path = "/api/v1/singleplayer" if self._config.mode == "singleplayer" else "/api/v1/multiplayer"
        base = f"http://{self._config.host}:{self._config.port}{path}"
        query = parse.urlencode({"format": "json"})
        return f"{base}?{query}"

    def _fetch_payload(self, warnings: list[str]) -> dict[str, object] | None:
        endpoint = self._endpoint_url()
        req = request.Request(endpoint, method="GET")
        try:
            with request.urlopen(req, timeout=self._config.timeout_seconds) as resp:
                self._last_http_code = int(getattr(resp, "status", 200))
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            self._last_http_code = int(exc.code)
            self._last_error = f"HTTP {exc.code}: {exc.reason}"
            warnings.append(f"mcp request failed: {self._last_error}")
            if exc.code == 409:
                warnings.append("endpoint mode mismatch: switch singleplayer/multiplayer mode")
            return None
        except error.URLError as exc:
            self._last_http_code = None
            self._last_error = f"connection error: {exc.reason}"
            warnings.append(f"mcp request failed: {self._last_error}")
            return None
        except TimeoutError:
            self._last_http_code = None
            self._last_error = "request timed out"
            warnings.append("mcp request failed: request timed out")
            return None

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            self._last_error = f"invalid json: {exc.msg}"
            warnings.append(f"mcp payload parse failed: {self._last_error}")
            return None

        self._last_error = None
        if not isinstance(payload, dict):
            warnings.append("mcp payload is not an object")
            return None

        status_text = str(payload.get("status", "ok"))
        if status_text.lower() == "error":
            err_text = str(payload.get("error") or "unknown error")
            self._last_error = err_text
            warnings.append(f"mcp api error: {err_text}")
            return None

        return payload


def _parse_player(raw: object) -> PlayerState:
    if not isinstance(raw, dict):
        return PlayerState(hp=0, max_hp=0, energy=0)

    hp = _to_int_or_none(raw.get("hp"))
    max_hp = _to_int_or_none(raw.get("max_hp"))
    block = _to_int_or_none(raw.get("block"))
    energy = _to_int_or_none(raw.get("energy"))

    hand = _extract_hand_ids(raw.get("hand"))

    draw_count = _to_int_or_none(raw.get("draw_pile_count"))
    discard_count = _to_int_or_none(raw.get("discard_pile_count"))

    if draw_count is None and isinstance(raw.get("draw_pile"), list):
        draw_count = len(raw["draw_pile"])
    if discard_count is None and isinstance(raw.get("discard_pile"), list):
        discard_count = len(raw["discard_pile"])

    return PlayerState(
        hp=hp or 0,
        max_hp=max_hp or hp or 0,
        block=block or 0,
        energy=energy or 0,
        hand=hand,
        draw_pile_count=draw_count or 0,
        discard_pile_count=discard_count or 0,
    )


def _extract_hand_ids(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []

    hand: list[str] = []
    for card in raw:
        if isinstance(card, str):
            card_id = card.strip()
            if card_id:
                hand.append(card_id)
            continue

        if not isinstance(card, dict):
            continue

        card_id = card.get("id") or card.get("card_id") or card.get("name")
        if card_id is None:
            continue
        card_text = str(card_id).strip()
        if card_text:
            hand.append(card_text)
    return hand


def _parse_enemies(raw: object) -> list[EnemyState]:
    if not isinstance(raw, list):
        return []

    enemies: list[EnemyState] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        enemy_id = str(item.get("entity_id") or item.get("enemy_id") or item.get("id") or "unknown_enemy")
        hp = _to_int_or_none(item.get("hp"))
        max_hp = _to_int_or_none(item.get("max_hp"))
        block = _to_int_or_none(item.get("block"))
        intents = _parse_intents(enemy_id=enemy_id, raw=item.get("intents"))

        enemies.append(
            EnemyState(
                enemy_id=enemy_id,
                hp=hp or 0,
                max_hp=max_hp or hp or 0,
                block=block or 0,
                intents=intents,
            )
        )
    return enemies


def _parse_intents(enemy_id: str, raw: object) -> list[EnemyIntent]:
    if not isinstance(raw, list):
        return []

    intents: list[EnemyIntent] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        min_amount = _to_int_or_none(item.get("min_amount"))
        max_amount = _to_int_or_none(item.get("max_amount"))
        parsed_amount, parsed_hits = _parse_intent_numbers(item)

        intents.append(
            EnemyIntent(
                enemy_id=enemy_id,
                intent_type=_normalize_intent(item.get("intent_type") or item.get("type") or item.get("title")),
                amount=_to_int_or_none(item.get("amount")) or parsed_amount,
                hits=_to_int_or_none(item.get("hits")) or parsed_hits,
                min_amount=min_amount,
                max_amount=max_amount,
                probability=_to_float_or_default(item.get("probability"), 1.0),
                is_random=bool(item.get("is_random", False)),
                intent_text=str(item.get("description") or item.get("label") or item.get("title") or "").strip() or None,
            )
        )
    return intents


def _parse_intent_numbers(raw_intent: dict[str, object]) -> tuple[int | None, int | None]:
    label = str(raw_intent.get("label") or "").strip()
    if not label:
        return None, None

    # Typical MCP labels include patterns like "12", "3x2", "3×2".
    multi = re.search(r"(\d+)\s*[xX×]\s*(\d+)", label)
    if multi:
        return int(multi.group(1)), int(multi.group(2))

    single = re.search(r"(\d+)", label)
    if single:
        return int(single.group(1)), 1
    return None, None


def _parse_event(payload: dict[str, object], state_type: str) -> EventState | None:
    if state_type != "event":
        return None

    raw_event = payload.get("event")
    if not isinstance(raw_event, dict):
        return EventState(event_id="event", title="Event", options=[])

    event_id = str(raw_event.get("id") or raw_event.get("event_id") or "event")
    title = str(raw_event.get("name") or raw_event.get("title") or event_id)
    options = _extract_event_options(raw_event.get("options"))
    return EventState(event_id=event_id, title=title, options=options)


def _extract_event_options(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []

    options: list[str] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
            if text:
                options.append(text)
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("title") or item.get("label") or item.get("description")
        if text is None:
            continue
        normalized = str(text).strip()
        if normalized:
            options.append(normalized)
    return options


def _normalize_mode(raw: object) -> str:
    text = str(raw or "singleplayer").strip().lower()
    if text not in {"singleplayer", "multiplayer"}:
        return "singleplayer"
    return text


def _normalize_intent(raw: object) -> str:
    text = str(raw or "unknown").strip().lower()
    if "multi" in text and "attack" in text:
        return "multi_attack"
    if "attack" in text or "damage" in text:
        return "attack"
    if "defend" in text or "block" in text:
        return "defend"
    if "debuff" in text:
        return "debuff"
    if "buff" in text:
        return "buff"
    return "unknown"


def _to_int_or_none(raw: object) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)

    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _to_float_or_default(raw: object, default: float) -> float:
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _screen_section(payload: dict[str, object], state_type: str) -> dict[str, object]:
    key_map = {
        "monster": "battle",
        "elite": "battle",
        "boss": "battle",
        "hand_select": "battle",
        "event": "event",
        "map": "map",
        "shop": "shop",
        "rest_site": "rest_site",
        "card_reward": "card_reward",
        "combat_rewards": "combat_rewards",
        "card_select": "card_select",
        "relic_select": "relic_select",
        "treasure": "treasure",
        "overlay": "overlay",
        "menu": "menu",
    }
    key = key_map.get(state_type)
    if not key:
        return payload
    section = payload.get(key)
    if isinstance(section, dict):
        return section
    return payload


def _extract_run_player(payload: dict[str, object]) -> object:
    run = payload.get("run")
    if not isinstance(run, dict):
        return None
    return run.get("player")
