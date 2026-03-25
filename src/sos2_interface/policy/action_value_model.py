from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sos2_interface.contracts.action import ActionCommand
from sos2_interface.contracts.state import GameStateSnapshot


@dataclass
class ActionValueModel:
    weights: dict[str, float]
    bias: float = 0.0

    def score(self, state: GameStateSnapshot, action: ActionCommand) -> float:
        features = extract_features(state, action)
        total = self.bias
        for key, value in features.items():
            total += self.weights.get(key, 0.0) * value
        return float(total)


def load_action_value_model(path: str | None) -> ActionValueModel | None:
    if not path:
        return None
    model_path = Path(path)
    if not model_path.exists():
        return None

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None

    raw_weights = payload.get("weights")
    if not isinstance(raw_weights, dict):
        return None

    weights: dict[str, float] = {}
    for key, value in raw_weights.items():
        try:
            weights[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    bias = payload.get("bias", 0.0)
    try:
        bias_value = float(bias)
    except (TypeError, ValueError):
        bias_value = 0.0

    return ActionValueModel(weights=weights, bias=bias_value)


def extract_features(state: GameStateSnapshot, action: ActionCommand) -> dict[str, float]:
    enemies_alive = sum(1 for enemy in state.enemies if enemy.hp > 0)
    enemy_hp_total = sum(max(0, enemy.hp) for enemy in state.enemies)

    features: dict[str, float] = {
        "bias": 1.0,
        "player_hp_norm": _safe_div(state.player.hp, max(1, state.player.max_hp)),
        "player_block_norm": min(200.0, float(state.player.block)) / 100.0,
        "player_energy_norm": min(10.0, float(state.player.energy)) / 5.0,
        "hand_count_norm": min(20.0, float(len(state.player.hand))) / 10.0,
        "enemies_alive_norm": min(10.0, float(enemies_alive)) / 5.0,
        "enemy_hp_norm": min(400.0, float(enemy_hp_total)) / 100.0,
        "in_combat": 1.0 if state.in_combat else 0.0,
        "in_event": 1.0 if state.in_event else 0.0,
        f"action_type={action.action_type}": 1.0,
    }

    if action.card_id:
        features[f"card_id={_normalize_card_id(action.card_id)}"] = 1.0

    cost = action.metadata.get("cost")
    if isinstance(cost, (int, float)):
        features["action_cost_norm"] = min(10.0, float(cost)) / 5.0

    if bool(action.metadata.get("random")):
        features["action_random"] = 1.0

    if action.option_index is not None:
        features["option_index_norm"] = min(20.0, float(action.option_index)) / 10.0

    return features


def extract_features_from_compact(compact_state: dict[str, object], action_payload: dict[str, object]) -> dict[str, float]:
    player = compact_state.get("player") if isinstance(compact_state.get("player"), dict) else {}
    enemies_raw = compact_state.get("enemies") if isinstance(compact_state.get("enemies"), list) else []
    enemies = [enemy for enemy in enemies_raw if isinstance(enemy, dict)]

    player_hp = _to_float(player.get("hp"), 0.0)
    player_max_hp = max(1.0, _to_float(player.get("max_hp"), 1.0))
    player_block = _to_float(player.get("block"), 0.0)
    player_energy = _to_float(player.get("energy"), 0.0)
    hand_count = _to_float(player.get("hand_count"), 0.0)

    enemies_alive = sum(1 for enemy in enemies if _to_float(enemy.get("hp"), 0.0) > 0)
    enemy_hp_total = sum(max(0.0, _to_float(enemy.get("hp"), 0.0)) for enemy in enemies)

    action_type = str(action_payload.get("action_type") or "noop")
    card_id = str(action_payload.get("card_id") or "")
    option_index = action_payload.get("option_index")
    metadata = action_payload.get("metadata") if isinstance(action_payload.get("metadata"), dict) else {}

    features: dict[str, float] = {
        "bias": 1.0,
        "player_hp_norm": _safe_div(player_hp, player_max_hp),
        "player_block_norm": min(200.0, player_block) / 100.0,
        "player_energy_norm": min(10.0, player_energy) / 5.0,
        "hand_count_norm": min(20.0, hand_count) / 10.0,
        "enemies_alive_norm": min(10.0, float(enemies_alive)) / 5.0,
        "enemy_hp_norm": min(400.0, enemy_hp_total) / 100.0,
        "in_combat": 1.0 if bool(compact_state.get("in_combat")) else 0.0,
        "in_event": 1.0 if bool(compact_state.get("in_event")) else 0.0,
        f"action_type={action_type}": 1.0,
    }

    if card_id:
        features[f"card_id={_normalize_card_id(card_id)}"] = 1.0

    cost = metadata.get("cost") if isinstance(metadata, dict) else None
    if isinstance(cost, (int, float)):
        features["action_cost_norm"] = min(10.0, float(cost)) / 5.0

    if isinstance(metadata, dict) and bool(metadata.get("random")):
        features["action_random"] = 1.0

    if isinstance(option_index, int):
        features["option_index_norm"] = min(20.0, float(option_index)) / 10.0

    return features


def _normalize_card_id(card_id: str) -> str:
    lowered = card_id.strip().lower().replace("-", "_").replace(" ", "_")
    cleaned = "".join(ch for ch in lowered if ch.isalnum() or ch == "_")
    return cleaned or "unknown"


def _safe_div(value: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(value) / float(denom)


def _to_float(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


