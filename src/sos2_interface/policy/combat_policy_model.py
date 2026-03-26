from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sos2_interface.contracts.action import ActionCommand
from sos2_interface.contracts.state import GameStateSnapshot


ACTION_TYPES = [
    "play_card",
    "use_potion",
    "end_turn",
    "combat_select_card",
    "combat_confirm_selection",
    "noop",
]


@dataclass
class CombatPolicyModel:
    input_dim: int
    hidden1: int
    hidden2: int
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray

    def score(self, state: GameStateSnapshot, action: ActionCommand) -> float:
        x = extract_feature_vector(state, action, self.input_dim)
        h1 = np.maximum(0.0, x @ self.w1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)
        out = h2 @ self.w3 + self.b3
        return float(out[0])


def load_combat_policy_model(path: str | None) -> CombatPolicyModel | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None

    try:
        input_dim = int(payload.get("input_dim", 192))
        hidden1 = int(payload.get("hidden1", 256))
        hidden2 = int(payload.get("hidden2", 128))
        w1 = np.array(payload["w1"], dtype=np.float32)
        b1 = np.array(payload["b1"], dtype=np.float32)
        w2 = np.array(payload["w2"], dtype=np.float32)
        b2 = np.array(payload["b2"], dtype=np.float32)
        w3 = np.array(payload["w3"], dtype=np.float32)
        b3 = np.array(payload["b3"], dtype=np.float32)
    except Exception:
        return None

    return CombatPolicyModel(
        input_dim=input_dim,
        hidden1=hidden1,
        hidden2=hidden2,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        w3=w3,
        b3=b3,
    )


def extract_feature_vector(state: GameStateSnapshot, action: ActionCommand, input_dim: int = 192) -> np.ndarray:
    vec = np.zeros((input_dim,), dtype=np.float32)

    enemy_hp_total = 0.0
    enemy_block_total = 0.0
    incoming_damage = 0.0
    enemy_count = 0.0
    for enemy in state.enemies:
        if enemy.hp <= 0:
            continue
        enemy_count += 1.0
        enemy_hp_total += float(enemy.hp)
        enemy_block_total += float(enemy.block)
        for intent in enemy.intents:
            if intent.intent_type in {"attack", "multi_attack", "unknown"}:
                amount = intent.amount or intent.max_amount or intent.min_amount or 0
                hits = intent.hits or 1
                incoming_damage += float(max(0, amount) * max(1, hits)) * float(intent.probability)

    # Core numeric slice (0-31)
    core = [
        float(state.player.hp),
        float(max(1, state.player.max_hp)),
        float(state.player.block),
        float(state.player.energy),
        float(len(state.player.hand)),
        float(state.player.draw_pile_count),
        float(state.player.discard_pile_count),
        enemy_count,
        enemy_hp_total,
        enemy_block_total,
        incoming_damage,
        1.0 if state.in_combat else 0.0,
        1.0 if state.in_event else 0.0,
        float(state.turn or 0),
    ]
    for idx, value in enumerate(core[:32]):
        vec[idx] = value / (50.0 if idx in {0, 1, 8} else 10.0)

    # Action type one-hot slice (32-63)
    offset = 32
    if action.action_type in ACTION_TYPES:
        vec[offset + ACTION_TYPES.index(action.action_type)] = 1.0

    # Cost and option features
    cost = action.metadata.get("cost")
    if isinstance(cost, (int, float)):
        vec[48] = float(cost) / 5.0
    if action.option_index is not None:
        vec[49] = float(action.option_index) / 10.0

    # Hashed sparse features for ids/text (64+)
    _set_hashed(vec, 64, 64, action.card_id)
    _set_hashed(vec, 64, 64, action.target_id)
    _set_hashed(vec, 64, 64, action.metadata.get("potion_id"))

    raw = state.raw_state if isinstance(state.raw_state, dict) else {}
    battle = raw.get("battle") if isinstance(raw.get("battle"), dict) else {}
    player = battle.get("player") if isinstance(battle.get("player"), dict) else {}

    statuses = player.get("status") if isinstance(player.get("status"), list) else []
    vec[50] = float(len(statuses)) / 20.0

    potions = player.get("potions") if isinstance(player.get("potions"), list) else []
    vec[51] = float(len(potions)) / 10.0

    for potion in potions:
        if isinstance(potion, dict):
            _set_hashed(vec, 128, 64, potion.get("id") or potion.get("name"))

    return vec


def extract_feature_vector_from_compact(
    compact_state: dict[str, object],
    action_payload: dict[str, object],
    input_dim: int = 192,
) -> np.ndarray:
    vec = np.zeros((input_dim,), dtype=np.float32)

    player = compact_state.get("player") if isinstance(compact_state.get("player"), dict) else {}
    enemies_raw = compact_state.get("enemies") if isinstance(compact_state.get("enemies"), list) else []
    enemies = [enemy for enemy in enemies_raw if isinstance(enemy, dict)]

    enemy_hp_total = 0.0
    enemy_block_total = 0.0
    incoming_damage = 0.0
    enemy_count = 0.0
    for enemy in enemies:
        hp = _to_float(enemy.get("hp"), 0.0)
        if hp <= 0:
            continue
        enemy_count += 1.0
        enemy_hp_total += hp
        enemy_block_total += _to_float(enemy.get("block"), 0.0)

    core = [
        _to_float(player.get("hp"), 0.0),
        max(1.0, _to_float(player.get("max_hp"), 1.0)),
        _to_float(player.get("block"), 0.0),
        _to_float(player.get("energy"), 0.0),
        _to_float(player.get("hand_count"), 0.0),
        _to_float(player.get("draw_pile_count"), 0.0),
        _to_float(player.get("discard_pile_count"), 0.0),
        enemy_count,
        enemy_hp_total,
        enemy_block_total,
        incoming_damage,
        1.0 if bool(compact_state.get("in_combat")) else 0.0,
        1.0 if bool(compact_state.get("in_event")) else 0.0,
        _to_float(compact_state.get("turn"), 0.0),
    ]
    for idx, value in enumerate(core[:32]):
        vec[idx] = value / (50.0 if idx in {0, 1, 8} else 10.0)

    action_type = str(action_payload.get("action_type") or "noop")
    offset = 32
    if action_type in ACTION_TYPES:
        vec[offset + ACTION_TYPES.index(action_type)] = 1.0

    metadata = action_payload.get("metadata") if isinstance(action_payload.get("metadata"), dict) else {}
    cost = metadata.get("cost") if isinstance(metadata, dict) else None
    if isinstance(cost, (int, float)):
        vec[48] = float(cost) / 5.0

    option_index = action_payload.get("option_index")
    if isinstance(option_index, int):
        vec[49] = float(option_index) / 10.0

    _set_hashed(vec, 64, 64, action_payload.get("card_id"))
    _set_hashed(vec, 64, 64, action_payload.get("target_id"))
    _set_hashed(vec, 64, 64, metadata.get("potion_id") if isinstance(metadata, dict) else None)
    return vec


def _set_hashed(vec: np.ndarray, start: int, width: int, value: object) -> None:
    if start >= vec.shape[0] or width <= 0:
        return
    text = str(value or "").strip().lower()
    if not text:
        return
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % width
    target = start + idx
    if 0 <= target < vec.shape[0]:
        vec[target] += 1.0


def _to_float(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


