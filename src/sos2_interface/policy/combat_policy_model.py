from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

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

DEFAULT_TOKEN_BUCKETS = 8192
DEFAULT_TOKEN_SEQ_LEN = 40
DEFAULT_NUMERIC_DIM = 24


@dataclass
class CombatPolicyModel:
    # legacy MLP fields
    input_dim: int = 0
    hidden1: int = 0
    hidden2: int = 0
    w1: np.ndarray | None = None
    b1: np.ndarray | None = None
    w2: np.ndarray | None = None
    b2: np.ndarray | None = None
    w3: np.ndarray | None = None
    b3: np.ndarray | None = None
    # transformer fields
    transformer: "CombatTransformerRegressor | None" = None
    token_buckets: int = DEFAULT_TOKEN_BUCKETS
    token_seq_len: int = DEFAULT_TOKEN_SEQ_LEN
    numeric_dim: int = DEFAULT_NUMERIC_DIM
    model_kind: str = "mlp"

    def score(self, state: GameStateSnapshot, action: ActionCommand) -> float:
        if self.model_kind == "transformer" and self.transformer is not None:
            token_ids, numeric = extract_deep_features(
                state,
                action,
                token_seq_len=self.token_seq_len,
                token_buckets=self.token_buckets,
                numeric_dim=self.numeric_dim,
            )
            with torch.no_grad():
                token_t = torch.from_numpy(token_ids.astype(np.int64)).unsqueeze(0)
                numeric_t = torch.from_numpy(numeric.astype(np.float32)).unsqueeze(0)
                out = self.transformer(token_t, numeric_t)
            return float(out[0, 0].detach().cpu())

        x = extract_feature_vector(state, action, max(64, int(self.input_dim or 192)))
        assert self.w1 is not None and self.b1 is not None and self.w2 is not None and self.b2 is not None and self.w3 is not None and self.b3 is not None
        h1 = np.maximum(0.0, x @ self.w1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)
        out = h2 @ self.w3 + self.b3
        return float(out[0])


class CombatTransformerRegressor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_seq_len: int,
        numeric_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_seq_len = int(token_seq_len)
        self.numeric_dim = int(numeric_dim)
        self.token_embed = nn.Embedding(int(vocab_size), int(d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, int(d_model)))
        self.pos = nn.Parameter(torch.zeros(1, self.token_seq_len + 1, int(d_model)))
        self.numeric_proj = nn.Sequential(
            nn.Linear(self.numeric_dim, int(d_model)),
            nn.GELU(),
            nn.LayerNorm(int(d_model)),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.head = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), 1),
        )

    def forward(self, token_ids: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        tok = self.token_embed(token_ids)
        num_tok = self.numeric_proj(numeric).unsqueeze(1)
        cls = self.cls.expand(token_ids.shape[0], -1, -1)
        x = torch.cat([cls + num_tok, tok], dim=1)
        x = x + self.pos[:, : x.shape[1], :]
        h = self.encoder(x)
        return self.head(h[:, 0, :])


def load_combat_policy_model(path: str | None) -> CombatPolicyModel | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None

    model_type = str(payload.get("model_type") or "")
    if model_type == "combat_transformer_value":
        try:
            token_buckets = int(payload.get("token_buckets", DEFAULT_TOKEN_BUCKETS))
            token_seq_len = int(payload.get("token_seq_len", DEFAULT_TOKEN_SEQ_LEN))
            numeric_dim = int(payload.get("numeric_dim", DEFAULT_NUMERIC_DIM))
            d_model = int(payload.get("d_model", 192))
            nhead = int(payload.get("nhead", 6))
            num_layers = int(payload.get("num_layers", 3))
            ff_dim = int(payload.get("ff_dim", 384))
            dropout = float(payload.get("dropout", 0.1))
            raw_state_dict = payload.get("state_dict")
            if not isinstance(raw_state_dict, dict):
                return None
            state_dict = {key: torch.tensor(value, dtype=torch.float32) for key, value in raw_state_dict.items()}
        except Exception:
            return None

        model = CombatTransformerRegressor(
            vocab_size=token_buckets,
            token_seq_len=token_seq_len,
            numeric_dim=numeric_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            return None
        model.eval()
        return CombatPolicyModel(
            transformer=model,
            token_buckets=token_buckets,
            token_seq_len=token_seq_len,
            numeric_dim=numeric_dim,
            model_kind="transformer",
        )

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
        model_kind="mlp",
    )


def extract_feature_vector(state: GameStateSnapshot, action: ActionCommand, input_dim: int = 192) -> np.ndarray:
    vec = np.zeros((input_dim,), dtype=np.float32)

    enemy_hp_total = 0.0
    enemy_block_total = 0.0
    incoming_damage = 0.0
    enemy_count = 0.0
    has_invincible = False
    for enemy in state.enemies:
        hp = float(enemy.hp)
        if hp > 9999.0:
            hp = 999.0
            has_invincible = True
        if hp <= 0:
            continue
        enemy_count += 1.0
        enemy_hp_total += hp
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
        
    if has_invincible:
        vec[52] = 1.0

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
        if hp > 9999.0:
            hp = 999.0
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


def extract_deep_features(
    state: GameStateSnapshot,
    action: ActionCommand,
    token_seq_len: int = DEFAULT_TOKEN_SEQ_LEN,
    token_buckets: int = DEFAULT_TOKEN_BUCKETS,
    numeric_dim: int = DEFAULT_NUMERIC_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    compact_state = {
        "state_type": state.state_type,
        "in_combat": state.in_combat,
        "in_event": state.in_event,
        "turn": state.turn,
        "player": {
            "hp": state.player.hp,
            "max_hp": state.player.max_hp,
            "block": state.player.block,
            "energy": state.player.energy,
            "hand_count": len(state.player.hand),
            "draw_pile_count": state.player.draw_pile_count,
            "discard_pile_count": state.player.discard_pile_count,
        },
        "enemies": [
            {
                "hp": enemy.hp,
                "max_hp": enemy.max_hp,
                "block": enemy.block,
            }
            for enemy in state.enemies
        ],
    }
    action_payload = {
        "action_type": action.action_type,
        "card_id": action.card_id,
        "target_id": action.target_id,
        "option_index": action.option_index,
        "metadata": action.metadata,
    }
    return extract_deep_features_from_compact(
        compact_state,
        action_payload,
        token_seq_len=token_seq_len,
        token_buckets=token_buckets,
        numeric_dim=numeric_dim,
    )


def extract_deep_features_from_compact(
    compact_state: dict[str, object],
    action_payload: dict[str, object],
    token_seq_len: int = DEFAULT_TOKEN_SEQ_LEN,
    token_buckets: int = DEFAULT_TOKEN_BUCKETS,
    numeric_dim: int = DEFAULT_NUMERIC_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    metadata = action_payload.get("metadata") if isinstance(action_payload.get("metadata"), dict) else {}
    
    player = compact_state.get("player") if isinstance(compact_state.get("player"), dict) else {}
    enemies = compact_state.get("enemies") if isinstance(compact_state.get("enemies"), list) else []
    
    enemy_hp_total = 0.0
    enemy_block_total = 0.0
    enemy_alive = 0.0
    has_invincible = False
    for item in enemies:
        if not isinstance(item, dict):
            continue
        hp = _to_float(item.get("hp"), 0.0)
        if hp > 9999.0:
            has_invincible = True
            hp = 999.0
            
        if hp > 0:
            enemy_alive += 1.0
            enemy_hp_total += hp
            enemy_block_total += _to_float(item.get("block"), 0.0)

    tokens = [
        f"state:{compact_state.get('state_type') or 'unknown'}",
        f"action:{action_payload.get('action_type') or 'noop'}",
        f"card:{action_payload.get('card_id') or ''}",
        f"target:{action_payload.get('target_id') or ''}",
        f"potion:{metadata.get('potion_id') if isinstance(metadata, dict) else ''}",
        f"invincible:{'1' if has_invincible else '0'}",
    ]
    token_ids = np.zeros((int(token_seq_len),), dtype=np.int64)
    for idx, token in enumerate(tokens[:token_seq_len]):
        token_ids[idx] = _hash_to_bucket(str(token), int(token_buckets))

    numeric = np.zeros((int(numeric_dim),), dtype=np.float32)
    base = [
        _to_float(player.get("hp"), 0.0),
        max(1.0, _to_float(player.get("max_hp"), 1.0)),
        _to_float(player.get("block"), 0.0),
        _to_float(player.get("energy"), 0.0),
        _to_float(player.get("hand_count"), 0.0),
        _to_float(player.get("draw_pile_count"), 0.0),
        _to_float(player.get("discard_pile_count"), 0.0),
        enemy_alive,
        enemy_hp_total,
        enemy_block_total,
        1.0 if bool(compact_state.get("in_combat")) else 0.0,
        1.0 if bool(compact_state.get("in_event")) else 0.0,
        _to_float(compact_state.get("turn"), 0.0),
        _to_float(action_payload.get("option_index"), 0.0),
        _to_float(metadata.get("cost") if isinstance(metadata, dict) else None, 0.0),
    ]
    for idx, value in enumerate(base[:numeric_dim]):
        scale = 50.0 if idx in {0, 1, 8} else 10.0
        numeric[idx] = float(value) / scale
    return token_ids, numeric


def _hash_to_bucket(text: str, buckets: int) -> int:
    if buckets <= 2:
        return 1
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (buckets - 1) + 1


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

