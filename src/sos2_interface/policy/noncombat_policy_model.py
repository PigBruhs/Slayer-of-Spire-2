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

STATE_TYPES = [
    "event",
    "combat_rewards",
    "card_reward",
    "map",
    "rest_site",
    "shop",
    "card_select",
    "relic_select",
    "treasure",
    "overlay",
    "menu",
    "unknown",
]

ACTION_TYPES = [
    "claim_reward",
    "select_card_reward",
    "skip_card_reward",
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
    "proceed",
    "noop",
]

DEFAULT_TOKEN_BUCKETS = 8192
DEFAULT_TOKEN_SEQ_LEN = 40
DEFAULT_NUMERIC_DIM = 24


@dataclass
class NonCombatPolicyModel:
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
    transformer: "NonCombatTransformerRegressor | None" = None
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

        x = extract_feature_vector(state, action, max(64, int(self.input_dim or 224)))
        assert self.w1 is not None and self.b1 is not None and self.w2 is not None and self.b2 is not None and self.w3 is not None and self.b3 is not None
        h1 = np.maximum(0.0, x @ self.w1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)
        out = h2 @ self.w3 + self.b3
        return float(out[0])


class NonCombatTransformerRegressor(nn.Module):
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


@dataclass
class CompactActionView:
    state_type: str | None
    in_combat: bool
    in_event: bool
    turn: int | None
    player_hp: float
    player_max_hp: float
    player_block: float
    player_energy: float
    hand_count: float
    draw_pile_count: float
    discard_pile_count: float
    enemies_alive: float
    enemy_hp_total: float


@dataclass
class CompactAction:
    action_type: str
    card_id: str | None
    target_id: str | None
    option_index: int | None
    metadata: dict[str, object]


def load_noncombat_policy_model(path: str | None) -> NonCombatPolicyModel | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None

    model_type = str(payload.get("model_type") or "")
    if model_type == "noncombat_transformer_value":
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

        model = NonCombatTransformerRegressor(
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
        return NonCombatPolicyModel(
            transformer=model,
            token_buckets=token_buckets,
            token_seq_len=token_seq_len,
            numeric_dim=numeric_dim,
            model_kind="transformer",
        )

    try:
        input_dim = int(payload.get("input_dim", 224))
        hidden1 = int(payload.get("hidden1", 384))
        hidden2 = int(payload.get("hidden2", 192))
        w1 = np.array(payload["w1"], dtype=np.float32)
        b1 = np.array(payload["b1"], dtype=np.float32)
        w2 = np.array(payload["w2"], dtype=np.float32)
        b2 = np.array(payload["b2"], dtype=np.float32)
        w3 = np.array(payload["w3"], dtype=np.float32)
        b3 = np.array(payload["b3"], dtype=np.float32)
    except Exception:
        return None

    return NonCombatPolicyModel(
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


def extract_feature_vector(state: GameStateSnapshot, action: ActionCommand, input_dim: int = 224) -> np.ndarray:
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
    return extract_feature_vector_from_compact(compact_state, action_payload, input_dim=input_dim)


def extract_feature_vector_from_compact(
    compact_state: dict[str, object],
    action_payload: dict[str, object],
    input_dim: int = 224,
) -> np.ndarray:
    vec = np.zeros((input_dim,), dtype=np.float32)

    player = compact_state.get("player") if isinstance(compact_state.get("player"), dict) else {}
    enemies_raw = compact_state.get("enemies") if isinstance(compact_state.get("enemies"), list) else []
    enemies = [enemy for enemy in enemies_raw if isinstance(enemy, dict)]

    player_hp = _to_float(player.get("hp"), 0.0)
    player_max_hp = max(1.0, _to_float(player.get("max_hp"), 1.0))
    player_block = _to_float(player.get("block"), 0.0)
    player_energy = _to_float(player.get("energy"), 0.0)
    hand_count = _to_float(player.get("hand_count"), 0.0)
    draw_pile_count = _to_float(player.get("draw_pile_count"), 0.0)
    discard_pile_count = _to_float(player.get("discard_pile_count"), 0.0)

    enemies_alive = sum(1 for enemy in enemies if _to_float(enemy.get("hp"), 0.0) > 0)
    enemy_hp_total = sum(max(0.0, _to_float(enemy.get("hp"), 0.0)) for enemy in enemies)

    state_type = str(compact_state.get("state_type") or "unknown").strip().lower()
    if state_type not in STATE_TYPES:
        state_type = "unknown"

    # Core numeric slice.
    core = [
        player_hp,
        player_max_hp,
        player_block,
        player_energy,
        hand_count,
        draw_pile_count,
        discard_pile_count,
        float(enemies_alive),
        enemy_hp_total,
        1.0 if bool(compact_state.get("in_event")) else 0.0,
        1.0 if bool(compact_state.get("in_combat")) else 0.0,
        _to_float(compact_state.get("turn"), 0.0),
    ]
    for idx, value in enumerate(core):
        if idx >= 32:
            break
        scale = 100.0 if idx in {0, 1, 8} else 10.0
        vec[idx] = float(value) / scale

    # State type one-hot.
    state_offset = 32
    vec[state_offset + STATE_TYPES.index(state_type)] = 1.0

    # Action type one-hot.
    action_type = str(action_payload.get("action_type") or "noop")
    action_offset = 64
    if action_type in ACTION_TYPES:
        vec[action_offset + ACTION_TYPES.index(action_type)] = 1.0

    # Index and cost hints.
    option_index = action_payload.get("option_index")
    if isinstance(option_index, int):
        vec[120] = float(option_index) / 20.0

    metadata = action_payload.get("metadata") if isinstance(action_payload.get("metadata"), dict) else {}
    cost = metadata.get("cost") if isinstance(metadata, dict) else None
    if isinstance(cost, (int, float)):
        vec[121] = float(cost) / 8.0

    # Hashed sparse identifiers.
    _set_hashed(vec, 128, 48, action_payload.get("card_id"))
    _set_hashed(vec, 128, 48, action_payload.get("target_id"))
    _set_hashed(vec, 176, 48, metadata.get("potion_id") if isinstance(metadata, dict) else None)
    _set_hashed(vec, 176, 48, metadata.get("event_id") if isinstance(metadata, dict) else None)
    _set_hashed(vec, 176, 48, metadata.get("shop_item_id") if isinstance(metadata, dict) else None)
    _set_hashed(vec, 176, 48, metadata.get("shop_item_kind") if isinstance(metadata, dict) else None)
    _set_hashed(vec, 176, 48, metadata.get("shop_category") if isinstance(metadata, dict) else None)

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
    tokens = [
        f"state:{compact_state.get('state_type') or 'unknown'}",
        f"action:{action_payload.get('action_type') or 'noop'}",
        f"card:{action_payload.get('card_id') or ''}",
        f"target:{action_payload.get('target_id') or ''}",
        f"shop_item:{metadata.get('shop_item_id') if isinstance(metadata, dict) else ''}",
        f"shop_kind:{metadata.get('shop_item_kind') if isinstance(metadata, dict) else ''}",
        f"shop_cat:{metadata.get('shop_category') if isinstance(metadata, dict) else ''}",
        f"event:{metadata.get('event_id') if isinstance(metadata, dict) else ''}",
        f"potion:{metadata.get('potion_id') if isinstance(metadata, dict) else ''}",
    ]
    token_ids = np.zeros((int(token_seq_len),), dtype=np.int64)
    for idx, token in enumerate(tokens[:token_seq_len]):
        token_ids[idx] = _hash_to_bucket(str(token), int(token_buckets))

    player = compact_state.get("player") if isinstance(compact_state.get("player"), dict) else {}
    enemies = compact_state.get("enemies") if isinstance(compact_state.get("enemies"), list) else []
    enemy_hp_total = 0.0
    enemy_alive = 0.0
    for item in enemies:
        if not isinstance(item, dict):
            continue
        hp = _to_float(item.get("hp"), 0.0)
        if hp > 0:
            enemy_alive += 1.0
            enemy_hp_total += hp

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
        1.0 if bool(compact_state.get("in_event")) else 0.0,
        1.0 if bool(compact_state.get("in_combat")) else 0.0,
        _to_float(compact_state.get("turn"), 0.0),
        _to_float(action_payload.get("option_index"), 0.0),
        _to_float(metadata.get("cost") if isinstance(metadata, dict) else None, 0.0),
        1.0 if bool(metadata.get("is_on_sale")) else 0.0,
    ]
    for idx, value in enumerate(base[:numeric_dim]):
        scale = 100.0 if idx in {0, 1, 8} else 10.0
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

