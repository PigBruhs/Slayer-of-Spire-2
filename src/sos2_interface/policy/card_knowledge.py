from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


DEFAULT_CARD_KNOWLEDGE_FILE = "config/card_knowledge.local.json"

KNOWN_CARD_COSTS: dict[str, int] = {
    "strike": 1,
    "defend": 1,
    "bash": 2,
    "anger": 0,
    "shrug_it_off": 1,
}

KNOWN_CARD_ALIASES: dict[str, str] = {
    "strike": "strike",
    "defend": "defend",
    "bash": "bash",
    "anger": "anger",
    "shrug_it_off": "shrug_it_off",
    "shrug it off": "shrug_it_off",
}

RANDOM_BOUNDARY_CARDS: set[str] = {
    "shrug_it_off",
    "pommel_strike",
    "battle_trance",
    "wild_strike",
    "infernal_blade",
}


@dataclass
class CardEffects:
    damage: int = 0
    block: int = 0
    draw: int = 0
    energy_gain: int = 0
    self_hp_loss: int = 0
    strength_delta: int = 0
    dexterity_delta: int = 0
    vulnerable: int = 0
    weak: int = 0
    frail: int = 0


@dataclass
class _KnowledgeSnapshot:
    costs: dict[str, int]
    random_boundary_cards: set[str]
    aliases: dict[str, str]
    effects: dict[str, CardEffects]


def normalize_card_id(card_id: str | None) -> str:
    if not card_id:
        return ""
    normalized = card_id.strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def normalize_alias_text(text: str | None) -> str:
    if not text:
        return ""
    lowered = text.strip().lower()
    lowered = re.sub(r"\[[^]]*]", " ", lowered)
    lowered = re.sub(r"[^\w]+", " ", lowered)
    lowered = lowered.replace("_", " ")
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _compact_alias_text(text: str) -> str:
    return text.replace(" ", "")


def _build_default_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for alias, card_id in KNOWN_CARD_ALIASES.items():
        normalized_card_id = normalize_card_id(card_id)
        if not normalized_card_id:
            continue
        alias_key = normalize_alias_text(alias)
        if alias_key:
            aliases[alias_key] = normalized_card_id
        _register_card_aliases(aliases, normalized_card_id)
    return aliases


def _register_card_aliases(aliases: dict[str, str], card_id: str) -> None:
    alias_variants = {
        card_id,
        card_id.replace("_", " "),
        card_id.replace("_", ""),
    }
    for variant in alias_variants:
        key = normalize_alias_text(variant)
        if key:
            aliases[key] = card_id


def _to_card_effects(payload: dict[str, object]) -> CardEffects:
    return CardEffects(
        damage=_as_non_negative_int(payload.get("damage")),
        block=_as_non_negative_int(payload.get("block")),
        draw=_as_non_negative_int(payload.get("draw")),
        energy_gain=_as_int(payload.get("energy_gain")),
        self_hp_loss=_as_non_negative_int(payload.get("self_hp_loss")),
        strength_delta=_as_int(payload.get("strength_delta")),
        dexterity_delta=_as_int(payload.get("dexterity_delta")),
        vulnerable=_as_non_negative_int(payload.get("vulnerable")),
        weak=_as_non_negative_int(payload.get("weak")),
        frail=_as_non_negative_int(payload.get("frail")),
    )


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return 0


def _as_non_negative_int(value: object) -> int:
    return max(0, _as_int(value))


def _has_non_zero_effect(effects: CardEffects) -> bool:
    return any(
        [
            effects.damage,
            effects.block,
            effects.draw,
            effects.energy_gain,
            effects.self_hp_loss,
            effects.strength_delta,
            effects.dexterity_delta,
            effects.vulnerable,
            effects.weak,
            effects.frail,
        ]
    )


def _resolve_card_id_from_snapshot(raw_text: str, snapshot: _KnowledgeSnapshot) -> str | None:
    normalized_id = normalize_card_id(raw_text)
    if normalized_id in snapshot.costs or normalized_id in snapshot.random_boundary_cards:
        return normalized_id

    alias_key = normalize_alias_text(raw_text)
    if not alias_key:
        return None

    direct = snapshot.aliases.get(alias_key)
    if direct:
        return direct

    compact_key = _compact_alias_text(alias_key)
    if compact_key:
        for key, mapped in snapshot.aliases.items():
            if _compact_alias_text(key) == compact_key:
                return mapped

    keys = list(snapshot.aliases.keys())
    if not keys:
        return None
    match = difflib.get_close_matches(alias_key, keys, n=1, cutoff=0.86)
    if match:
        return snapshot.aliases.get(match[0])
    return None


def _load_knowledge_file(path: Path) -> _KnowledgeSnapshot:
    data = json.loads(path.read_text(encoding="utf-8-sig"))

    raw_costs = data.get("costs", {})
    costs: dict[str, int] = dict(KNOWN_CARD_COSTS)
    if isinstance(raw_costs, dict):
        for card_id, value in raw_costs.items():
            normalized = normalize_card_id(str(card_id))
            if normalized and isinstance(value, int) and value >= 0:
                costs[normalized] = value

    random_cards: set[str] = set(RANDOM_BOUNDARY_CARDS)
    raw_random_cards = data.get("random_boundary_cards", [])
    if isinstance(raw_random_cards, list):
        for item in raw_random_cards:
            normalized = normalize_card_id(str(item))
            if normalized:
                random_cards.add(normalized)

    aliases: dict[str, str] = _build_default_aliases()
    raw_aliases = data.get("aliases", {})
    if isinstance(raw_aliases, dict):
        for alias_raw, card_raw in raw_aliases.items():
            card_id = normalize_card_id(str(card_raw))
            alias_key = normalize_alias_text(str(alias_raw))
            if card_id and alias_key:
                aliases[alias_key] = card_id

    for card_id in costs.keys():
        _register_card_aliases(aliases, card_id)

    raw_cards = data.get("cards", [])
    if isinstance(raw_cards, list):
        for entry in raw_cards:
            if not isinstance(entry, dict):
                continue
            card_id = normalize_card_id(str(entry.get("id") or entry.get("card_id") or ""))
            if not card_id:
                continue

            cost = entry.get("cost")
            if isinstance(cost, int) and cost >= 0:
                costs[card_id] = cost

            if bool(entry.get("random_boundary")):
                random_cards.add(card_id)

            names = entry.get("aliases")
            if isinstance(names, list):
                for name in names:
                    alias_key = normalize_alias_text(str(name))
                    if alias_key:
                        aliases[alias_key] = card_id

            _register_card_aliases(aliases, card_id)

    effects: dict[str, CardEffects] = {}
    raw_effects = data.get("effects", {})
    if isinstance(raw_effects, dict):
        for card_raw, payload in raw_effects.items():
            card_id = normalize_card_id(str(card_raw))
            if card_id and isinstance(payload, dict):
                effects[card_id] = _to_card_effects(payload)

    if isinstance(raw_cards, list):
        for entry in raw_cards:
            if not isinstance(entry, dict):
                continue
            card_id = normalize_card_id(str(entry.get("id") or entry.get("card_id") or ""))
            if not card_id or card_id in effects:
                continue
            inferred = _to_card_effects(entry)
            if _has_non_zero_effect(inferred):
                effects[card_id] = inferred

    return _KnowledgeSnapshot(costs=costs, random_boundary_cards=random_cards, aliases=aliases, effects=effects)


class _LocalCardKnowledgeStore:
    def __init__(self) -> None:
        env_override = os.getenv("SOS2_CARD_KNOWLEDGE_PATH")
        self._path = Path(env_override) if env_override else Path(DEFAULT_CARD_KNOWLEDGE_FILE)
        self._snapshot = _KnowledgeSnapshot(
            costs=dict(KNOWN_CARD_COSTS),
            random_boundary_cards=set(RANDOM_BOUNDARY_CARDS),
            aliases=_build_default_aliases(),
            effects={},
        )
        self._last_mtime: float | None = None
        self._lock = Lock()

    def estimate_cost(self, card_id: str | None) -> int | None:
        normalized = normalize_card_id(card_id)
        if not normalized:
            return None
        self._reload_if_needed()
        return self._snapshot.costs.get(normalized)

    def estimate_effects(self, card_id: str | None) -> CardEffects | None:
        normalized = normalize_card_id(card_id)
        if not normalized:
            return None
        self._reload_if_needed()
        return self._snapshot.effects.get(normalized)

    def is_random_boundary(self, card_id: str | None) -> bool:
        normalized = normalize_card_id(card_id)
        if not normalized:
            return False
        self._reload_if_needed()
        return normalized in self._snapshot.random_boundary_cards

    def resolve_card_id(self, raw_text: str | None) -> str | None:
        if not raw_text:
            return None
        self._reload_if_needed()
        return _resolve_card_id_from_snapshot(raw_text, self._snapshot)

    def _reload_if_needed(self) -> None:
        with self._lock:
            if not self._path.exists():
                return

            current_mtime = self._path.stat().st_mtime
            if self._last_mtime is not None and current_mtime == self._last_mtime:
                return

            loaded = _load_knowledge_file(self._path)
            self._snapshot = _KnowledgeSnapshot(
                costs=loaded.costs,
                random_boundary_cards=loaded.random_boundary_cards,
                aliases=loaded.aliases,
                effects=loaded.effects,
            )
            self._last_mtime = current_mtime


_STORE = _LocalCardKnowledgeStore()


def estimate_card_cost(card_id: str | None) -> int | None:
    return _STORE.estimate_cost(card_id)


def estimate_card_effects(card_id: str | None) -> CardEffects | None:
    return _STORE.estimate_effects(card_id)


def is_random_boundary_card(card_id: str | None) -> bool:
    return _STORE.is_random_boundary(card_id)


def resolve_card_id(raw_text: str | None) -> str | None:
    return _STORE.resolve_card_id(raw_text)


