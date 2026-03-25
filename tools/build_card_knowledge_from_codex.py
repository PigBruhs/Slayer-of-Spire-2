from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def add_alias(aliases: dict[str, str], alias_text: str | None, card_id: str) -> None:
    key = normalize_alias_text(alias_text)
    if key:
        aliases[key] = card_id


RANDOM_HINTS = (
    "draw",
    "random",
    "discover",
    "create",
    "add a",
    "add ",
    "shuffle",
    "choose 1",
    "one of",
    "抽",
    "随机",
    "发现",
    "生成",
)


def _first_match_int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    if not match:
        return 0
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full card knowledge mapping from local Spire Codex card JSON files"
    )
    parser.add_argument(
        "--cards-json",
        action="append",
        default=[],
        help="Path to a card JSON file. Can be repeated.",
    )
    parser.add_argument(
        "--cards-glob",
        action="append",
        default=[],
        help="Glob pattern for card JSON files, e.g. 'E:/spire-codex/data/eng/cards*.json'",
    )
    parser.add_argument("--out", default="config/card_knowledge.local.json")
    return parser.parse_args()


def iter_card_entries(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(payload, dict):
        cards_node = payload.get("cards")
        if isinstance(cards_node, list):
            for item in cards_node:
                if isinstance(item, dict):
                    yield item
            return

        values = list(payload.values())
        if values and all(isinstance(value, dict) for value in values):
            for value in values:
                yield value


def extract_card_id(entry: dict[str, Any]) -> str:
    for key in ("id", "card_id", "key", "slug"):
        if key in entry:
            card_id = normalize_card_id(str(entry[key]))
            if card_id:
                return card_id
    return ""


def extract_int(entry: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
    return None


def extract_text(entry: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def extract_text_list(entry: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
    return []


def is_random_boundary(entry: dict[str, Any]) -> bool:
    text_parts: list[str] = []
    text_parts.extend(
        [
            extract_text(entry, ("description", "description_raw", "text", "rules_text")),
            extract_text(entry, ("name", "title", "display_name")),
        ]
    )
    text_parts.extend(extract_text_list(entry, ("keywords", "tags")))

    lowered = " ".join(part.lower() for part in text_parts if part)
    return any(hint in lowered for hint in RANDOM_HINTS)


def infer_effects(entry: dict[str, Any]) -> dict[str, int]:
    description = extract_text(entry, ("description", "description_raw", "text", "rules_text")).lower()

    damage = _first_match_int(r"deal\s+(\d+)\s+damage", description)
    if damage == 0:
        damage = _first_match_int(r"造成\s*(\d+)\s*点?伤害", description)

    block = _first_match_int(r"gain\s+(\d+)\s+block", description)
    if block == 0:
        block = _first_match_int(r"获得\s*(\d+)\s*点?格挡", description)

    draw = _first_match_int(r"draw\s+(\d+)\s+card", description)
    if draw == 0:
        draw = _first_match_int(r"抽\s*(\d+)\s*张?牌", description)

    energy_gain = _first_match_int(r"gain\s+(\d+)\s+energy", description)
    if energy_gain == 0:
        energy_gain = _first_match_int(r"获得\s*(\d+)\s*点?能量", description)

    vulnerable = _first_match_int(r"(\d+)\s+vulnerable", description)
    weak = _first_match_int(r"(\d+)\s+weak", description)
    frail = _first_match_int(r"(\d+)\s+frail", description)

    strength_delta = _first_match_int(r"gain\s+(\d+)\s+strength", description)
    dexterity_delta = _first_match_int(r"gain\s+(\d+)\s+dexterity", description)

    effects = {
        "damage": damage,
        "block": block,
        "draw": draw,
        "energy_gain": energy_gain,
        "vulnerable": vulnerable,
        "weak": weak,
        "frail": frail,
        "strength_delta": strength_delta,
        "dexterity_delta": dexterity_delta,
        "self_hp_loss": 0,
    }
    return {key: value for key, value in effects.items() if value != 0}


def collect_source_files(args: argparse.Namespace) -> list[Path]:
    paths: set[Path] = set()

    for raw in args.cards_json:
        candidate = Path(raw).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            paths.add(candidate)

    for pattern in args.cards_glob:
        for raw_match in glob.glob(pattern, recursive=True):
            candidate = Path(raw_match).expanduser().resolve()
            if candidate.exists() and candidate.is_file():
                paths.add(candidate)

    return sorted(paths)


def main() -> None:
    args = parse_args()
    source_files = collect_source_files(args)
    if not source_files:
        raise SystemExit("No source card JSON files found. Use --cards-json or --cards-glob.")

    costs: dict[str, int] = {}
    random_boundary_cards: set[str] = set()
    aliases: dict[str, str] = {}
    effects: dict[str, dict[str, int]] = {}

    for source_file in source_files:
        payload = json.loads(source_file.read_text(encoding="utf-8-sig"))

        for entry in iter_card_entries(payload):
            card_id = extract_card_id(entry)
            if not card_id:
                continue

            cost = extract_int(entry, ("cost", "base_cost", "baseCost", "energy", "energy_cost"))
            if cost is not None and cost >= 0:
                costs[card_id] = cost

            add_alias(aliases, card_id, card_id)
            add_alias(aliases, card_id.replace("_", " "), card_id)
            add_alias(aliases, card_id.replace("_", ""), card_id)
            add_alias(aliases, extract_text(entry, ("name", "title", "display_name")), card_id)

            for extra_alias in extract_text_list(entry, ("aliases", "alternate_names")):
                add_alias(aliases, extra_alias, card_id)

            inferred_effects = infer_effects(entry)
            if inferred_effects:
                effects[card_id] = inferred_effects

            if is_random_boundary(entry):
                random_boundary_cards.add(card_id)

    output = {
        "notes": "Generated from local Spire Codex card JSON files. You can still edit manually.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(path) for path in source_files],
        "costs": dict(sorted(costs.items())),
        "random_boundary_cards": sorted(random_boundary_cards),
        "aliases": dict(sorted(aliases.items())),
        "effects": dict(sorted(effects.items())),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"OK: {out_path}")
    print(f"source files: {len(source_files)}")
    print(f"cards with cost: {len(costs)}")
    print(f"random boundary cards: {len(random_boundary_cards)}")
    print(f"ocr aliases: {len(aliases)}")
    print(f"cards with inferred effects: {len(effects)}")


if __name__ == "__main__":
    main()

