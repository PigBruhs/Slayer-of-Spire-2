from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local card knowledge file")
    parser.add_argument("--file", default="config/card_knowledge.local.json")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"File does not exist: {path}")

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    costs = data.get("costs", {})
    random_boundary_cards = data.get("random_boundary_cards", [])
    aliases = data.get("aliases", {})
    effects = data.get("effects", {})

    if not isinstance(costs, dict):
        raise SystemExit("Invalid schema: 'costs' must be an object")
    if not isinstance(random_boundary_cards, list):
        raise SystemExit("Invalid schema: 'random_boundary_cards' must be a list")
    if not isinstance(aliases, dict):
        raise SystemExit("Invalid schema: 'aliases' must be an object")
    if not isinstance(effects, dict):
        raise SystemExit("Invalid schema: 'effects' must be an object")

    invalid_costs = [name for name, value in costs.items() if not isinstance(value, int) or value < 0]
    if invalid_costs:
        raise SystemExit(f"Invalid cost entries: {invalid_costs}")

    invalid_random_cards = [item for item in random_boundary_cards if not isinstance(item, str) or not item.strip()]
    if invalid_random_cards:
        raise SystemExit("Invalid random_boundary_cards entries: every item must be a non-empty string")

    invalid_aliases = [
        alias for alias, card_id in aliases.items() if not isinstance(alias, str) or not alias.strip() or not isinstance(card_id, str)
    ]
    if invalid_aliases:
        raise SystemExit(f"Invalid alias entries: {invalid_aliases[:10]}")

    allowed_effect_keys = {
        "damage",
        "block",
        "draw",
        "energy_gain",
        "self_hp_loss",
        "strength_delta",
        "dexterity_delta",
        "vulnerable",
        "weak",
        "frail",
    }
    for card_id, payload in effects.items():
        if not isinstance(card_id, str) or not card_id.strip() or not isinstance(payload, dict):
            raise SystemExit(f"Invalid effects entry for card: {card_id}")
        unknown_keys = [key for key in payload.keys() if key not in allowed_effect_keys]
        if unknown_keys:
            raise SystemExit(f"Unknown effects keys for {card_id}: {unknown_keys}")
        invalid_values = [key for key, value in payload.items() if not isinstance(value, int)]
        if invalid_values:
            raise SystemExit(f"Non-integer effects values for {card_id}: {invalid_values}")

    print(f"OK: {path}")
    print(f"cards with explicit costs: {len(costs)}")
    print(f"random boundary cards: {len(random_boundary_cards)}")
    print(f"ocr aliases: {len(aliases)}")
    print(f"cards with effects: {len(effects)}")


if __name__ == "__main__":
    main()

