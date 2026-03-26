from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_WEIGHTS = {
    "damage": 1.0,
    "block": 0.9,
    "draw": 0.6,
    "energy_gain": 2.0,
    "strength_delta": 2.0,
    "dexterity_delta": 1.5,
    "vulnerable": 1.0,
    "weak": 1.0,
    "frail": 0.5,
    "self_hp_loss": 1.5,
    "incoming_damage": 1.25,
    "end_turn_bonus": 0.6,
    "no_end_turn_penalty": 0.3,
    "premature_end_turn_penalty": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train branch factor weights from action-value dataset")
    parser.add_argument("--dataset", default="runtime/action_value_dataset.jsonl")
    parser.add_argument("--out", default="runtime/branch_factor_weights.json")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.dataset))
    if not rows:
        raise RuntimeError("empty dataset")

    # Train only factors we can reliably observe from traces.
    train_keys = ["damage", "block", "self_hp_loss", "end_turn_bonus"]
    weights = dict(DEFAULT_WEIGHTS)

    for _ in range(args.epochs):
        for row in rows:
            feats = row["features"]
            target = row["target"]
            pred = sum(weights[k] * feats.get(k, 0.0) for k in train_keys)
            err = pred - target
            for key in train_keys:
                grad = err * feats.get(key, 0.0) + args.l2 * weights[key]
                weights[key] -= args.lr * grad
                # Keep weights numerically stable and interpretable.
                weights[key] = max(0.0, min(10.0, weights[key]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(weights, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"branch_weight_saved={out_path}")


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                continue

            transition = row.get("transition") if isinstance(row.get("transition"), dict) else {}
            action = row.get("action") if isinstance(row.get("action"), dict) else {}

            enemy_damage = 0.0
            enemy_rows = transition.get("enemy_hp_delta") if isinstance(transition.get("enemy_hp_delta"), list) else []
            for item in enemy_rows:
                if isinstance(item, dict):
                    hp_delta = float(item.get("hp_delta") or 0.0)
                    enemy_damage += max(0.0, -hp_delta)

            block_gain = max(0.0, float(transition.get("player_block_delta") or 0.0))
            hp_delta = float(transition.get("player_hp_delta") or 0.0)
            self_hp_loss = max(0.0, -hp_delta)
            end_turn = 1.0 if str(action.get("action_type") or "") == "end_turn" else 0.0

            target = float(row.get("return") or row.get("reward") or 0.0)
            rows.append(
                {
                    "features": {
                        "damage": enemy_damage,
                        "block": block_gain,
                        "self_hp_loss": self_hp_loss,
                        "end_turn_bonus": end_turn,
                    },
                    "target": target,
                }
            )
    return rows


if __name__ == "__main__":
    main()

