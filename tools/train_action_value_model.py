from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from sos2_interface.policy.action_value_model import extract_features_from_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear action-value model from action dataset")
    parser.add_argument("--dataset", default="runtime/action_value_dataset.jsonl")
    parser.add_argument("--out", default="runtime/action_value_model.json")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", choices=["reward", "return"], default="return")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    rows = load_rows(Path(args.dataset), max_samples=args.max_samples)
    if not rows:
        raise RuntimeError("empty dataset; run build_action_dataset.py first")

    split = max(1, int(len(rows) * 0.9))
    train_rows = rows[:split]
    valid_rows = rows[split:]

    weights: dict[str, float] = {}
    bias = 0.0

    for epoch in range(args.epochs):
        random.shuffle(train_rows)
        total_loss = 0.0
        for row in train_rows:
            features = extract_features_from_compact(row["before"], row["action"])
            target = float(row[args.target])
            pred = bias + dot(weights, features)
            err = pred - target
            total_loss += err * err

            bias -= args.lr * err
            for key, value in features.items():
                w = weights.get(key, 0.0)
                grad = err * value + args.l2 * w
                weights[key] = w - args.lr * grad

        train_rmse = math.sqrt(total_loss / max(1, len(train_rows)))
        valid_rmse = evaluate_rmse(weights, bias, valid_rows, args.target)
        print(f"epoch={epoch + 1} train_rmse={train_rmse:.4f} valid_rmse={valid_rmse:.4f} features={len(weights)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_ms": int(time.time() * 1000),
        "model_type": "linear_regression_sgd",
        "target": args.target,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "l2": args.l2,
        "seed": args.seed,
        "bias": bias,
        "weights": weights,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"model_saved={out_path} features={len(weights)}")


def load_rows(path: Path, max_samples: int) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            if not isinstance(raw, dict):
                continue
            before = raw.get("before") if isinstance(raw.get("before"), dict) else {}
            action = raw.get("action") if isinstance(raw.get("action"), dict) else {}
            reward = float(raw.get("reward") or 0.0)
            ret = float(raw.get("return") or reward)
            rows.append({"before": before, "action": action, "reward": reward, "return": ret})
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def dot(weights: dict[str, float], features: dict[str, float]) -> float:
    total = 0.0
    for key, value in features.items():
        total += weights.get(key, 0.0) * value
    return total


def evaluate_rmse(weights: dict[str, float], bias: float, rows: list[dict[str, object]], target: str) -> float:
    if not rows:
        return 0.0

    loss = 0.0
    for row in rows:
        features = extract_features_from_compact(row["before"], row["action"])
        pred = bias + dot(weights, features)
        err = pred - float(row[target])
        loss += err * err
    return math.sqrt(loss / len(rows))


if __name__ == "__main__":
    main()

