from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from sos2_interface.contracts.action import ActionCommand
from sos2_interface.contracts.state import GameStateSnapshot, PlayerState
from sos2_interface.policy.combat_policy_model import extract_feature_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train larger combat policy model (MLP) with HP-loss objective")
    parser.add_argument("--dataset", default="runtime/combat_episode_dataset.jsonl")
    parser.add_argument("--out", default="runtime/combat_policy_model.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = load_rows(Path(args.dataset), args.input_dim)
    if not rows:
        raise RuntimeError("empty combat dataset")

    split = max(1, int(len(rows) * 0.9))
    train = rows[:split]
    valid = rows[split:]

    in_dim = args.input_dim
    h1 = args.hidden1
    h2 = args.hidden2

    w1 = np.random.randn(in_dim, h1).astype(np.float32) * 0.02
    b1 = np.zeros((h1,), dtype=np.float32)
    w2 = np.random.randn(h1, h2).astype(np.float32) * 0.02
    b2 = np.zeros((h2,), dtype=np.float32)
    w3 = np.random.randn(h2, 1).astype(np.float32) * 0.02
    b3 = np.zeros((1,), dtype=np.float32)

    for epoch in range(args.epochs):
        random.shuffle(train)
        loss_sum = 0.0

        for x, y in train:
            x = x.reshape(1, -1)
            y_t = np.array([[y]], dtype=np.float32)

            z1 = x @ w1 + b1
            h_1 = np.maximum(0.0, z1)
            z2 = h_1 @ w2 + b2
            h_2 = np.maximum(0.0, z2)
            out = h_2 @ w3 + b3

            err = out - y_t
            loss_sum += float(err[0, 0] ** 2)

            # Backprop (MSE)
            d_out = 2.0 * err
            d_w3 = h_2.T @ d_out
            d_b3 = d_out[0]

            d_h2 = d_out @ w3.T
            d_z2 = d_h2 * (z2 > 0)
            d_w2 = h_1.T @ d_z2
            d_b2 = d_z2[0]

            d_h1 = d_z2 @ w2.T
            d_z1 = d_h1 * (z1 > 0)
            d_w1 = x.T @ d_z1
            d_b1 = d_z1[0]

            w3 -= args.lr * d_w3
            b3 -= args.lr * d_b3
            w2 -= args.lr * d_w2
            b2 -= args.lr * d_b2
            w1 -= args.lr * d_w1
            b1 -= args.lr * d_b1

        train_rmse = math.sqrt(loss_sum / max(1, len(train)))
        valid_rmse = evaluate(valid, w1, b1, w2, b2, w3, b3)
        print(f"epoch={epoch + 1} train_rmse={train_rmse:.4f} valid_rmse={valid_rmse:.4f}")

    payload = {
        "created_at_ms": int(time.time() * 1000),
        "model_type": "mlp_combat_policy",
        "objective": "minimize_run_hp_loss",
        "input_dim": in_dim,
        "hidden1": h1,
        "hidden2": h2,
        "w1": w1.tolist(),
        "b1": b1.tolist(),
        "w2": w2.tolist(),
        "b2": b2.tolist(),
        "w3": w3.tolist(),
        "b3": b3.tolist(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    print(f"combat_model_saved={out_path}")


def load_rows(path: Path, input_dim: int) -> list[tuple[np.ndarray, float]]:
    if not path.exists():
        raise FileNotFoundError(f"combat dataset not found: {path}")

    rows: list[tuple[np.ndarray, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if not isinstance(item, dict):
                continue

            before = item.get("before") if isinstance(item.get("before"), dict) else {}
            action = item.get("action") if isinstance(item.get("action"), dict) else {}
            target = float(item.get("target") or 0.0)

            dummy_state = _to_snapshot(before)
            dummy_action = ActionCommand(
                action_type=str(action.get("action_type") or "noop"),
                card_id=action.get("card_id"),
                target_id=action.get("target_id"),
                option_index=action.get("option_index"),
                metadata=action.get("metadata") if isinstance(action.get("metadata"), dict) else {},
            )
            vec = extract_feature_vector(dummy_state, dummy_action, input_dim)
            rows.append((vec.astype(np.float32), target))
    return rows


def _to_snapshot(before: dict[str, object]) -> GameStateSnapshot:
    player = before.get("player") if isinstance(before.get("player"), dict) else {}
    return GameStateSnapshot(
        source="mock",
        frame_id=int(before.get("frame_id") or 0),
        timestamp_ms=0,
        in_combat=bool(before.get("in_combat")),
        in_event=bool(before.get("in_event")),
        turn=int(before.get("turn") or 0) if before.get("turn") is not None else None,
        player=PlayerState(
            hp=int(player.get("hp") or 0),
            max_hp=int(player.get("max_hp") or max(1, int(player.get("hp") or 0))),
            block=int(player.get("block") or 0),
            energy=int(player.get("energy") or 0),
            hand=[],
            draw_pile_count=int(player.get("draw_pile_count") or 0),
            discard_pile_count=int(player.get("discard_pile_count") or 0),
        ),
        enemies=[],
        warnings=[],
    )


def evaluate(rows: list[tuple[np.ndarray, float]], w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, w3: np.ndarray, b3: np.ndarray) -> float:
    if not rows:
        return 0.0
    loss = 0.0
    for x, y in rows:
        z1 = x.reshape(1, -1) @ w1 + b1
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ w2 + b2
        h2 = np.maximum(0.0, z2)
        out = h2 @ w3 + b3
        err = float(out[0, 0] - y)
        loss += err * err
    return math.sqrt(loss / len(rows))


if __name__ == "__main__":
    main()

