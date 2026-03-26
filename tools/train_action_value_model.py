from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

from sos2_interface.policy.action_value_model import extract_features_from_compact
from sos2_interface.policy.tensorboard_logger import TensorboardLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear action-value model from action dataset")
    parser.add_argument("--dataset", default="runtime/action_value_dataset.jsonl")
    parser.add_argument("--trace-input", default="runtime/planner_action_trace.jsonl", help="Trace source used to auto-build dataset when missing")
    parser.add_argument("--dataset-gamma", type=float, default=0.95, help="Gamma used if auto-building dataset from traces")
    parser.add_argument("--no-auto-build-dataset", action="store_true", help="Disable auto-building dataset when --dataset is missing")
    parser.add_argument("--out", default="runtime/action_value_model.json")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", choices=["reward", "return"], default="return")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--tb-logdir", default="runtime/tensorboard/train", help="TensorBoard log directory")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists() and not args.no_auto_build_dataset:
        _auto_build_dataset(dataset_path, Path(args.trace_input), gamma=float(args.dataset_gamma))

    rows = load_rows(dataset_path, max_samples=args.max_samples)
    if not rows:
        raise RuntimeError("empty dataset; run build_action_dataset.py first")

    split = max(1, int(len(rows) * 0.9))
    train_rows = rows[:split]
    valid_rows = rows[split:]
    tb_logger = TensorboardLogger(None if args.no_tensorboard else args.tb_logdir)
    if not args.no_tensorboard and not tb_logger.enabled:
        print("tensorboard disabled: install tensorboardX to enable")

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
        tb_logger.add_scalar("train/train_rmse", train_rmse, epoch + 1)
        tb_logger.add_scalar("train/valid_rmse", valid_rmse, epoch + 1)
        tb_logger.add_scalar("train/feature_count", float(len(weights)), epoch + 1)

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
    tb_logger.close()


def _auto_build_dataset(dataset_path: Path, trace_input_path: Path, gamma: float) -> None:
    resolved_trace = _resolve_trace_input(trace_input_path)
    if resolved_trace is None:
        raise FileNotFoundError(
            f"dataset not found: {dataset_path}; trace source also missing: {trace_input_path}. "
            "Run tools/build_action_dataset.py first or pass --trace-input to an existing trace file."
        )
    if resolved_trace != trace_input_path:
        print(f"trace-input not found, using fallback trace: {resolved_trace}")

    root = Path(__file__).resolve().parents[1]
    build_script = root / "tools" / "build_action_dataset.py"
    cmd = [
        sys.executable,
        str(build_script),
        "--input",
        str(resolved_trace),
        "--output",
        str(dataset_path),
        "--gamma",
        str(gamma),
    ]
    print(f"dataset missing, auto-building: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "auto-build dataset failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if result.stdout.strip():
        print(result.stdout.strip())


def _resolve_trace_input(trace_input_path: Path) -> Path | None:
    if trace_input_path.exists():
        return trace_input_path

    # Only auto-fallback when caller uses the standard planner trace naming.
    if trace_input_path.name != "planner_action_trace.jsonl":
        return None

    runtime_dir = trace_input_path.parent
    if not runtime_dir.exists():
        return None

    candidates = sorted(runtime_dir.glob("planner_action_trace*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            if candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue
    return None


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

