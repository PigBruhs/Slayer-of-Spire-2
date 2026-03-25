from __future__ import annotations

import argparse
import json
from pathlib import Path

from sos2_interface.policy.action_value_model import ActionValueModel, extract_features_from_compact, load_action_value_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score action trace entries with a trained action-value model")
    parser.add_argument("--model", default="runtime/action_value_model.json")
    parser.add_argument("--trace", default="runtime/planner_action_trace.jsonl")
    parser.add_argument("--count", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_action_value_model(args.model)
    if model is None:
        raise FileNotFoundError(f"model not found or invalid: {args.model}")

    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise FileNotFoundError(f"trace not found: {trace_path}")

    shown = 0
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            before = row.get("before") if isinstance(row.get("before"), dict) else {}
            action = row.get("action") if isinstance(row.get("action"), dict) else {}
            score = score_compact(model, before, action)
            print(
                json.dumps(
                    {
                        "iteration": row.get("iteration"),
                        "action_type": action.get("action_type"),
                        "card_id": action.get("card_id"),
                        "option_index": action.get("option_index"),
                        "predicted_value": round(score, 4),
                    },
                    ensure_ascii=True,
                )
            )
            shown += 1
            if shown >= args.count:
                break


def score_compact(model: ActionValueModel, compact_state: dict[str, object], action_payload: dict[str, object]) -> float:
    features = extract_features_from_compact(compact_state, action_payload)
    total = model.bias
    for key, value in features.items():
        total += model.weights.get(key, 0.0) * value
    return total


if __name__ == "__main__":
    main()

