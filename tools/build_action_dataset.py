from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build supervised action-value dataset from planner action traces")
    parser.add_argument("--input", default="runtime/planner_action_trace.jsonl", help="Action trace JSONL input")
    parser.add_argument("--output", default="runtime/action_value_dataset.jsonl", help="Output dataset JSONL")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor used for return target")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input trace file not found: {input_path}")

    groups: dict[int, list[dict[str, object]]] = {}
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                continue
            iteration = row.get("iteration")
            if not isinstance(iteration, int):
                continue
            groups.setdefault(iteration, []).append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for iteration, rows in sorted(groups.items(), key=lambda item: item[0]):
            rows.sort(key=lambda row: int(row.get("timestamp_ms") or 0))
            rewards = [immediate_reward(row) for row in rows]
            returns = discounted_returns(rewards, gamma=float(args.gamma))

            for idx, row in enumerate(rows):
                sample = {
                    "iteration": iteration,
                    "step": idx,
                    "before": row.get("before") or {},
                    "after": row.get("after") or {},
                    "action": row.get("action") or {},
                    "transition": row.get("transition") or {},
                    "reward": rewards[idx],
                    "return": returns[idx],
                }
                out.write(json.dumps(sample, ensure_ascii=True) + "\n")
                sample_count += 1

    print(f"dataset_written={output_path} samples={sample_count} episodes={len(groups)}")


def immediate_reward(row: dict[str, object]) -> float:
    transition = row.get("transition") if isinstance(row.get("transition"), dict) else {}
    enemy_hp_delta_rows = transition.get("enemy_hp_delta") if isinstance(transition.get("enemy_hp_delta"), list) else []

    enemy_damage = 0.0
    enemy_kills = 0.0
    for item in enemy_hp_delta_rows:
        if not isinstance(item, dict):
            continue
        hp_delta = float(item.get("hp_delta") or 0.0)
        enemy_damage += max(0.0, -hp_delta)
        if bool(item.get("died")):
            enemy_kills += 1.0

    player_hp_delta = float(transition.get("player_hp_delta") or 0.0)
    player_block_delta = float(transition.get("player_block_delta") or 0.0)
    player_energy_delta = float(transition.get("player_energy_delta") or 0.0)
    hand_delta = float(transition.get("hand_count_delta") or 0.0)
    warning_count = len(transition.get("new_warnings") or []) if isinstance(transition.get("new_warnings"), list) else 0

    reward = 0.0
    reward += enemy_damage * 1.0
    reward += enemy_kills * 6.0
    reward += max(0.0, player_block_delta) * 0.35
    reward += player_hp_delta * 2.5
    reward += player_energy_delta * 0.2
    reward += -abs(min(0.0, hand_delta)) * 0.05
    reward -= float(warning_count) * 1.5

    if bool(transition.get("combat_ended")) and not bool(transition.get("player_died")):
        reward += 8.0
    if bool(transition.get("player_died")):
        reward -= 40.0

    return reward


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    out = [0.0] * len(rewards)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + gamma * running
        out[idx] = running
    return out


if __name__ == "__main__":
    main()

