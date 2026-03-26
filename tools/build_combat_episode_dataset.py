from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combat episode dataset with HP-loss targets")
    parser.add_argument("--trace", default="runtime/planner_action_trace.selfplay.jsonl")
    parser.add_argument("--metrics", default="runtime/selfplay_metrics.jsonl")
    parser.add_argument("--out", default="runtime/combat_episode_dataset.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace)
    metrics_path = Path(args.metrics)
    out_path = Path(args.out)

    if not trace_path.exists():
        raise FileNotFoundError(f"trace not found: {trace_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics not found: {metrics_path}")

    episodes = load_episode_ranges(metrics_path)
    rows = load_trace_rows(trace_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row in rows:
            iteration = int(row.get("iteration") or -1)
            episode = find_episode(episodes, iteration)
            if episode is None:
                continue

            target = -float(episode.get("hp_loss", 0))
            sample = {
                "episode": int(episode.get("episode", 0)),
                "iteration": iteration,
                "before": row.get("before") or {},
                "action": row.get("action") or {},
                "target": target,
            }
            out.write(json.dumps(sample, ensure_ascii=True) + "\n")
            count += 1

    print(f"combat_dataset_written={out_path} samples={count} episodes={len(episodes)}")


def load_episode_ranges(path: Path) -> list[dict[str, object]]:
    episodes: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            episodes.append(item)

    episodes.sort(key=lambda e: int(e.get("episode", 0)))

    # Build iteration ranges using cycles count as fallback when exact boundaries are unavailable.
    cursor = 1
    for item in episodes:
        cycles = int(item.get("cycles", 0))
        item["iter_start"] = cursor
        item["iter_end"] = cursor + max(0, cycles) - 1
        cursor = int(item["iter_end"]) + 1
    return episodes


def load_trace_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    rows.sort(key=lambda r: int(r.get("iteration") or 0))
    return rows


def find_episode(episodes: list[dict[str, object]], iteration: int) -> dict[str, object] | None:
    for item in episodes:
        start = int(item.get("iter_start", -1))
        end = int(item.get("iter_end", -1))
        if start <= iteration <= end:
            return item
    return None


if __name__ == "__main__":
    main()

