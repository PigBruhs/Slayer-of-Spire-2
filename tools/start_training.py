from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-file offline trainer with explicit metric summary")
    parser.add_argument("--trace", default="runtime/planner_action_trace.selfplay.jsonl", help="Action trace input")
    parser.add_argument("--metrics", default="runtime/selfplay_metrics.jsonl", help="Episode metrics for combat-episode dataset")
    parser.add_argument("--dataset", default="runtime/action_value_dataset.autobuild.jsonl", help="Action-value dataset output")
    parser.add_argument("--combat-dataset", default="runtime/combat_episode_dataset.autobuild.jsonl", help="Combat episode dataset output")
    parser.add_argument("--action-model", default="runtime/action_value_model.autobuild.json", help="Action-value model output")
    parser.add_argument("--combat-model", default="runtime/combat_policy_model.autobuild.json", help="Combat model output")
    parser.add_argument("--noncombat-model", default="runtime/noncombat_policy_model.autobuild.json", help="Non-combat model output")
    parser.add_argument("--branch-weight", default="runtime/branch_factor_weights.autobuild.json", help="Branch-weight model output")
    parser.add_argument("--summary", default="runtime/training_summary.json", help="Latest run summary JSON")
    parser.add_argument("--summary-log", default="runtime/training_metrics.jsonl", help="Append-only summary log JSONL")
    parser.add_argument("--dataset-gamma", type=float, default=0.95)
    parser.add_argument("--action-epochs", type=int, default=8)
    parser.add_argument("--action-lr", type=float, default=0.03)
    parser.add_argument("--combat-epochs", type=int, default=10)
    parser.add_argument("--dual-epochs", type=int, default=10)
    parser.add_argument("--branch-epochs", type=int, default=6)
    parser.add_argument("--skip-action", action="store_true", help="Skip action-value training")
    parser.add_argument("--skip-combat", action="store_true", help="Skip combat-policy training")
    parser.add_argument("--skip-dual", action="store_true", help="Skip dual RL combat/noncombat training")
    parser.add_argument("--skip-branch", action="store_true", help="Skip branch-weight training")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    started = time.time()

    trace_path = _resolve_repo_path(args.trace, repo_root)
    if not trace_path.exists():
        raise FileNotFoundError(f"trace not found: {trace_path}")

    steps = _build_steps(args, repo_root)
    if args.dry_run:
        print("[trainer] dry-run steps:")
        for step in steps:
            print(f"  - {step['name']}: {' '.join(step['cmd'])}")
        return

    run_records: list[dict[str, object]] = []
    for step in steps:
        record = _run_step(step_name=str(step["name"]), cmd=list(step["cmd"]), cwd=repo_root)
        run_records.append(record)
        if int(record.get("returncode", 1)) != 0:
            _finalize_and_write_summary(args, started, run_records, repo_root)
            raise RuntimeError(f"step failed: {step['name']}")

    summary = _finalize_and_write_summary(args, started, run_records, repo_root)
    _print_final_summary(summary)


def _resolve_repo_path(path_like: str, repo_root: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (repo_root / path)


def _build_steps(args: argparse.Namespace, repo_root: Path) -> list[dict[str, object]]:
    tools_dir = repo_root / "tools"
    py = sys.executable

    steps: list[dict[str, object]] = [
        {
            "name": "build_action_dataset",
            "cmd": [
                py,
                str(tools_dir / "build_action_dataset.py"),
                "--input",
                args.trace,
                "--output",
                args.dataset,
                "--gamma",
                str(args.dataset_gamma),
            ],
        }
    ]

    if not args.skip_action:
        steps.append(
            {
                "name": "train_action_value_model",
                "cmd": [
                    py,
                    str(tools_dir / "train_action_value_model.py"),
                    "--dataset",
                    args.dataset,
                    "--out",
                    args.action_model,
                    "--epochs",
                    str(max(1, int(args.action_epochs))),
                    "--lr",
                    str(float(args.action_lr)),
                    "--target",
                    "return",
                ],
            }
        )

    if not args.skip_combat:
        steps.append(
            {
                "name": "build_combat_episode_dataset",
                "cmd": [
                    py,
                    str(tools_dir / "build_combat_episode_dataset.py"),
                    "--trace",
                    args.trace,
                    "--metrics",
                    args.metrics,
                    "--out",
                    args.combat_dataset,
                ],
            }
        )
        steps.append(
            {
                "name": "train_combat_policy_model",
                "cmd": [
                    py,
                    str(tools_dir / "train_combat_policy_model.py"),
                    "--dataset",
                    args.combat_dataset,
                    "--out",
                    args.combat_model,
                    "--epochs",
                    str(max(1, int(args.combat_epochs))),
                ],
            }
        )

    if not args.skip_dual:
        steps.append(
            {
                "name": "train_dual_rl_models",
                "cmd": [
                    py,
                    str(tools_dir / "train_dual_rl_models.py"),
                    "--trace",
                    args.trace,
                    "--combat-out",
                    args.combat_model,
                    "--noncombat-out",
                    args.noncombat_model,
                    "--epochs",
                    str(max(1, int(args.dual_epochs))),
                ],
            }
        )

    if not args.skip_branch:
        steps.append(
            {
                "name": "train_branch_factor_weights",
                "cmd": [
                    py,
                    str(tools_dir / "train_branch_factor_weights.py"),
                    "--dataset",
                    args.dataset,
                    "--out",
                    args.branch_weight,
                    "--epochs",
                    str(max(1, int(args.branch_epochs))),
                ],
            }
        )

    return steps


def _run_step(step_name: str, cmd: list[str], cwd: Path) -> dict[str, object]:
    print(f"[trainer] start: {step_name}")
    print(f"[trainer] cmd: {' '.join(cmd)}")

    started = time.time()
    lines: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.rstrip("\r\n")
        lines.append(text)
        print(f"[{step_name}] {text}")

    returncode = proc.wait()
    duration_sec = round(max(0.0, time.time() - started), 3)
    print(f"[trainer] done: {step_name} returncode={returncode} duration_sec={duration_sec}")

    record = {
        "name": step_name,
        "cmd": cmd,
        "returncode": int(returncode),
        "duration_sec": duration_sec,
        "metrics": _extract_metrics(step_name, lines),
        "artifacts": _extract_artifacts(lines),
    }
    return record


def _extract_metrics(step_name: str, lines: list[str]) -> dict[str, object]:
    if step_name == "train_action_value_model":
        return _extract_epoch_metrics(lines, r"epoch=(\d+) train_rmse=([0-9.]+) valid_rmse=([0-9.]+)", ["epoch", "train_rmse", "valid_rmse"])
    if step_name == "train_combat_policy_model":
        return _extract_epoch_metrics(lines, r"epoch=(\d+) train_rmse=([0-9.]+) valid_rmse=([0-9.]+)", ["epoch", "train_rmse", "valid_rmse"])
    if step_name == "train_dual_rl_models":
        payload: dict[str, object] = {}
        for tag in ["combat", "noncombat"]:
            found = _extract_epoch_metrics(
                lines,
                rf"\[{tag}\] epoch=(\d+) train_huber=([0-9.]+) valid_rmse=([0-9.]+)",
                ["epoch", "train_huber", "valid_rmse"],
            )
            if found:
                payload[tag] = found
        return payload
    return {}


def _extract_epoch_metrics(lines: list[str], pattern: str, keys: list[str]) -> dict[str, object]:
    regex = re.compile(pattern)
    best: dict[str, object] = {}
    for line in lines:
        m = regex.search(line)
        if not m:
            continue
        values = list(m.groups())
        row: dict[str, object] = {}
        for key, value in zip(keys, values):
            if key == "epoch":
                row[key] = int(value)
            else:
                row[key] = float(value)
        best = row
    return best


def _extract_artifacts(lines: list[str]) -> list[str]:
    out: list[str] = []
    patterns = [
        re.compile(r"(?:model_saved|combat_model_saved|branch_weight_saved|dataset_written|combat_dataset_written)=([^\s]+)"),
    ]
    for line in lines:
        for regex in patterns:
            m = regex.search(line)
            if m:
                out.append(m.group(1).strip())
    return out


def _model_param_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    if all(key in payload for key in ["w1", "w2", "w3", "b1", "b2", "b3"]):
        total = 0
        for key in ["w1", "w2", "w3", "b1", "b2", "b3"]:
            total += _nested_list_size(payload.get(key))
        return int(total)

    if isinstance(payload.get("state_dict"), dict):
        total = 0
        for value in payload["state_dict"].values():
            total += _nested_list_size(value)
        return int(total)

    if isinstance(payload.get("weights"), dict):
        return int(len(payload["weights"])) + 1

    return None


def _nested_list_size(value: object) -> int:
    if isinstance(value, list):
        if not value:
            return 0
        return sum(_nested_list_size(item) for item in value)
    return 1


def _finalize_and_write_summary(
    args: argparse.Namespace,
    started: float,
    run_records: list[dict[str, object]],
    repo_root: Path,
) -> dict[str, object]:
    finished_ms = int(time.time() * 1000)
    summary = {
        "created_at_ms": finished_ms,
        "duration_sec": round(max(0.0, time.time() - started), 3),
        "status": "ok" if all(int(item.get("returncode", 1)) == 0 for item in run_records) else "failed",
        "inputs": {
            "trace": args.trace,
            "metrics": args.metrics,
            "dataset": args.dataset,
            "combat_dataset": args.combat_dataset,
        },
        "outputs": {
            "action_model": args.action_model,
            "combat_model": args.combat_model,
            "noncombat_model": args.noncombat_model,
            "branch_weight": args.branch_weight,
        },
        "steps": run_records,
        "model_params": {
            "action_model": _model_param_count(_resolve_repo_path(args.action_model, repo_root)),
            "combat_model": _model_param_count(_resolve_repo_path(args.combat_model, repo_root)),
            "noncombat_model": _model_param_count(_resolve_repo_path(args.noncombat_model, repo_root)),
        },
    }

    summary_path = _resolve_repo_path(args.summary, repo_root)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    log_path = _resolve_repo_path(args.summary_log, repo_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=True) + "\n")

    print(f"[trainer] summary_written={summary_path}")
    print(f"[trainer] summary_appended={log_path}")
    return summary


def _print_final_summary(summary: dict[str, object]) -> None:
    print("[trainer] ===== FINAL METRICS =====")
    print(f"[trainer] status={summary.get('status')} duration_sec={summary.get('duration_sec')}")

    steps = summary.get("steps") if isinstance(summary.get("steps"), list) else []
    for step in steps:
        if not isinstance(step, dict):
            continue
        name = str(step.get("name") or "")
        rc = int(step.get("returncode") or 0)
        metrics = step.get("metrics") if isinstance(step.get("metrics"), dict) else {}
        print(f"[trainer] step={name} returncode={rc} metrics={json.dumps(metrics, ensure_ascii=True)}")

    params = summary.get("model_params") if isinstance(summary.get("model_params"), dict) else {}
    print(
        "[trainer] model_params "
        f"action={params.get('action_model')} "
        f"combat={params.get('combat_model')} "
        f"noncombat={params.get('noncombat_model')}"
    )


if __name__ == "__main__":
    main()

