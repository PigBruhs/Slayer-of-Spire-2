from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import psutil

from sos2_interface.actions.mcp_post_executor import McpPostActionExecutor, McpPostExecutorConfig
from sos2_interface.policy.planner_loop import PlannerLoop, SegmentPlanner
from sos2_interface.policy.tensorboard_logger import TensorboardLogger
from sos2_interface.readers.mcp_api_reader import McpApiReader, McpApiReaderConfig


@dataclass
class EpisodeStats:
    episode: int
    win: bool
    duration_sec: float
    cycles: int
    max_act: int
    max_floor: int
    saw_act3_boss: bool
    saw_player_death: bool
    start_hp: int
    end_hp: int
    hp_loss: int
    model_path: str | None


class EpisodeTracker:
    def __init__(self) -> None:
        self.max_act = 0
        self.max_floor = 0
        self.saw_act3_boss = False
        self.saw_terminal_win = False
        self.saw_player_death = False
        self.cycles = 0
        self.start_hp: int | None = None
        self.end_hp: int | None = None
        self._start_ts = time.time()

    def observe(self, state) -> None:
        self.cycles += 1
        raw = state.raw_state if isinstance(state.raw_state, dict) else {}
        run = raw.get("run") if isinstance(raw.get("run"), dict) else {}

        act = _to_int(run.get("act"))
        floor = _to_int(run.get("floor"))
        self.max_act = max(self.max_act, act)
        self.max_floor = max(self.max_floor, floor)

        if state.state_type == "boss" and act >= 3:
            self.saw_act3_boss = True
        if _is_terminal_win_state(state, self.saw_act3_boss):
            self.saw_terminal_win = True
        if state.player.hp <= 0:
            self.saw_player_death = True

        if self.start_hp is None and state.player.hp > 0:
            self.start_hp = int(state.player.hp)
        self.end_hp = int(state.player.hp)

    def finalize(self, episode_number: int, model_path: str | None) -> EpisodeStats:
        win = self.saw_terminal_win and (not self.saw_player_death)
        start_hp = self.start_hp if self.start_hp is not None else 0
        end_hp = self.end_hp if self.end_hp is not None else 0
        hp_loss = max(0, start_hp - end_hp)
        return EpisodeStats(
            episode=episode_number,
            win=win,
            duration_sec=max(0.0, time.time() - self._start_ts),
            cycles=self.cycles,
            max_act=self.max_act,
            max_floor=self.max_floor,
            saw_act3_boss=self.saw_act3_boss,
            saw_player_death=self.saw_player_death,
            start_hp=start_hp,
            end_hp=end_hp,
            hp_loss=hp_loss,
            model_path=model_path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto self-play training loop with MCP API")
    parser.add_argument("--game-exe", default=None, help="Path to SlayTheSpire2.exe (optional if game already running)")
    parser.add_argument("--process-name", default="SlayTheSpire2.exe")
    parser.add_argument("--mcp-host", default="127.0.0.1")
    parser.add_argument("--mcp-port", type=int, default=15526)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--interval-ms", type=int, default=0, help="Optional extra delay after each cycle (default 0, completion-driven)")
    parser.add_argument("--play-card-interval-ms", type=int, default=500, help="Additional play_card-only delay (global operation delay has hard minimum 500ms)")
    parser.add_argument("--wait-play-phase", action="store_true", default=True)
    parser.add_argument("--no-wait-play-phase", action="store_false", dest="wait_play_phase")
    parser.add_argument("--combat-ready-timeout-ms", type=int, default=1500)
    parser.add_argument("--trace-file", default="runtime/planner_cycles.selfplay.jsonl")
    parser.add_argument("--action-trace-file", default="runtime/planner_action_trace.selfplay.jsonl")
    parser.add_argument("--metrics-file", default="runtime/selfplay_metrics.jsonl")
    parser.add_argument("--dataset-file", default="runtime/action_value_dataset.selfplay.jsonl")
    parser.add_argument("--combat-dataset-file", default="runtime/combat_episode_dataset.jsonl")
    parser.add_argument("--model-file", default="runtime/action_value_model.selfplay.json")
    parser.add_argument("--combat-model-file", default="runtime/combat_policy_model.json")
    parser.add_argument("--noncombat-model-file", default="runtime/noncombat_policy_model.json")
    parser.add_argument("--retrain-every", type=int, default=5)
    parser.add_argument("--value-model-weight", type=float, default=0.35)
    parser.add_argument("--combat-model-weight", type=float, default=0.8)
    parser.add_argument("--max-segment-actions", type=int, default=0, help="0 means explore full deterministic turn segment")
    parser.add_argument("--max-branches", type=int, default=3000)
    parser.add_argument("--branch-weight-file", default="runtime/branch_factor_weights.json")
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--train-epochs", type=int, default=8)
    parser.add_argument("--train-lr", type=float, default=0.03)
    parser.add_argument("--tb-logdir", default="runtime/tensorboard/selfplay", help="TensorBoard log directory")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--combat-only", action="store_true", default=True, help="Only automate combat decisions")
    parser.add_argument("--no-combat-only", action="store_false", dest="combat_only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        print("[auto-train] episodes <= 0, nothing to run")
        return

    _ensure_game_running(args.process_name, args.game_exe)

    reader = McpApiReader(McpApiReaderConfig(host=args.mcp_host, port=args.mcp_port, mode="singleplayer"))
    executor = McpPostActionExecutor(
        McpPostExecutorConfig(host=args.mcp_host, port=args.mcp_port, mode="singleplayer"),
        allow_live_actions=True,
    )

    model_path = Path(args.model_file) if Path(args.model_file).exists() else None

    loop = _build_planner_loop(args, reader, executor, model_path)
    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorboardLogger(None if args.no_tensorboard else args.tb_logdir)
    if not args.no_tensorboard and not tb_logger.enabled:
        print("[auto-train] TensorBoard logging disabled (install tensorboardX to enable)")

    wins = deque(maxlen=max(1, args.rolling_window))
    episode_count = 0
    tracker: EpisodeTracker | None = None
    waiting_menu_logged = False
    waiting_death_logged = False

    print("[auto-train] run start/menu reset are user-managed in this version; script only plays active runs.")

    while episode_count < args.episodes:
        state = reader.read_state()

        if state.state_type == "menu":
            if tracker is not None:
                stats = tracker.finalize(episode_number=episode_count, model_path=str(model_path) if model_path else None)
                wins.append(1 if stats.win else 0)
                rolling = sum(wins) / float(len(wins)) if wins else 0.0
                _append_metric(metrics_path, stats, rolling)
                print(
                    f"[auto-train] episode={stats.episode} win={stats.win} max_act={stats.max_act} "
                    f"max_floor={stats.max_floor} hp_loss={stats.hp_loss} cycles={stats.cycles} duration={stats.duration_sec:.1f}s "
                    f"rolling_win_rate={rolling:.3f}"
                )
                tb_logger.add_scalar("selfplay/win", 1.0 if stats.win else 0.0, stats.episode)
                tb_logger.add_scalar("selfplay/rolling_win_rate", rolling, stats.episode)
                tb_logger.add_scalar("selfplay/max_act", float(stats.max_act), stats.episode)
                tb_logger.add_scalar("selfplay/max_floor", float(stats.max_floor), stats.episode)
                tb_logger.add_scalar("selfplay/hp_loss", float(stats.hp_loss), stats.episode)
                tb_logger.add_scalar("selfplay/cycles", float(stats.cycles), stats.episode)
                tb_logger.add_scalar("selfplay/duration_sec", float(stats.duration_sec), stats.episode)

                if args.retrain_every > 0 and episode_count % args.retrain_every == 0:
                    trained = _retrain_model(args)
                    if trained:
                        model_path = Path(args.model_file)
                        loop = _build_planner_loop(args, reader, executor, model_path)
                        print(f"[auto-train] model updated: {model_path}")

                tracker = None

            if not waiting_menu_logged:
                print("[auto-train] waiting in menu; please start next run manually")
                waiting_menu_logged = True

            time.sleep(1.0)
            continue

        waiting_menu_logged = False

        if state.player.hp <= 0:
            if not waiting_death_logged:
                print("[auto-train] player dead detected; waiting for run to transition to menu")
                waiting_death_logged = True
            time.sleep(0.5)
            continue

        waiting_death_logged = False

        if tracker is None:
            episode_count += 1
            tracker = EpisodeTracker()
            print(f"[auto-train] episode {episode_count} started")

        tracker.observe(state)
        cycle = loop.run_once()
        print(
            f"[auto-train] cycle={cycle.iteration} decision={cycle.decision_summary} "
            f"planned={cycle.planned_actions} executed={cycle.executed_actions}"
        )
        if cycle.boundary_reason.startswith("episode_terminal:"):
            print(f"[auto-train] terminal cycle boundary={cycle.boundary_reason}; waiting for menu transition")
            time.sleep(0.5)
            continue
        if cycle.boundary_reason.startswith("soft_loop_detected"):
            print(f"[auto-train] soft loop boundary={cycle.boundary_reason}; pausing before retry")
            time.sleep(1.0)
            continue
        if cycle.warnings:
            print(f"[auto-train] warnings: {cycle.warnings}")
        if args.interval_ms > 0:
            time.sleep(args.interval_ms / 1000.0)

    print("[auto-train] completed requested episode count")
    tb_logger.close()


def _build_planner_loop(args: argparse.Namespace, reader: McpApiReader, executor: McpPostActionExecutor, model_path: Path | None) -> PlannerLoop:
    planner = SegmentPlanner(
        max_segment_actions=args.max_segment_actions,
        max_branches=args.max_branches,
        value_model_path=str(model_path) if model_path else None,
        value_model_weight=args.value_model_weight,
        combat_model_path=args.combat_model_file,
        combat_model_weight=args.combat_model_weight,
        noncombat_model_path=args.noncombat_model_file,
        noncombat_model_weight=1.0,
        enable_noncombat_policy=not bool(args.combat_only),
        branch_weight_path=args.branch_weight_file,
    )
    return PlannerLoop(
        reader=reader,
        executor=executor,
        planner=planner,
        trace_file=args.trace_file,
        action_trace_file=args.action_trace_file,
        capture_action_trace=True,
        include_raw_state_in_action_trace=False,
        min_play_card_interval_ms=args.play_card_interval_ms,
        wait_for_play_phase_before_card=bool(args.wait_play_phase),
        combat_ready_timeout_ms=args.combat_ready_timeout_ms,
        combat_only=bool(args.combat_only),
    )


def _append_metric(path: Path, stats: EpisodeStats, rolling_win_rate: float) -> None:
    payload = {
        "timestamp_ms": int(time.time() * 1000),
        "episode": stats.episode,
        "win": stats.win,
        "duration_sec": round(stats.duration_sec, 3),
        "cycles": stats.cycles,
        "max_act": stats.max_act,
        "max_floor": stats.max_floor,
        "saw_act3_boss": stats.saw_act3_boss,
        "saw_player_death": stats.saw_player_death,
        "start_hp": stats.start_hp,
        "end_hp": stats.end_hp,
        "hp_loss": stats.hp_loss,
        "rolling_win_rate": round(rolling_win_rate, 4),
        "model_path": stats.model_path,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _retrain_model(args: argparse.Namespace) -> bool:
    root = Path(__file__).resolve().parents[1]
    dataset_cmd = [
        sys.executable,
        str(root / "tools" / "build_action_dataset.py"),
        "--input",
        args.action_trace_file,
        "--output",
        args.dataset_file,
        "--gamma",
        "0.95",
    ]
    train_cmd = [
        sys.executable,
        str(root / "tools" / "train_action_value_model.py"),
        "--dataset",
        args.dataset_file,
        "--out",
        args.model_file,
        "--epochs",
        str(args.train_epochs),
        "--lr",
        str(args.train_lr),
        "--target",
        "return",
    ]
    combat_dataset_cmd = [
        sys.executable,
        str(root / "tools" / "build_combat_episode_dataset.py"),
        "--trace",
        args.action_trace_file,
        "--metrics",
        args.metrics_file,
        "--out",
        args.combat_dataset_file,
    ]
    combat_train_cmd = [
        sys.executable,
        str(root / "tools" / "train_combat_policy_model.py"),
        "--dataset",
        args.combat_dataset_file,
        "--out",
        args.combat_model_file,
    ]
    dual_train_cmd = [
        sys.executable,
        str(root / "tools" / "train_dual_rl_models.py"),
        "--trace",
        args.action_trace_file,
        "--combat-out",
        args.combat_model_file,
        "--noncombat-out",
        args.noncombat_model_file,
        "--epochs",
        str(args.train_epochs),
    ]
    branch_cmd = [
        sys.executable,
        str(root / "tools" / "train_branch_factor_weights.py"),
        "--dataset",
        args.dataset_file,
        "--out",
        args.branch_weight_file,
    ]

    print("[auto-train] retraining model...")
    first = subprocess.run(dataset_cmd, capture_output=True, text=True)
    if first.returncode != 0:
        print("[auto-train] dataset build failed:")
        print(first.stdout)
        print(first.stderr)
        return False

    second = subprocess.run(train_cmd, capture_output=True, text=True)
    if second.returncode != 0:
        print("[auto-train] train failed:")
        print(second.stdout)
        print(second.stderr)
        return False

    second_b = subprocess.run(combat_dataset_cmd, capture_output=True, text=True)
    if second_b.returncode != 0:
        print("[auto-train] combat dataset build failed:")
        print(second_b.stdout)
        print(second_b.stderr)
        return False

    second_c = subprocess.run(combat_train_cmd, capture_output=True, text=True)
    if second_c.returncode != 0:
        print("[auto-train] combat model train failed:")
        print(second_c.stdout)
        print(second_c.stderr)
        return False

    dual = subprocess.run(dual_train_cmd, capture_output=True, text=True)
    if dual.returncode != 0:
        print("[auto-train] dual RL train failed:")
        print(dual.stdout)
        print(dual.stderr)
        return False

    third = subprocess.run(branch_cmd, capture_output=True, text=True)
    if third.returncode != 0:
        print("[auto-train] branch-weight train failed:")
        print(third.stdout)
        print(third.stderr)
        return False

    print(first.stdout.strip())
    print(second.stdout.strip())
    print(second_b.stdout.strip())
    print(second_c.stdout.strip())
    print(dual.stdout.strip())
    print(third.stdout.strip())
    return True


def _ensure_game_running(process_name: str, game_exe: str | None) -> None:
    if _find_process(process_name):
        return

    if not game_exe:
        raise RuntimeError(
            f"Game process '{process_name}' not found. Start game manually or pass --game-exe path for auto launch."
        )

    exe_path = Path(game_exe)
    if not exe_path.exists():
        raise FileNotFoundError(f"game exe not found: {exe_path}")

    subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))
    print(f"[auto-train] launched game: {exe_path}")

    deadline = time.time() + 60.0
    while time.time() < deadline:
        if _find_process(process_name):
            print("[auto-train] game process detected")
            return
        time.sleep(1.0)

    raise TimeoutError(f"Timed out waiting for process '{process_name}'")


def _find_process(process_name: str):
    lowered = process_name.lower()
    for proc in psutil.process_iter(attrs=["name"]):
        name = str(proc.info.get("name") or "").lower()
        if name == lowered:
            return proc
    return None




def _to_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return 0


def _is_terminal_win_state(state, saw_act3_boss: bool) -> bool:
    raw = state.raw_state if isinstance(state.raw_state, dict) else {}
    run = raw.get("run") if isinstance(raw.get("run"), dict) else {}

    for key in ["victory", "won", "is_victory", "is_win"]:
        value = run.get(key)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, (int, float)) and int(value) == 1:
            return True

    act = _to_int(run.get("act"))
    state_type = str(state.state_type or "").strip().lower()
    if saw_act3_boss and (not state.in_combat) and state.player.hp > 0 and act >= 3:
        return state_type in {"menu", "victory", "win", "game_over", "credits"}
    return False


if __name__ == "__main__":
    main()

