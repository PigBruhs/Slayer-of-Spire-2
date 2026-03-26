from __future__ import annotations

import argparse

from sos2_interface.actions.dry_run_executor import DryRunActionExecutor
from sos2_interface.actions.mcp_post_executor import McpPostActionExecutor, McpPostExecutorConfig
from sos2_interface.actions.noop_executor import NoopActionExecutor
from sos2_interface.policy.planner_loop import PlannerLoop, SegmentPlanner
from sos2_interface.readers.hybrid_reader import HybridReader
from sos2_interface.readers.mcp_api_reader import McpApiReader, McpApiReaderConfig
from sos2_interface.readers.memory_reader import MemoryReader, MemoryReaderConfig
from sos2_interface.readers.mod_reader import ModReader
from sos2_interface.readers.mock_reader import MockReader
from sos2_interface.readers.screen_reader import ScreenReader, ScreenReaderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic-segment planner loop")
    parser.add_argument("--reader", choices=["mock", "memory", "screen", "hybrid", "mod", "mcp-api"], default="mock")
    parser.add_argument("--memory-map", default=None)
    parser.add_argument("--screen-map", default=None)
    parser.add_argument("--mcp-config", default=None)
    parser.add_argument("--mcp-host", default="127.0.0.1")
    parser.add_argument("--mcp-port", type=int, default=15526)
    parser.add_argument("--mcp-mode", choices=["singleplayer", "multiplayer"], default="singleplayer")
    parser.add_argument("--interval-ms", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=20, help="Number of planner cycles")
    parser.add_argument("--max-segment-actions", type=int, default=0, help="0 means explore full deterministic turn segment")
    parser.add_argument("--max-branches", type=int, default=3000)
    parser.add_argument("--value-model", default=None, help="Path to trained action-value model JSON")
    parser.add_argument("--value-model-weight", type=float, default=0.35, help="Blend weight for learned action value")
    parser.add_argument("--combat-model", default="runtime/combat_policy_model.json", help="Path to trained combat policy model JSON")
    parser.add_argument("--combat-model-weight", type=float, default=0.8, help="Blend weight for combat policy model")
    parser.add_argument("--noncombat-model", default="runtime/noncombat_policy_model.json", help="Path to trained non-combat policy model JSON")
    parser.add_argument("--noncombat-model-weight", type=float, default=1.0, help="Weight for non-combat policy model")
    parser.add_argument("--disable-noncombat-policy", action="store_true", help="Disable non-combat model control")
    parser.add_argument("--branch-weight-file", default="runtime/branch_factor_weights.json", help="JSON file for branch factor weights")
    parser.add_argument("--executor", choices=["dry-run", "noop", "mcp-post"], default="dry-run")
    parser.add_argument("--enable-live-actions", action="store_true", help="Required safety gate for mcp-post executor")
    parser.add_argument("--capture-action-trace", action="store_true", help="Capture per-action pre/post transitions")
    parser.add_argument("--action-trace-file", default="runtime/planner_action_trace.jsonl")
    parser.add_argument("--trace-raw-state", action="store_true", help="Include full state dumps in per-action trace")
    parser.add_argument("--combat-only", action="store_true", default=True, help="Control combat only; non-combat handled manually")
    parser.add_argument("--no-combat-only", action="store_false", dest="combat_only")
    parser.add_argument("--trace-file", default="runtime/planner_cycles.jsonl")
    return parser.parse_args()


def build_reader(
    reader_mode: str,
    memory_map: str | None,
    screen_map: str | None,
    mcp_config: str | None,
    mcp_host: str,
    mcp_port: int,
    mcp_mode: str,
):
    if reader_mode == "memory":
        cfg = MemoryReaderConfig.from_json(memory_map) if memory_map else MemoryReaderConfig()
        return MemoryReader(cfg)

    if reader_mode == "screen":
        cfg = ScreenReaderConfig.from_json(screen_map) if screen_map else ScreenReaderConfig()
        return ScreenReader(cfg)

    if reader_mode == "hybrid":
        memory_cfg = MemoryReaderConfig.from_json(memory_map) if memory_map else MemoryReaderConfig()
        screen_cfg = ScreenReaderConfig.from_json(screen_map) if screen_map else ScreenReaderConfig()
        return HybridReader(memory_reader=MemoryReader(memory_cfg), screen_reader=ScreenReader(screen_cfg))

    if reader_mode == "mod":
        return ModReader()

    if reader_mode == "mcp-api":
        cfg = McpApiReaderConfig.from_json(mcp_config) if mcp_config else McpApiReaderConfig()
        cfg.host = mcp_host or cfg.host
        cfg.port = mcp_port or cfg.port
        cfg.mode = mcp_mode or cfg.mode
        return McpApiReader(cfg)

    return MockReader()


def main() -> None:
    args = parse_args()

    reader = build_reader(
        args.reader,
        args.memory_map,
        args.screen_map,
        args.mcp_config,
        args.mcp_host,
        args.mcp_port,
        args.mcp_mode,
    )
    if args.executor == "dry-run":
        executor = DryRunActionExecutor()
    elif args.executor == "mcp-post":
        mcp_cfg = McpPostExecutorConfig(host=args.mcp_host, port=args.mcp_port, mode=args.mcp_mode)
        executor = McpPostActionExecutor(config=mcp_cfg, allow_live_actions=args.enable_live_actions)
    else:
        executor = NoopActionExecutor()

    loop = PlannerLoop(
        reader=reader,
        executor=executor,
        planner=SegmentPlanner(
            max_segment_actions=args.max_segment_actions,
            max_branches=args.max_branches,
            value_model_path=args.value_model,
            value_model_weight=args.value_model_weight,
            combat_model_path=args.combat_model,
            combat_model_weight=args.combat_model_weight,
            noncombat_model_path=args.noncombat_model,
            noncombat_model_weight=args.noncombat_model_weight,
            enable_noncombat_policy=not args.disable_noncombat_policy,
            branch_weight_path=args.branch_weight_file,
        ),
        trace_file=args.trace_file,
        action_trace_file=args.action_trace_file,
        capture_action_trace=args.capture_action_trace,
        include_raw_state_in_action_trace=args.trace_raw_state,
        combat_only=bool(args.combat_only),
    )
    loop.run_forever(interval_ms=args.interval_ms, max_iterations=args.iterations)


if __name__ == "__main__":
    main()
