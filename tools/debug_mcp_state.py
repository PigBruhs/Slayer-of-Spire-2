from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sos2_interface.readers.mcp_api_reader import McpApiReader, McpApiReaderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug STS2 MCP state polling")
    parser.add_argument("--mcp-config", default=None, help="JSON config for MCP endpoint")
    parser.add_argument("--mcp-host", default="127.0.0.1")
    parser.add_argument("--mcp-port", type=int, default=15526)
    parser.add_argument("--mcp-mode", choices=["singleplayer", "multiplayer"], default="singleplayer")
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--count", type=int, default=0, help="0 means run forever")
    parser.add_argument("--out", default="runtime/mcp_debug.jsonl", help="Optional JSONL output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = McpApiReaderConfig.from_json(args.mcp_config) if args.mcp_config else McpApiReaderConfig()
    cfg.host = args.mcp_host or cfg.host
    cfg.port = args.mcp_port or cfg.port
    cfg.mode = args.mcp_mode or cfg.mode

    reader = McpApiReader(cfg)

    out_path = Path(args.out).expanduser() if args.out else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[debug] polling http://{cfg.host}:{cfg.port} mode={cfg.mode} interval={args.interval_ms}ms")
    print("[debug] press Ctrl+C to stop")

    seen = 0
    last_state_type = None
    try:
        while True:
            snapshot = reader.read_state()
            seen += 1

            state_type = snapshot.state_type or "unknown"
            hp_text = f"{snapshot.player.hp}/{snapshot.player.max_hp}"
            compact = {
                "frame": snapshot.frame_id,
                "state_type": state_type,
                "game_mode": snapshot.game_mode,
                "net_type": snapshot.net_type,
                "in_combat": snapshot.in_combat,
                "in_event": snapshot.in_event,
                "turn": snapshot.turn,
                "hp": hp_text,
                "block": snapshot.player.block,
                "energy": snapshot.player.energy,
                "hand": len(snapshot.player.hand),
                "enemies": len(snapshot.enemies),
                "warnings": snapshot.warnings,
            }

            transition = ""
            if state_type != last_state_type:
                transition = f"  <transition {last_state_type or '-'} -> {state_type}>"
                last_state_type = state_type

            print(json.dumps(compact, ensure_ascii=True) + transition)

            if out_path is not None:
                record = snapshot.model_dump()
                record["captured_at_ms"] = int(time.time() * 1000)
                with out_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")

            if args.count > 0 and seen >= args.count:
                break

            time.sleep(args.interval_ms / 1000)
    except KeyboardInterrupt:
        print("[debug] stopped by user")


if __name__ == "__main__":
    main()

