from __future__ import annotations

import argparse
import json
import time

from sos2_interface.readers.hybrid_reader import HybridReader
from sos2_interface.readers.mcp_api_reader import McpApiReader, McpApiReaderConfig
from sos2_interface.readers.memory_reader import MemoryReader, MemoryReaderConfig
from sos2_interface.readers.mock_reader import MockReader
from sos2_interface.readers.screen_reader import ScreenReader, ScreenReaderConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll state and print it as JSON")
    parser.add_argument("--reader", choices=["mock", "memory", "screen", "hybrid", "mcp-api"], default="mock")
    parser.add_argument("--memory-map", default=None)
    parser.add_argument("--screen-map", default=None)
    parser.add_argument("--mcp-config", default=None)
    parser.add_argument("--mcp-host", default="127.0.0.1")
    parser.add_argument("--mcp-port", type=int, default=15526)
    parser.add_argument("--mcp-mode", choices=["singleplayer", "multiplayer"], default="singleplayer")
    parser.add_argument("--interval-ms", type=int, default=500)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    if args.reader == "memory":
        cfg = MemoryReaderConfig.from_json(args.memory_map) if args.memory_map else MemoryReaderConfig()
        reader = MemoryReader(cfg)
    elif args.reader == "screen":
        cfg = ScreenReaderConfig.from_json(args.screen_map) if args.screen_map else ScreenReaderConfig()
        reader = ScreenReader(cfg)
    elif args.reader == "hybrid":
        memory_cfg = MemoryReaderConfig.from_json(args.memory_map) if args.memory_map else MemoryReaderConfig()
        screen_cfg = ScreenReaderConfig.from_json(args.screen_map) if args.screen_map else ScreenReaderConfig()
        reader = HybridReader(memory_reader=MemoryReader(memory_cfg), screen_reader=ScreenReader(screen_cfg))
    elif args.reader == "mcp-api":
        cfg = McpApiReaderConfig.from_json(args.mcp_config) if args.mcp_config else McpApiReaderConfig()
        cfg.host = args.mcp_host or cfg.host
        cfg.port = args.mcp_port or cfg.port
        cfg.mode = args.mcp_mode or cfg.mode
        reader = McpApiReader(cfg)
    else:
        reader = MockReader()

    for _ in range(args.count):
        state = reader.read_state()
        print(json.dumps(state.model_dump(), ensure_ascii=True))
        time.sleep(args.interval_ms / 1000)


if __name__ == "__main__":
    main()
