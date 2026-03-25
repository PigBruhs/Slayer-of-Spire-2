from __future__ import annotations

import argparse

import uvicorn

from sos2_interface.actions.noop_executor import NoopActionExecutor
from sos2_interface.api import create_app
from sos2_interface.core.runtime import Runtime
from sos2_interface.readers.hybrid_reader import HybridReader
from sos2_interface.readers.mcp_api_reader import McpApiReader, McpApiReaderConfig
from sos2_interface.readers.memory_reader import MemoryReader, MemoryReaderConfig
from sos2_interface.readers.mod_reader import ModReader
from sos2_interface.readers.mock_reader import MockReader
from sos2_interface.readers.screen_reader import ScreenReader, ScreenReaderConfig


def build_runtime(
    reader_mode: str,
    interval_ms: int,
    memory_map: str | None,
    screen_map: str | None,
    mcp_config: str | None,
    mcp_host: str,
    mcp_port: int,
    mcp_mode: str,
) -> Runtime:
    if reader_mode == "memory":
        config = MemoryReaderConfig.from_json(memory_map) if memory_map else MemoryReaderConfig()
        reader = MemoryReader(config)
    elif reader_mode == "screen":
        config = ScreenReaderConfig.from_json(screen_map) if screen_map else ScreenReaderConfig()
        reader = ScreenReader(config)
    elif reader_mode == "hybrid":
        memory_config = MemoryReaderConfig.from_json(memory_map) if memory_map else MemoryReaderConfig()
        screen_config = ScreenReaderConfig.from_json(screen_map) if screen_map else ScreenReaderConfig()
        reader = HybridReader(memory_reader=MemoryReader(memory_config), screen_reader=ScreenReader(screen_config))
    elif reader_mode == "mod":
        reader = ModReader()
    elif reader_mode == "mcp-api":
        cfg = McpApiReaderConfig.from_json(mcp_config) if mcp_config else McpApiReaderConfig()
        cfg.host = mcp_host or cfg.host
        cfg.port = mcp_port or cfg.port
        cfg.mode = mcp_mode or cfg.mode
        reader = McpApiReader(cfg)
    else:
        reader = MockReader()

    executor = NoopActionExecutor()
    runtime = Runtime(reader=reader, executor=executor, interval_ms=interval_ms)
    runtime.start()
    return runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slay the Spire 2 interface layer")
    parser.add_argument("--reader", choices=["mock", "memory", "screen", "hybrid", "mod", "mcp-api"], default="mock")
    parser.add_argument("--memory-map", default=None, help="JSON file with process name and memory addresses")
    parser.add_argument("--screen-map", default=None, help="JSON file with OCR ROI and screen reader settings")
    parser.add_argument("--mcp-config", default=None, help="JSON file with MCP REST endpoint settings")
    parser.add_argument("--mcp-host", default="127.0.0.1", help="STS2MCP host")
    parser.add_argument("--mcp-port", type=int, default=15526, help="STS2MCP port")
    parser.add_argument("--mcp-mode", choices=["singleplayer", "multiplayer"], default="singleplayer")
    parser.add_argument("--interval-ms", type=int, default=200)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = build_runtime(
        args.reader,
        args.interval_ms,
        args.memory_map,
        args.screen_map,
        args.mcp_config,
        args.mcp_host,
        args.mcp_port,
        args.mcp_mode,
    )
    app = create_app(runtime)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        runtime.stop()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
