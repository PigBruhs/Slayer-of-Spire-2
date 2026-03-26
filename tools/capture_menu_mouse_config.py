from __future__ import annotations

import argparse
import ctypes
import copy
import json
import time
from pathlib import Path


CAPTURE_SEQUENCE_PLAN: list[tuple[str, int]] = [
    ("start_singleplayer_defect", 4),
    ("softlock_troubleshoot", 1),
    ("return_to_main_menu", 1),
]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture mouse coordinates for auto menu sequences")
    parser.add_argument("--input", default="", help="Optional fallback template JSON (used only when output does not exist)")
    parser.add_argument("--output", default="config/auto_menu_mouse.local.json", help="Output JSON path")
    parser.add_argument("--countdown", type=float, default=3.0, help="Seconds before capturing cursor")
    parser.add_argument("--normalized", action="store_true", help="Save as rx/ry ratios instead of x/y pixels")
    parser.add_argument("--game-relative", action="store_true", default=True, help="Save coordinates relative to game window (gx/gy or grx/gry)")
    parser.add_argument("--screen-absolute", action="store_false", dest="game_relative", help="Save absolute screen coordinates (x/y)")
    parser.add_argument("--process-name", default="SlayTheSpire2.exe", help="Game process name used for window-relative capture")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload, source_path = _load_base_payload(args.output, args.input)
    if not isinstance(payload, dict):
        raise RuntimeError("invalid config: root must be object")
    print(f"[capture] loaded base: {source_path}")

    original_payload = copy.deepcopy(payload)

    sequences = payload.get("sequences")
    if not isinstance(sequences, dict):
        sequences = {}
        payload["sequences"] = sequences

    payload["sequences"] = _normalize_capture_sequences(sequences)
    sequences = payload["sequences"]
    selected = [name for name, _count in CAPTURE_SEQUENCE_PLAN]
    modified_steps: set[tuple[str, int]] = set()

    print("[capture] selected sequences:", ", ".join(selected))
    print("[capture] fixed total captures: 6 (start x4, troubleshoot x1, return x1)")
    print("[capture] move cursor to target area; capture runs after countdown")
    print("[capture] commands: Enter=capture, s=skip, q=save-and-quit")

    width = ctypes.windll.user32.GetSystemMetrics(0)
    height = ctypes.windll.user32.GetSystemMetrics(1)
    game_rect = get_game_window_rect(args.process_name) if args.game_relative else None
    if args.game_relative and game_rect is None:
        raise RuntimeError(f"game window not found for process: {args.process_name}")

    for seq_name, step_count in CAPTURE_SEQUENCE_PLAN:
        steps = sequences.get(seq_name)
        if not isinstance(steps, list):
            continue

        print(f"\n=== Sequence: {seq_name} ({step_count} steps) ===")
        for idx in range(step_count):
            step = steps[idx]
            if not isinstance(step, dict):
                continue

            if bool(step.get("sleep_only", False)):
                print(f"step {idx + 1}: sleep_only, skipped")
                continue

            old_xy = _step_xy(step)
            print(f"step {idx + 1}: old={old_xy} clicks={step.get('clicks', 1)} delay_ms={step.get('delay_ms', 0)}")
            cmd = input("capture this step? [Enter/s/q]: ").strip().lower()
            if cmd == "q":
                _save_modified_only(args.output, original_payload, payload, modified_steps)
                print("[capture] partial save complete")
                return
            if cmd == "s":
                continue

            _countdown(max(0.2, float(args.countdown)))
            x, y = get_cursor_pos()
            if args.game_relative and game_rect is not None:
                left, top, gwidth, gheight = game_rect
                rel_x = int(x - left)
                rel_y = int(y - top)
                step.pop("x", None)
                step.pop("y", None)
                step.pop("rx", None)
                step.pop("ry", None)
                if args.normalized:
                    step.pop("grx", None)
                    step.pop("gry", None)
                    step["grx"] = round(float(rel_x) / float(max(1, gwidth)), 6)
                    step["gry"] = round(float(rel_y) / float(max(1, gheight)), 6)
                    print(f"  -> captured grx={step['grx']} gry={step['gry']}")
                else:
                    step.pop("gx", None)
                    step.pop("gy", None)
                    step["gx"] = rel_x
                    step["gy"] = rel_y
                    print(f"  -> captured gx={rel_x} gy={rel_y}")
            elif args.normalized:
                step.pop("x", None)
                step.pop("y", None)
                step["rx"] = round(float(x) / float(max(1, width)), 6)
                step["ry"] = round(float(y) / float(max(1, height)), 6)
                print(f"  -> captured rx={step['rx']} ry={step['ry']}")
            else:
                step.pop("rx", None)
                step.pop("ry", None)
                step["x"] = int(x)
                step["y"] = int(y)
                print(f"  -> captured x={x} y={y}")
            modified_steps.add((seq_name, idx))

    _save_modified_only(args.output, original_payload, payload, modified_steps)
    print(f"[capture] saved: {Path(args.output).resolve()}")


def _normalize_capture_sequences(sequences: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    normalized: dict[str, list[dict[str, object]]] = {}
    for name, count in CAPTURE_SEQUENCE_PLAN:
        existing = sequences.get(name)
        steps: list[dict[str, object]] = []
        if isinstance(existing, list):
            for step in existing:
                if isinstance(step, dict):
                    steps.append(dict(step))

        while len(steps) < count:
            steps.append({"clicks": 1, "delay_ms": 600, "button": "left"})
        normalized[name] = steps[:count]
    return normalized


def _load_base_payload(output_path: str, input_path: str) -> tuple[dict[str, object], str]:
    out = Path(output_path)
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8")), str(out.resolve())

    if input_path.strip():
        inp = Path(input_path)
        if not inp.exists():
            raise FileNotFoundError(f"input config not found: {inp}")
        return json.loads(inp.read_text(encoding="utf-8")), str(inp.resolve())

    return {"sequences": {}}, "generated-empty"


def _step_xy(step: dict[str, object]) -> str:
    if isinstance(step.get("gx"), (int, float)) and isinstance(step.get("gy"), (int, float)):
        return f"(gx={int(step['gx'])}, gy={int(step['gy'])})"
    if isinstance(step.get("grx"), (int, float)) and isinstance(step.get("gry"), (int, float)):
        return f"(grx={step['grx']}, gry={step['gry']})"
    if isinstance(step.get("x"), (int, float)) and isinstance(step.get("y"), (int, float)):
        return f"({int(step['x'])}, {int(step['y'])})"
    if isinstance(step.get("rx"), (int, float)) and isinstance(step.get("ry"), (int, float)):
        return f"(rx={step['rx']}, ry={step['ry']})"
    return "(unset)"


def _countdown(seconds: float) -> None:
    total = max(1, int(round(seconds * 10)))
    for tick in range(total, 0, -1):
        remain = tick / 10.0
        print(f"  capture in {remain:.1f}s...", end="\r", flush=True)
        time.sleep(0.1)
    print(" " * 32, end="\r", flush=True)


def get_cursor_pos() -> tuple[int, int]:
    point = POINT()
    ok = ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    if not ok:
        raise RuntimeError("GetCursorPos failed")
    return int(point.x), int(point.y)


def get_game_window_rect(process_name: str) -> tuple[int, int, int, int] | None:
    pid = _find_pid_by_name(process_name)
    if pid is None:
        return None

    user32 = ctypes.windll.user32
    found: list[int] = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def _enum_cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        pid_out = ctypes.c_ulong(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_out))
        if int(pid_out.value) == int(pid):
            found.append(int(hwnd))
            return False
        return True

    user32.EnumWindows(EnumWindowsProc(_enum_cb), 0)
    if not found:
        return None

    rect = RECT()
    ok = user32.GetClientRect(found[0], ctypes.byref(rect))
    if not ok:
        return None

    origin = POINT()
    origin.x = 0
    origin.y = 0
    ok = user32.ClientToScreen(found[0], ctypes.byref(origin))
    if not ok:
        return None
    return int(origin.x), int(origin.y), int(rect.right - rect.left), int(rect.bottom - rect.top)


def _find_pid_by_name(process_name: str) -> int | None:
    kernel32 = ctypes.windll.kernel32
    TH32CS_SNAPPROCESS = 0x00000002
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong),
            ("cntUsage", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong),
            ("th32DefaultHeapID", ctypes.c_void_p),
            ("th32ModuleID", ctypes.c_ulong),
            ("cntThreads", ctypes.c_ulong),
            ("th32ParentProcessID", ctypes.c_ulong),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_ulong),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == INVALID_HANDLE_VALUE:
        return None

    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        target = str(process_name or "").strip().lower()
        while ok:
            name = str(entry.szExeFile or "").strip().lower()
            if name == target:
                return int(entry.th32ProcessID)
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return None


def _save_modified_only(
    output_path: str,
    original_payload: dict[str, object],
    edited_payload: dict[str, object],
    modified_steps: set[tuple[str, int]],
) -> None:
    if not modified_steps:
        print("[capture] no modified steps; skip writing")
        return

    out_payload = copy.deepcopy(original_payload)
    if not isinstance(out_payload.get("sequences"), dict):
        out_payload["sequences"] = {}
    out_sequences = out_payload["sequences"] if isinstance(out_payload.get("sequences"), dict) else {}
    edited_sequences = edited_payload.get("sequences") if isinstance(edited_payload.get("sequences"), dict) else {}

    for seq_name, idx in sorted(modified_steps):
        src_list = edited_sequences.get(seq_name) if isinstance(edited_sequences.get(seq_name), list) else []
        if idx < 0 or idx >= len(src_list):
            continue
        src_step = src_list[idx]
        if not isinstance(src_step, dict):
            continue

        dst_list = out_sequences.get(seq_name)
        if not isinstance(dst_list, list):
            dst_list = []
            out_sequences[seq_name] = dst_list
        while len(dst_list) <= idx:
            dst_list.append({"clicks": 1, "delay_ms": 600, "button": "left"})
        dst_list[idx] = copy.deepcopy(src_step)

    _save(output_path, out_payload)


def _save(output_path: str, payload: dict[str, object]) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[capture] saved file: {out.resolve()}")


if __name__ == "__main__":
    main()

