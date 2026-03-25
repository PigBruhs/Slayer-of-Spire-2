from __future__ import annotations

import ctypes
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import psutil

from sos2_interface.contracts.state import GameStateSnapshot, PlayerState
from sos2_interface.readers.base import GameReader


PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400


@dataclass
class MemoryReaderConfig:
    process_name: str = "SlayTheSpire2.exe"
    # single-address fields stay for backward compatibility
    hp_address: int | None = None
    max_hp_address: int | None = None
    energy_address: int | None = None
    # preferred path: multi-candidate lists for auto-fallback
    hp_candidates: list[int] = field(default_factory=list)
    max_hp_candidates: list[int] = field(default_factory=list)
    energy_candidates: list[int] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str | Path) -> "MemoryReaderConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        hp_address = _to_int_or_none(data.get("hp_address"))
        max_hp_address = _to_int_or_none(data.get("max_hp_address"))
        energy_address = _to_int_or_none(data.get("energy_address"))

        hp_candidates = _merge_candidates(hp_address, _to_int_list(data.get("hp_candidates")))
        max_hp_candidates = _merge_candidates(max_hp_address, _to_int_list(data.get("max_hp_candidates")))
        energy_candidates = _merge_candidates(energy_address, _to_int_list(data.get("energy_candidates")))

        return cls(
            process_name=data.get("process_name", "SlayTheSpire2.exe"),
            hp_address=hp_address,
            max_hp_address=max_hp_address,
            energy_address=energy_address,
            hp_candidates=hp_candidates,
            max_hp_candidates=max_hp_candidates,
            energy_candidates=energy_candidates,
        )


class MemoryReader(GameReader):
    """Windows ReadProcessMemory wrapper using user-provided addresses."""

    def __init__(self, config: MemoryReaderConfig) -> None:
        self._config = config
        self._frame = 0
        self._resolved_addresses: dict[str, int] = {}
        self._last_values: dict[str, int] = {}

    def _find_pid(self) -> int | None:
        target_name = (self._config.process_name or "").strip().casefold()
        for proc in psutil.process_iter(["name", "pid"]):
            process_name = str(proc.info.get("name") or "").strip().casefold()
            if process_name == target_name:
                return int(proc.info["pid"])
        return None

    def _read_int32(self, pid: int, address: int) -> int | None:
        process_handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_VM_READ | PROCESS_QUERY_INFORMATION,
            False,
            pid,
        )
        if not process_handle:
            return None
        try:
            value = ctypes.c_int32()
            bytes_read = ctypes.c_size_t()
            ok = ctypes.windll.kernel32.ReadProcessMemory(
                process_handle,
                ctypes.c_void_p(address),
                ctypes.byref(value),
                ctypes.sizeof(value),
                ctypes.byref(bytes_read),
            )
            if not ok or bytes_read.value != ctypes.sizeof(value):
                return None
            return int(value.value)
        finally:
            ctypes.windll.kernel32.CloseHandle(process_handle)

    def read_state(self) -> GameStateSnapshot:
        self._frame += 1
        warnings: list[str] = []

        pid = self._find_pid()
        if pid is None:
            warnings.append("game process not found")
            player = PlayerState(hp=0, max_hp=0, energy=0)
            return GameStateSnapshot(
                source="memory",
                frame_id=self._frame,
                timestamp_ms=int(time.time() * 1000),
                in_combat=False,
                player=player,
                warnings=warnings,
            )

        hp = self._read_with_candidates(
            pid=pid,
            field_name="hp",
            candidates=self._config.hp_candidates,
            validator=lambda v: 0 <= v <= 1500,
            warnings=warnings,
        )
        max_hp = self._read_with_candidates(
            pid=pid,
            field_name="max_hp",
            candidates=self._config.max_hp_candidates,
            validator=lambda v: 1 <= v <= 1500,
            warnings=warnings,
        )
        energy = self._read_with_candidates(
            pid=pid,
            field_name="energy",
            candidates=self._config.energy_candidates,
            validator=lambda v: 0 <= v <= 20,
            warnings=warnings,
        )

        if hp is not None and max_hp is not None and hp > max_hp:
            warnings.append("hp > max_hp detected; max_hp candidate may be incorrect")

        if hp is not None:
            self._last_values["hp"] = hp
        if max_hp is not None:
            self._last_values["max_hp"] = max_hp
        if energy is not None:
            self._last_values["energy"] = energy

        player = PlayerState(
            hp=hp or 0,
            max_hp=max_hp or hp or 0,
            energy=energy or 0,
        )
        return GameStateSnapshot(
            source="memory",
            frame_id=self._frame,
            timestamp_ms=int(time.time() * 1000),
            in_combat=(hp is not None and hp > 0),
            player=player,
            warnings=warnings,
        )

    def status(self) -> dict[str, str | bool]:
        pid = self._find_pid()
        response: dict[str, str | bool] = {
            "ok": pid is not None,
            "mode": "memory",
            "process_found": pid is not None,
        }
        for field_name, address in self._resolved_addresses.items():
            response[f"{field_name}_address"] = hex(address)
        return response

    def _read_with_candidates(
        self,
        pid: int,
        field_name: str,
        candidates: list[int],
        validator: Callable[[int], bool],
        warnings: list[str],
    ) -> int | None:
        if not candidates:
            warnings.append(f"{field_name} candidates are not configured")
            return None

        addresses = _ordered_candidates(candidates, self._resolved_addresses.get(field_name))
        valid_values: list[tuple[int, int]] = []

        for address in addresses:
            value = self._read_int32(pid, address)
            if value is None:
                continue
            if validator(value):
                valid_values.append((address, value))

        if not valid_values:
            address_text = ", ".join(hex(a) for a in addresses)
            warnings.append(f"failed to read {field_name} from candidates: [{address_text}]")
            self._resolved_addresses.pop(field_name, None)
            return None

        selected_address, selected_value = self._pick_best_candidate(field_name, valid_values)
        self._resolved_addresses[field_name] = selected_address
        return selected_value

    def _pick_best_candidate(self, field_name: str, valid_values: list[tuple[int, int]]) -> tuple[int, int]:
        last_value = self._last_values.get(field_name)
        if last_value is None:
            return valid_values[0]
        return min(valid_values, key=lambda item: abs(item[1] - last_value))



def _ordered_candidates(candidates: list[int], resolved: int | None) -> list[int]:
    addresses: list[int] = []
    if resolved is not None and resolved in candidates:
        addresses.append(resolved)
    for address in candidates:
        if address not in addresses:
            addresses.append(address)
    return addresses


def _to_int_or_none(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip().replace("_", "")
        if not text:
            return None
        lower_text = text.lower()
        if lower_text.startswith("0x"):
            return int(lower_text, 16)
        if any(char in "abcdef" for char in lower_text):
            return int(lower_text, 16)
        return int(lower_text, 10)
    raise TypeError(f"unsupported address type: {type(value)!r}")


def _to_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (str, int)):
        single = _to_int_or_none(value)
        return [single] if single is not None else []
    if isinstance(value, list):
        items: list[int] = []
        for raw in value:
            parsed = _to_int_or_none(raw if isinstance(raw, (str, int)) else None)
            if parsed is not None:
                items.append(parsed)
        return items
    raise TypeError(f"unsupported candidate list type: {type(value)!r}")


def _merge_candidates(primary: int | None, candidates: list[int]) -> list[int]:
    merged: list[int] = []
    if primary is not None:
        merged.append(primary)
    for address in candidates:
        if address not in merged:
            merged.append(address)
    return merged


