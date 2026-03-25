from __future__ import annotations

from sos2_interface.contracts.state import GameStateSnapshot
from sos2_interface.readers.base import GameReader


class HybridReader(GameReader):
    """Combine memory and screen readers.

    Memory remains the primary source for numeric stability.
    Screen OCR fills gaps when memory values are missing or invalid.
    """

    def __init__(self, memory_reader: GameReader, screen_reader: GameReader) -> None:
        self._memory_reader = memory_reader
        self._screen_reader = screen_reader

    def read_state(self) -> GameStateSnapshot:
        memory_state = self._memory_reader.read_state()
        screen_state = self._screen_reader.read_state()

        merged = memory_state.model_copy(deep=True)
        merged.source = "hybrid"

        if merged.player.hp <= 0 and screen_state.player.hp > 0:
            merged.player.hp = screen_state.player.hp
        if merged.player.max_hp <= 0 and screen_state.player.max_hp > 0:
            merged.player.max_hp = screen_state.player.max_hp
        if merged.player.energy <= 0 and screen_state.player.energy >= 0:
            merged.player.energy = screen_state.player.energy

        merged.in_event = memory_state.in_event or screen_state.in_event
        merged.in_combat = memory_state.in_combat or screen_state.in_combat

        merged.warnings.extend(screen_state.warnings)
        return merged

    def status(self) -> dict[str, str | bool]:
        response: dict[str, str | bool] = {
            "ok": True,
            "mode": "hybrid",
        }
        for key, value in self._memory_reader.status().items():
            response[f"memory_{key}"] = value
        for key, value in self._screen_reader.status().items():
            response[f"screen_{key}"] = value
        return response

