from __future__ import annotations

import itertools
import time

from sos2_interface.contracts.state import EnemyIntent, EnemyState, EventState, GameStateSnapshot, PlayerState
from sos2_interface.readers.base import GameReader


class MockReader(GameReader):
    """Deterministic fake reader so the full pipeline can be tested before memory offsets are known."""

    def __init__(self) -> None:
        self._counter = itertools.count(1)

    def read_state(self) -> GameStateSnapshot:
        frame_id = next(self._counter)
        in_event = frame_id % 7 == 0
        in_combat = not in_event

        if in_event:
            player = PlayerState(hp=63, max_hp=80, block=0, energy=0, hand=[])
            return GameStateSnapshot(
                source="mock",
                frame_id=frame_id,
                timestamp_ms=int(time.time() * 1000),
                in_event=True,
                in_combat=False,
                player=player,
                event=EventState(
                    event_id="golden_shrine",
                    title="Golden Shrine",
                    options=["Pray", "Desecrate", "Leave"],
                ),
            )

        enemy_intent = EnemyIntent(enemy_id="slaver_red", intent_type="attack", amount=12, hits=1)
        enemy = EnemyState(
            enemy_id="slaver_red",
            hp=max(5, 50 - frame_id),
            max_hp=50,
            block=0,
            intents=[enemy_intent],
        )
        player = PlayerState(
            hp=63,
            max_hp=80,
            block=6 if frame_id % 2 == 0 else 0,
            energy=3 - (frame_id % 3),
            hand=["strike", "defend", "bash", "anger", "shrug_it_off"],
            draw_pile_count=12,
            discard_pile_count=4,
        )
        return GameStateSnapshot(
            source="mock",
            frame_id=frame_id,
            timestamp_ms=int(time.time() * 1000),
            in_event=False,
            in_combat=True,
            turn=(frame_id % 5) + 1,
            player=player,
            enemies=[enemy],
        )

    def status(self) -> dict[str, str | bool]:
        return {"ok": True, "mode": "mock"}

