from __future__ import annotations

from abc import ABC, abstractmethod

from sos2_interface.contracts.state import GameStateSnapshot


class GameReader(ABC):
    @abstractmethod
    def read_state(self) -> GameStateSnapshot:
        raise NotImplementedError

    @abstractmethod
    def status(self) -> dict[str, str | bool]:
        raise NotImplementedError

