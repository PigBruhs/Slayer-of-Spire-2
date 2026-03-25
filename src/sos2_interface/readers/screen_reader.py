from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sos2_interface.contracts.state import GameStateSnapshot, PlayerState
from sos2_interface.policy.card_knowledge import resolve_card_id
from sos2_interface.readers.base import GameReader

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None

try:
    from PIL import Image, ImageGrab, ImageOps
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageGrab = None
    ImageOps = None


@dataclass
class ScreenReaderConfig:
    # Optional static image for repeatable debugging; if unset, capture current screen.
    image_path: str | None = None
    tesseract_cmd: str | None = None
    hp_region: tuple[int, int, int, int] = (110, 960, 240, 60)
    energy_region: tuple[int, int, int, int] = (940, 860, 140, 120)
    event_region: tuple[int, int, int, int] = (420, 180, 1080, 220)
    hand_regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    event_keywords: list[str] = field(default_factory=lambda: ["event", "leave", "accept", "decline"])

    @classmethod
    def from_json(cls, path: str | Path) -> "ScreenReaderConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            image_path=data.get("image_path"),
            tesseract_cmd=data.get("tesseract_cmd"),
            hp_region=_to_region(data.get("hp_region"), (110, 960, 240, 60)),
            energy_region=_to_region(data.get("energy_region"), (940, 860, 140, 120)),
            event_region=_to_region(data.get("event_region"), (420, 180, 1080, 220)),
            hand_regions=_to_regions(data.get("hand_regions")),
            event_keywords=_to_str_list(data.get("event_keywords"), ["event", "leave", "accept", "decline"]),
        )


class ScreenReader(GameReader):
    """MVP screen reader using OCR on fixed ROIs.

    It focuses on durable fields (hp/max_hp/energy/event hint) and avoids full-scene parsing.
    """

    def __init__(self, config: ScreenReaderConfig) -> None:
        self._config = config
        self._frame = 0

        if pytesseract and self._config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self._config.tesseract_cmd

    def read_state(self) -> GameStateSnapshot:
        self._frame += 1
        warnings: list[str] = []

        frame = self._capture_frame(warnings)
        if frame is None:
            player = PlayerState(hp=0, max_hp=0, energy=0)
            return GameStateSnapshot(
                source="screen",
                frame_id=self._frame,
                timestamp_ms=int(time.time() * 1000),
                in_combat=False,
                in_event=False,
                player=player,
                warnings=warnings,
            )

        hp_text = self._ocr_region(frame, self._config.hp_region, warnings, psm=7)
        energy_text = self._ocr_region(frame, self._config.energy_region, warnings, psm=7)
        event_text = self._ocr_region(frame, self._config.event_region, warnings, psm=6)

        hp, max_hp = _parse_hp_pair(hp_text)
        energy = _parse_first_int(energy_text)

        if hp is None:
            warnings.append("screen hp parse failed")
        if max_hp is None:
            warnings.append("screen max_hp parse failed")
        if energy is None:
            warnings.append("screen energy parse failed")

        hand: list[str] = []
        for index, region in enumerate(self._config.hand_regions):
            card_text = self._ocr_region(frame, region, warnings, psm=7)
            card_id = resolve_card_id(card_text)
            if card_id:
                hand.append(card_id)
            elif _looks_like_card_text(card_text):
                warnings.append(f"hand card mapping failed at slot {index}: '{card_text.strip()}'")

        lowered_event_text = event_text.lower()
        in_event = any(keyword.lower() in lowered_event_text for keyword in self._config.event_keywords)
        in_combat = (hp is not None and hp > 0 and not in_event) or (energy is not None and energy >= 0 and not in_event)

        player = PlayerState(
            hp=hp or 0,
            max_hp=max_hp or hp or 0,
            energy=energy or 0,
            hand=hand,
        )
        return GameStateSnapshot(
            source="screen",
            frame_id=self._frame,
            timestamp_ms=int(time.time() * 1000),
            in_combat=in_combat,
            in_event=in_event,
            player=player,
            warnings=warnings,
        )

    def status(self) -> dict[str, str | bool]:
        return {
            "ok": True,
            "mode": "screen",
            "has_cv2": cv2 is not None,
            "has_numpy": np is not None,
            "has_pytesseract": pytesseract is not None,
            "has_pillow": Image is not None,
            "hand_regions_configured": bool(self._config.hand_regions),
        }

    def _capture_frame(self, warnings: list[str]) -> Any | None:
        if Image is None:
            warnings.append("Pillow is not installed; screen capture is unavailable")
            return None

        if self._config.image_path:
            image_path = Path(self._config.image_path)
            if not image_path.exists():
                warnings.append(f"screen image not found: {image_path}")
                return None
            return Image.open(image_path).convert("RGB")

        if ImageGrab is None:
            warnings.append("ImageGrab is unavailable on this environment")
            return None

        try:
            return ImageGrab.grab(all_screens=True).convert("RGB")
        except Exception as exc:  # pragma: no cover - depends on desktop session
            warnings.append(f"screen capture failed: {exc}")
            return None

    def _ocr_region(self, frame: Any, region: tuple[int, int, int, int], warnings: list[str], psm: int = 7) -> str:
        if pytesseract is None:
            warnings.append("pytesseract is not installed")
            return ""
        if np is None or cv2 is None:
            warnings.append("numpy/opencv is required for OCR preprocessing")
            return ""
        if ImageOps is None:
            warnings.append("Pillow ImageOps is unavailable")
            return ""

        x, y, w, h = region
        try:
            roi = frame.crop((x, y, x + w, y + h))
            gray = ImageOps.grayscale(roi)
            image_array = np.array(gray)
            processed = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            return pytesseract.image_to_string(processed, config=f"--psm {psm}")
        except Exception as exc:  # pragma: no cover - depends on OCR/runtime setup
            warnings.append(f"ocr failed for region {region}: {exc}")
            return ""


def _to_region(raw: object, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if raw is None:
        return default
    if not isinstance(raw, list) or len(raw) != 4:
        return default
    try:
        x, y, w, h = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
        return (x, y, w, h)
    except (TypeError, ValueError):
        return default


def _to_str_list(raw: object, default: list[str]) -> list[str]:
    if raw is None:
        return default
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return default


def _to_regions(raw: object) -> list[tuple[int, int, int, int]]:
    if not isinstance(raw, list):
        return []

    regions: list[tuple[int, int, int, int]] = []
    for item in raw:
        region = _to_region(item, default=(0, 0, 0, 0))
        if region[2] > 0 and region[3] > 0:
            regions.append(region)
    return regions


def _parse_hp_pair(text: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2))

    values = [int(token) for token in re.findall(r"\d+", text)]
    if len(values) >= 2:
        return values[0], values[1]
    if len(values) == 1:
        return values[0], None
    return None, None


def _parse_first_int(text: str) -> int | None:
    values = re.findall(r"\d+", text)
    if not values:
        return None
    return int(values[0])


def _looks_like_card_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", stripped))


