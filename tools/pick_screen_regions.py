from __future__ import annotations

import argparse
import json
from pathlib import Path
from tkinter import Canvas, Tk, simpledialog

from PIL import Image, ImageGrab, ImageTk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive region picker for screen_map config")
    parser.add_argument("--screen-map", default="config/screen_map.example.json")
    parser.add_argument("--image", default=None, help="Optional image path for offline calibration")
    parser.add_argument(
        "--elements",
        nargs="*",
        default=[
            "hp_region",
            "energy_region",
            "event_region",
            "hand_1",
            "hand_2",
            "hand_3",
            "hand_4",
            "hand_5",
            "hand_6",
            "hand_7",
            "hand_8",
            "hand_9",
            "hand_10",
        ],
    )
    return parser.parse_args()


def load_image(image_path: str | None) -> Image.Image:
    if image_path:
        return Image.open(image_path).convert("RGB")
    return ImageGrab.grab(all_screens=True).convert("RGB")


def to_region(x1: int, y1: int, x2: int, y2: int) -> list[int]:
    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return [left, top, width, height]


def main() -> None:
    args = parse_args()
    map_path = Path(args.screen_map)
    if map_path.exists():
        config = json.loads(map_path.read_text(encoding="utf-8"))
    else:
        config = {}

    image = load_image(args.image)
    root = Tk()
    root.title("SOS2 Region Picker")

    tk_image = ImageTk.PhotoImage(image)
    canvas = Canvas(root, width=image.width, height=image.height, cursor="cross")
    canvas.pack()
    canvas.create_image(0, 0, image=tk_image, anchor="nw")

    selected: dict[str, list[int]] = {}
    element_index = {"value": 0}
    start_xy = {"x": 0, "y": 0}
    current_rect: dict[str, int | None] = {"id": None}

    def current_element() -> str:
        if element_index["value"] >= len(args.elements):
            return ""
        return args.elements[element_index["value"]]

    def prompt_next() -> None:
        element = current_element()
        if not element:
            root.title("SOS2 Region Picker - done, press S to save")
            return
        root.title(f"SOS2 Region Picker - draw: {element} (N skip, U undo, S save)")

    def on_press(event) -> None:  # type: ignore[no-untyped-def]
        start_xy["x"] = event.x
        start_xy["y"] = event.y
        rect_id = current_rect["id"]
        if rect_id is not None:
            canvas.delete(rect_id)
        current_rect["id"] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=2)

    def on_drag(event) -> None:  # type: ignore[no-untyped-def]
        rect_id = current_rect["id"]
        if rect_id is not None:
            canvas.coords(rect_id, start_xy["x"], start_xy["y"], event.x, event.y)

    def on_release(event) -> None:  # type: ignore[no-untyped-def]
        element = current_element()
        if not element:
            return
        region = to_region(start_xy["x"], start_xy["y"], event.x, event.y)
        if region[2] < 8 or region[3] < 8:
            return

        custom_name = simpledialog.askstring("Element", f"Confirm key for current region ({element})", initialvalue=element)
        key = (custom_name or element).strip()
        if not key:
            key = element
        selected[key] = region
        element_index["value"] += 1
        prompt_next()

    def on_skip(event) -> None:  # type: ignore[no-untyped-def]
        if element_index["value"] < len(args.elements):
            element_index["value"] += 1
            prompt_next()

    def on_undo(event) -> None:  # type: ignore[no-untyped-def]
        if not selected:
            return
        last_key = list(selected.keys())[-1]
        del selected[last_key]
        element_index["value"] = max(0, element_index["value"] - 1)
        prompt_next()

    def on_save(event) -> None:  # type: ignore[no-untyped-def]
        hand_regions: list[list[int]] = []
        for key in sorted(selected.keys()):
            if key.startswith("hand_"):
                hand_regions.append(selected[key])
            else:
                config[key] = selected[key]

        if hand_regions:
            config["hand_regions"] = hand_regions

        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(json.dumps(config, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Saved: {map_path}")
        print(f"regions: {len(selected)}")
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("n", on_skip)
    root.bind("u", on_undo)
    root.bind("s", on_save)

    prompt_next()
    root.mainloop()


if __name__ == "__main__":
    main()

