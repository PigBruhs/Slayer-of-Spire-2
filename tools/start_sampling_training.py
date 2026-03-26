from __future__ import annotations

import argparse
import ctypes
import json
import random
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import psutil

from sos2_interface.actions.mcp_post_executor import McpPostActionExecutor, McpPostExecutorConfig
from sos2_interface.contracts.action import ActionCommand
from sos2_interface.contracts.state import GameStateSnapshot
from sos2_interface.policy.combat_policy_model import CombatPolicyModel, load_combat_policy_model
from sos2_interface.policy.noncombat_policy_model import NonCombatPolicyModel, load_noncombat_policy_model
from sos2_interface.policy.segment_simulator import DeterministicSegmentSimulator
from sos2_interface.policy.trace_utils import compact_state, summarize_transition
from sos2_interface.readers.mcp_api_reader import McpApiReader, McpApiReaderConfig


DEFEAT_EXIT_X = 853
DEFEAT_EXIT_Y = 744
DEFEAT_EXIT_CLICKS = 2
DEFEAT_EXIT_CLICK_INTERVAL_MS = 1000
DEFEAT_EXIT_POST_WAIT_S = 3.0
RETURN_SEQUENCE_REPEAT_INTERVAL_S = 2.5


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


@dataclass
class EpisodeStats:
    episode: int
    start_ms: int
    end_ms: int
    max_act: int
    max_floor: int
    hp_start: int
    hp_end: int
    hp_loss: int
    died: bool


@dataclass
class CardSelectMemory:
    key: str = ""
    picked: set[int] | None = None
    required_count: int = 1

    def reset(self, key: str, required_count: int) -> None:
        self.key = key
        self.picked = set()
        self.required_count = max(1, int(required_count))


@dataclass
class RewardMemory:
    key: str = ""
    failed_claim_indices: set[int] | None = None

    def reset(self, key: str) -> None:
        self.key = key
        self.failed_claim_indices = set()


@dataclass
class ShopMemory:
    key: str = ""
    failed_purchase_indices: set[int] | None = None

    def reset(self, key: str) -> None:
        self.key = key
        self.failed_purchase_indices = set()


class MouseMenuController:
    def __init__(self, config_path: str | None) -> None:
        self._config_path = config_path
        self._sequences: dict[str, list[dict[str, object]]] = {}
        self._last_run_ms: dict[str, int] = {}
        self._load()

    def has_sequence(self, name: str) -> bool:
        return bool(self._sequences.get(name))

    def run_sequence(
        self,
        name: str,
        min_interval_ms: int = 1500,
        window_controller: "GameWindowController | None" = None,
    ) -> bool:
        steps = self._sequences.get(name)
        if not steps:
            return False

        now_ms = int(time.time() * 1000)
        if now_ms - self._last_run_ms.get(name, 0) < max(0, int(min_interval_ms)):
            return False

        print(f"[menu] run sequence: {name} ({len(steps)} steps)")
        for step in steps:
            if not isinstance(step, dict):
                continue
            delay_ms = int(step.get("delay_ms") or 0)
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            if bool(step.get("sleep_only", False)):
                continue

            x, y = self._resolve_xy(step, window_controller)
            clicks = max(1, int(step.get("clicks") or 1))
            button = str(step.get("button") or "left").lower()
            interval_ms = max(0, int(step.get("interval_ms") or 120))
            if x is None or y is None:
                continue

            self._mouse_click(x, y, button=button, clicks=clicks, interval_ms=interval_ms)

        self._last_run_ms[name] = int(time.time() * 1000)
        return True

    def _load(self) -> None:
        if not self._config_path:
            return
        path = Path(self._config_path)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        seq = payload.get("sequences")
        if not isinstance(seq, dict):
            return

        for name, steps in seq.items():
            if not isinstance(name, str) or not isinstance(steps, list):
                continue
            normalized: list[dict[str, object]] = []
            for step in steps:
                if isinstance(step, dict):
                    normalized.append(step)
            if normalized:
                self._sequences[name] = normalized

    @staticmethod
    def _resolve_xy(step: dict[str, object], window_controller: "GameWindowController | None" = None) -> tuple[int | None, int | None]:
        x_raw = step.get("x")
        y_raw = step.get("y")
        if isinstance(x_raw, (int, float)) and isinstance(y_raw, (int, float)):
            return int(x_raw), int(y_raw)

        gx = step.get("gx")
        gy = step.get("gy")
        if isinstance(gx, (int, float)) and isinstance(gy, (int, float)) and window_controller is not None:
            rect = window_controller.get_window_rect()
            if rect is not None:
                left, top, _width, _height = rect
                return int(left + float(gx)), int(top + float(gy))

        grx = step.get("grx")
        gry = step.get("gry")
        if isinstance(grx, (int, float)) and isinstance(gry, (int, float)) and window_controller is not None:
            rect = window_controller.get_window_rect()
            if rect is not None:
                left, top, width, height = rect
                return int(left + float(grx) * max(1, width)), int(top + float(gry) * max(1, height))

        rx = step.get("rx")
        ry = step.get("ry")
        if isinstance(rx, (int, float)) and isinstance(ry, (int, float)):
            width = ctypes.windll.user32.GetSystemMetrics(0)
            height = ctypes.windll.user32.GetSystemMetrics(1)
            return int(float(rx) * width), int(float(ry) * height)
        return None, None

    @staticmethod
    def _mouse_click(x: int, y: int, button: str = "left", clicks: int = 1, interval_ms: int = 120) -> None:
        user32 = ctypes.windll.user32
        user32.SetCursorPos(int(x), int(y))

        left_down = 0x0002
        left_up = 0x0004
        right_down = 0x0008
        right_up = 0x0010
        down = right_down if button == "right" else left_down
        up = right_up if button == "right" else left_up

        for idx in range(max(1, clicks)):
            user32.mouse_event(down, 0, 0, 0, 0)
            user32.mouse_event(up, 0, 0, 0, 0)
            if idx < clicks - 1 and interval_ms > 0:
                time.sleep(interval_ms / 1000.0)


class GameWindowController:
    SW_RESTORE = 9
    HWND_TOPMOST = -1
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_NOACTIVATE = 0x0010

    def __init__(self, process_name: str) -> None:
        self._process_name = str(process_name or "").strip().lower()
        self._last_hwnd: int | None = None

    def keep_game_on_top(self) -> bool:
        hwnd = self._find_game_window()
        if hwnd is None:
            return False

        user32 = ctypes.windll.user32
        try:
            user32.ShowWindow(hwnd, self.SW_RESTORE)
            user32.SetForegroundWindow(hwnd)
            user32.BringWindowToTop(hwnd)
            user32.SetWindowPos(
                hwnd,
                self.HWND_TOPMOST,
                0,
                0,
                0,
                0,
                self.SWP_NOMOVE | self.SWP_NOSIZE | self.SWP_NOACTIVATE,
            )
        except Exception:
            return False
        return True

    def get_window_rect(self) -> tuple[int, int, int, int] | None:
        hwnd = self._find_game_window()
        if hwnd is None:
            return None

        rect = RECT()
        ok = ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
        if not ok:
            return None
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        return int(rect.left), int(rect.top), width, height

    def _find_game_window(self) -> int | None:
        if self._last_hwnd is not None and self._is_hwnd_alive(self._last_hwnd):
            return self._last_hwnd

        proc = find_process(self._process_name)
        if proc is None:
            return None
        pid = int(proc.pid)

        user32 = ctypes.windll.user32
        found: list[int] = []
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def _enum_cb(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            pid_out = ctypes.c_ulong(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_out))
            if int(pid_out.value) == pid:
                found.append(int(hwnd))
                return False
            return True

        user32.EnumWindows(EnumWindowsProc(_enum_cb), 0)
        if not found:
            return None
        self._last_hwnd = found[0]
        return self._last_hwnd

    @staticmethod
    def _is_hwnd_alive(hwnd: int) -> bool:
        user32 = ctypes.windll.user32
        return bool(user32.IsWindow(hwnd))


class DualRLAgent:
    def __init__(
        self,
        combat_model_path: str,
        noncombat_model_path: str,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_episodes: int,
        combat_horizon_weight: float = 0.3,
    ) -> None:
        self._combat_model_path = combat_model_path
        self._noncombat_model_path = noncombat_model_path
        self._combat_model: CombatPolicyModel | None = None
        self._noncombat_model: NonCombatPolicyModel | None = None
        self._eps_start = max(0.0, float(epsilon_start))
        self._eps_end = max(0.0, min(1.0, float(epsilon_end)))
        self._eps_decay = max(1, int(epsilon_decay_episodes))
        self._combat_horizon_weight = max(0.0, float(combat_horizon_weight))
        self.reload_models()

    def reload_models(self) -> None:
        self._combat_model = load_combat_policy_model(self._combat_model_path)
        self._noncombat_model = load_noncombat_policy_model(self._noncombat_model_path)

    def choose(self, state: GameStateSnapshot, episode_index: int) -> ActionCommand:
        candidates = self._candidates(state)
        if not candidates:
            return ActionCommand(action_type="noop", metadata={"reason": "no_candidates"})

        epsilon = self._epsilon(episode_index)
        if (not state.in_combat) and random.random() < epsilon:
            picked = random.choice(candidates)
            picked.metadata["policy"] = "explore"
            picked.metadata["epsilon"] = round(epsilon, 4)
            return picked

        if state.in_combat and self._combat_model is not None:
            scores = []
            for action in candidates:
                model_score = self._combat_model.score(state, action)
                horizon_score = self._short_horizon_score(state, action)
                total = model_score + self._combat_horizon_weight * horizon_score
                scores.append(total)
            idx = max(range(len(candidates)), key=lambda i: scores[i])
            picked = candidates[idx]
            picked.metadata["policy"] = "combat_model"
            picked.metadata["value"] = round(scores[idx], 4)
            return picked

        if (not state.in_combat) and self._noncombat_model is not None:
            scores = [self._noncombat_model.score(state, action) for action in candidates]
            idx = max(range(len(candidates)), key=lambda i: scores[i])
            picked = candidates[idx]
            picked.metadata["policy"] = "noncombat_model"
            picked.metadata["value"] = round(scores[idx], 4)
            return picked

        picked = random.choice(candidates)
        picked.metadata["policy"] = "random_no_model"
        return picked

    @staticmethod
    def _short_horizon_score(state: GameStateSnapshot, action: ActionCommand) -> float:
        if not state.in_combat:
            return 0.0
        try:
            sim = DeterministicSegmentSimulator(state)
            step = sim.apply(action)
            if not step.applied:
                return -3.0
            ended = action.action_type == "end_turn"
            score = sim.evaluate_branch(ended_turn=ended)
            return float(score.total)
        except Exception:
            return 0.0

    def _epsilon(self, episode_index: int) -> float:
        if episode_index <= 0:
            return self._eps_start
        ratio = min(1.0, float(episode_index) / float(self._eps_decay))
        return self._eps_start + (self._eps_end - self._eps_start) * ratio

    def _candidates(self, state: GameStateSnapshot) -> list[ActionCommand]:
        raw = state.raw_state if isinstance(state.raw_state, dict) else {}
        state_type = str(state.state_type or "").strip().lower()

        if state_type == "hand_select":
            return self._hand_select_candidates(raw)

        if state.in_combat and state_type in {"monster", "elite", "boss", ""}:
            return self._combat_candidates(raw)

        return self._noncombat_candidates(raw, state_type)

    def _hand_select_candidates(self, raw: dict[str, object]) -> list[ActionCommand]:
        actions: list[ActionCommand] = []
        section = raw.get("hand_select") if isinstance(raw.get("hand_select"), dict) else {}
        cards = section.get("cards") if isinstance(section.get("cards"), list) else []
        for card in cards:
            if isinstance(card, dict) and isinstance(card.get("index"), int):
                actions.append(ActionCommand(action_type="combat_select_card", option_index=int(card["index"])))
        if bool(section.get("can_confirm")):
            actions.append(ActionCommand(action_type="combat_confirm_selection"))
        return actions

    def _combat_candidates(self, raw: dict[str, object]) -> list[ActionCommand]:
        actions: list[ActionCommand] = []
        battle = raw.get("battle") if isinstance(raw.get("battle"), dict) else {}
        player = battle.get("player") if isinstance(battle.get("player"), dict) else {}

        # STS2MCP combat guards: only act during local player play phase.
        is_play_phase = battle.get("is_play_phase")
        if isinstance(is_play_phase, bool) and not is_play_phase:
            return [ActionCommand(action_type="noop", metadata={"policy": "combat_wait", "reason": "not_play_phase"})]

        turn = str(battle.get("turn") or "").strip().lower()
        if turn and turn not in {"player", "play", "local", "you"}:
            return [ActionCommand(action_type="noop", metadata={"policy": "combat_wait", "reason": f"turn={turn}"})]

        player_actions_disabled = battle.get("player_actions_disabled")
        if isinstance(player_actions_disabled, bool) and player_actions_disabled:
            return [ActionCommand(action_type="noop", metadata={"policy": "combat_wait", "reason": "player_actions_disabled"})]

        hand = player.get("hand") if isinstance(player.get("hand"), list) else []
        enemies = battle.get("enemies") if isinstance(battle.get("enemies"), list) else []
        targets = [
            str(enemy.get("entity_id"))
            for enemy in enemies
            if isinstance(enemy, dict) and int(enemy.get("hp") or 0) > 0 and enemy.get("entity_id")
        ]

        for card in hand:
            if not isinstance(card, dict):
                continue
            if not bool(card.get("can_play", False)):
                continue

            index = card.get("index")
            if not isinstance(index, int):
                continue

            card_id = str(card.get("id") or card.get("name") or "")
            target_type = str(card.get("target_type") or "").lower()
            metadata = {"card_index": index}
            cost = card.get("cost")
            if isinstance(cost, int):
                metadata["cost"] = cost

            if "enemy" in target_type and targets:
                for target_id in targets:
                    actions.append(
                        ActionCommand(
                            action_type="play_card",
                            card_id=card_id,
                            target_id=target_id,
                            metadata=metadata.copy(),
                        )
                    )
            else:
                actions.append(
                    ActionCommand(
                        action_type="play_card",
                        card_id=card_id,
                        metadata=metadata,
                    )
                )

        potions = player.get("potions") if isinstance(player.get("potions"), list) else []
        for potion in potions:
            if not isinstance(potion, dict):
                continue
            slot = potion.get("slot")
            if not isinstance(slot, int):
                continue

            can_use_in_combat = potion.get("can_use_in_combat")
            if isinstance(can_use_in_combat, bool) and not can_use_in_combat:
                continue

            usage = str(potion.get("usage") or "").strip().lower()
            if usage in {"automatic", "auto"}:
                continue

            potion_id = str(potion.get("id") or potion.get("potion_id") or "")
            target_type = str(potion.get("target_type") or "").lower()
            metadata: dict[str, str | int | float | bool] = {"slot": slot, "target_type": target_type}
            if potion_id:
                metadata["potion_id"] = potion_id
            if "enemy" in target_type and targets:
                for target_id in targets:
                    actions.append(ActionCommand(action_type="use_potion", option_index=slot, target_id=target_id, metadata=metadata.copy()))
            else:
                actions.append(ActionCommand(action_type="use_potion", option_index=slot, metadata=metadata))

        # Only offer end_turn when no other actionable combat moves are available.
        has_non_end_action = any(a.action_type in {"play_card", "use_potion"} for a in actions)
        if not has_non_end_action:
            actions.append(ActionCommand(action_type="end_turn"))
        return actions

    def _noncombat_candidates(self, raw: dict[str, object], state_type: str) -> list[ActionCommand]:
        actions: list[ActionCommand] = []

        if state_type == "combat_rewards":
            rewards = raw.get("rewards") if isinstance(raw.get("rewards"), dict) else {}
            items = rewards.get("items") if isinstance(rewards.get("items"), list) else []
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("index"), int):
                    actions.append(ActionCommand(action_type="claim_reward", option_index=int(item["index"])))
            if bool(rewards.get("can_proceed")):
                actions.append(ActionCommand(action_type="proceed"))

        elif state_type == "card_reward":
            section = raw.get("card_reward") if isinstance(raw.get("card_reward"), dict) else {}
            cards = section.get("cards") if isinstance(section.get("cards"), list) else []
            for card in cards:
                if isinstance(card, dict) and isinstance(card.get("index"), int):
                    actions.append(ActionCommand(action_type="select_card_reward", option_index=int(card["index"])))
            if bool(section.get("can_skip")):
                actions.append(ActionCommand(action_type="skip_card_reward"))

        elif state_type == "event":
            section = raw.get("event") if isinstance(raw.get("event"), dict) else {}
            if bool(section.get("in_dialogue")):
                actions.append(ActionCommand(action_type="advance_dialogue"))
            options = section.get("options") if isinstance(section.get("options"), list) else []
            unlocked_index = 0
            for item in options:
                if isinstance(item, dict) and isinstance(item.get("index"), int) and not bool(item.get("is_locked", False)):
                    actions.append(
                        ActionCommand(
                            action_type="event_choose",
                            option_index=unlocked_index,
                            metadata={
                                "event_option_raw_index": int(item["index"]),
                                "event_option_title": str(item.get("title") or ""),
                            },
                        )
                    )
                    unlocked_index += 1

        elif state_type == "map":
            section = raw.get("map") if isinstance(raw.get("map"), dict) else {}
            options = section.get("next_options") if isinstance(section.get("next_options"), list) else []
            for node in options:
                if isinstance(node, dict) and isinstance(node.get("index"), int):
                    actions.append(ActionCommand(action_type="map_choose", option_index=int(node["index"])))

        elif state_type == "rest_site":
            section = raw.get("rest_site") if isinstance(raw.get("rest_site"), dict) else {}
            options = section.get("options") if isinstance(section.get("options"), list) else []
            enabled_index = 0
            for item in options:
                if isinstance(item, dict) and isinstance(item.get("index"), int) and bool(item.get("is_enabled", False)):
                    actions.append(
                        ActionCommand(
                            action_type="choose_rest_option",
                            option_index=enabled_index,
                            metadata={
                                "rest_option_raw_index": int(item["index"]),
                                "rest_option_id": str(item.get("id") or ""),
                            },
                        )
                    )
                    enabled_index += 1
            if bool(section.get("can_proceed")):
                actions.append(ActionCommand(action_type="proceed"))

        elif state_type == "shop":
            section = raw.get("shop") if isinstance(raw.get("shop"), dict) else {}
            items = section.get("items") if isinstance(section.get("items"), list) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                if not bool(item.get("is_stocked", False)):
                    continue
                if not bool(item.get("can_afford", False)):
                    continue
                if isinstance(item.get("index"), int):
                    metadata: dict[str, str | int | float | bool] = {
                        "shop_category": str(item.get("category") or ""),
                        "cost": int(item.get("cost") or 0),
                        "is_on_sale": bool(item.get("on_sale", False)),
                    }
                    # Expose stable identifiers for RL features.
                    if item.get("card_id"):
                        metadata["shop_item_id"] = str(item.get("card_id"))
                        metadata["shop_item_kind"] = "card"
                    elif item.get("relic_id"):
                        metadata["shop_item_id"] = str(item.get("relic_id"))
                        metadata["shop_item_kind"] = "relic"
                    elif item.get("potion_id"):
                        metadata["shop_item_id"] = str(item.get("potion_id"))
                        metadata["shop_item_kind"] = "potion"
                    elif str(item.get("category") or "") == "card_removal":
                        metadata["shop_item_id"] = "card_removal"
                        metadata["shop_item_kind"] = "service"

                    actions.append(
                        ActionCommand(
                            action_type="shop_purchase",
                            option_index=int(item["index"]),
                            metadata=metadata,
                        )
                    )
            # Always include proceed from shop: MCP 'proceed' first closes inventory/back then advances.
            actions.append(ActionCommand(action_type="proceed", metadata={"shop_proceed_attempt": True}))

        elif state_type == "card_select":
            section = raw.get("card_select") if isinstance(raw.get("card_select"), dict) else {}
            cards = section.get("cards") if isinstance(section.get("cards"), list) else []
            for card in cards:
                if isinstance(card, dict) and isinstance(card.get("index"), int):
                    actions.append(ActionCommand(action_type="select_card", option_index=int(card["index"])))
            if bool(section.get("can_confirm")):
                actions.append(ActionCommand(action_type="confirm_selection"))
            if bool(section.get("can_cancel")):
                actions.append(ActionCommand(action_type="cancel_selection"))

        elif state_type == "relic_select":
            section = raw.get("relic_select") if isinstance(raw.get("relic_select"), dict) else {}
            relics = section.get("relics") if isinstance(section.get("relics"), list) else []
            for relic in relics:
                if isinstance(relic, dict) and isinstance(relic.get("index"), int):
                    actions.append(ActionCommand(action_type="select_relic", option_index=int(relic["index"])))
            if bool(section.get("can_skip")):
                actions.append(ActionCommand(action_type="skip_relic_selection"))

        elif state_type == "treasure":
            section = raw.get("treasure") if isinstance(raw.get("treasure"), dict) else {}
            relics = section.get("relics") if isinstance(section.get("relics"), list) else []
            for relic in relics:
                if isinstance(relic, dict) and isinstance(relic.get("index"), int):
                    actions.append(ActionCommand(action_type="claim_treasure_relic", option_index=int(relic["index"])))
            if bool(section.get("can_proceed")):
                actions.append(ActionCommand(action_type="proceed"))

        if not actions:
            actions.append(ActionCommand(action_type="noop", metadata={"reason": "screen_no_action", "state_type": state_type}))
        return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file dual-RL self-learning runner")
    parser.add_argument("--episodes", type=int, default=0, help="0 means keep running")
    parser.add_argument("--process-name", default="SlayTheSpire2.exe")
    parser.add_argument("--game-exe", default=None, help="Optional game exe path for auto launch")
    parser.add_argument("--mcp-host", default="127.0.0.1")
    parser.add_argument("--mcp-port", type=int, default=15526)
    parser.add_argument("--trace-file", default="runtime/planner_action_trace.selfplay.jsonl")
    parser.add_argument("--metrics-file", default="runtime/selfplay_metrics.jsonl")
    parser.add_argument("--combat-model", default="runtime/combat_policy_model.json")
    parser.add_argument("--noncombat-model", default="runtime/noncombat_policy_model.json")
    parser.add_argument("--retrain-every", type=int, default=5)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--combat-train-every-combat", action="store_true", default=False)
    parser.add_argument("--no-combat-train-every-combat", action="store_false", dest="combat_train_every_combat")
    parser.add_argument("--global-train-every-episode", action="store_true", default=True)
    parser.add_argument("--no-global-train-every-episode", action="store_false", dest="global_train_every_episode")
    parser.add_argument("--combat-train-epochs", type=int, default=2)
    parser.add_argument("--global-train-epochs", type=int, default=8)
    parser.add_argument("--replay-recent-segments", type=int, default=24, help="How many latest trace segments are considered 'new' for mixed replay")
    parser.add_argument("--replay-new-ratio", type=float, default=0.4, help="Training sample ratio from recent segments (0-1)")
    parser.add_argument("--replay-max-train-samples", type=int, default=12000, help="Per-train cap after mixed replay sampling, 0 disables cap")
    parser.add_argument("--combat-hidden1", type=int, default=768)
    parser.add_argument("--combat-hidden2", type=int, default=512)
    parser.add_argument("--noncombat-hidden1", type=int, default=768)
    parser.add_argument("--noncombat-hidden2", type=int, default=512)
    parser.add_argument("--step-delay-ms", type=int, default=250)
    parser.add_argument("--action-timeout-ms", type=int, default=5000)
    parser.add_argument("--min-action-interval-ms", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=0.25)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=120)
    parser.add_argument("--combat-horizon-weight", type=float, default=0.3)
    parser.add_argument("--menu-mouse-config", default="config/auto_menu_mouse.local.json")
    parser.add_argument("--menu-seq-start", default="start_singleplayer_defect")
    parser.add_argument("--menu-seq-end", default="end_run_to_menu")
    parser.add_argument("--menu-seq-return", default="return_to_main_menu")
    parser.add_argument("--menu-seq-continue", default="post_run_continue")
    parser.add_argument("--menu-seq-cooldown-ms", type=int, default=2200)
    parser.add_argument("--soft-loop-streak", type=int, default=6, help="Trigger soft-loop recovery when the same non-combat signature repeats consecutively")
    parser.add_argument("--soft-loop-window", type=int, default=16, help="Sliding window size for non-combat signature repetition checks")
    parser.add_argument("--soft-loop-hit-limit", type=int, default=10, help="Trigger soft-loop recovery when a signature appears this many times in the window")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.dry_run:
        print("[dry-run] dual RL self-learning entrypoint ready")
        print("[dry-run] trace_file=", args.trace_file)
        print("[dry-run] combat_model=", args.combat_model)
        print("[dry-run] noncombat_model=", args.noncombat_model)
        return

    ensure_game_running(args.process_name, args.game_exe)

    reader = McpApiReader(McpApiReaderConfig(host=args.mcp_host, port=args.mcp_port, mode="singleplayer"))
    executor = McpPostActionExecutor(McpPostExecutorConfig(host=args.mcp_host, port=args.mcp_port, mode="singleplayer"), allow_live_actions=True)

    trace_path = Path(args.trace_file)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    agent = DualRLAgent(
        combat_model_path=args.combat_model,
        noncombat_model_path=args.noncombat_model,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        combat_horizon_weight=float(args.combat_horizon_weight),
    )
    report_model_complexity(args.combat_model, args.noncombat_model)

    menu_controller = MouseMenuController(args.menu_mouse_config)
    window_controller = GameWindowController(args.process_name)
    if not menu_controller.has_sequence(args.menu_seq_start):
        print(
            "[warn] start menu sequence not found; auto start disabled until configured: "
            f"{args.menu_mouse_config} -> {args.menu_seq_start}"
        )

    run_loop(args, repo_root, reader, executor, agent, trace_path, metrics_path, menu_controller, window_controller)


def run_loop(
    args: argparse.Namespace,
    repo_root: Path,
    reader: McpApiReader,
    executor: McpPostActionExecutor,
    agent: DualRLAgent,
    trace_path: Path,
    metrics_path: Path,
    menu_controller: MouseMenuController,
    window_controller: GameWindowController,
) -> None:
    episode_idx = 0
    in_episode = False
    episode_start_ms = 0
    hp_start = 0
    max_act = 0
    max_floor = 0
    last_action_ms = 0
    last_menu_recovery_ms = 0
    card_select_memory = CardSelectMemory()
    reward_memory = RewardMemory()
    shop_memory = ShopMemory()
    noop_streak = 0
    saw_act3_boss = False
    same_noncombat_signature_streak = 0
    recent_noncombat_signatures: deque[tuple[object, ...]] = deque(maxlen=max(4, int(args.soft_loop_window)))

    print("[dual-rl] self-learning started")

    while True:
        window_controller.keep_game_on_top()
        state = reader.read_state()
        state_type = str(state.state_type or "unknown")
        act, floor = extract_run_progress(state)
        max_act = max(max_act, act)
        max_floor = max(max_floor, floor)
        if str(state.state_type or "").strip().lower() == "boss" and act >= 3:
            saw_act3_boss = True

        if state_type == "menu":
            if in_episode:
                episode_idx += 1
                hp_end = int(state.player.hp)
                stats = EpisodeStats(
                    episode=episode_idx,
                    start_ms=episode_start_ms,
                    end_ms=int(time.time() * 1000),
                    max_act=max_act,
                    max_floor=max_floor,
                    hp_start=hp_start,
                    hp_end=hp_end,
                    hp_loss=max(0, hp_start - hp_end),
                    died=hp_end <= 0,
                )
                write_episode_metrics(metrics_path, stats)
                print(
                    f"[dual-rl] episode={stats.episode} act={stats.max_act} floor={stats.max_floor} "
                    f"hp_loss={stats.hp_loss} died={stats.died}"
                )

                trained_any = False
                if bool(args.global_train_every_episode):
                    trained_any = retrain_dual_models(
                        repo_root=repo_root,
                        trace_path=trace_path,
                        combat_model_out=args.combat_model,
                        noncombat_model_out=args.noncombat_model,
                        epochs=max(1, int(args.global_train_epochs)),
                        combat_hidden1=int(args.combat_hidden1),
                        combat_hidden2=int(args.combat_hidden2),
                        noncombat_hidden1=int(args.noncombat_hidden1),
                        noncombat_hidden2=int(args.noncombat_hidden2),
                        replay_recent_segments=max(0, int(args.replay_recent_segments)),
                        replay_new_ratio=float(args.replay_new_ratio),
                        replay_max_train_samples=max(0, int(args.replay_max_train_samples)),
                    ) or trained_any

                # Optional periodic full refresh (both models) remains available.
                if args.retrain_every > 0 and stats.episode % args.retrain_every == 0:
                    trained_any = retrain_dual_models(
                        repo_root,
                        trace_path,
                        args.combat_model,
                        args.noncombat_model,
                        max(1, int(args.train_epochs)),
                        combat_hidden1=int(args.combat_hidden1),
                        combat_hidden2=int(args.combat_hidden2),
                        noncombat_hidden1=int(args.noncombat_hidden1),
                        noncombat_hidden2=int(args.noncombat_hidden2),
                        replay_recent_segments=max(0, int(args.replay_recent_segments)),
                        replay_new_ratio=float(args.replay_new_ratio),
                        replay_max_train_samples=max(0, int(args.replay_max_train_samples)),
                    ) or trained_any

                if trained_any:
                    agent.reload_models()
                    report_model_complexity(args.combat_model, args.noncombat_model)
                    print("[dual-rl] models reloaded")

                if args.episodes > 0 and stats.episode >= args.episodes:
                    print("[dual-rl] target episodes reached, stop")
                    return

            in_episode = False
            max_act = 0
            max_floor = 0
            saw_act3_boss = False
            noop_streak = 0

            # Auto-enter singleplayer and start Defect run via mouse sequence.
            print("[flow] menu detected -> running start sequence")
            window_controller.keep_game_on_top()
            menu_controller.run_sequence(
                args.menu_seq_start,
                min_interval_ms=args.menu_seq_cooldown_ms,
                window_controller=window_controller,
            )
            time.sleep(1.0)
            continue

        terminal_reason = detect_episode_terminal_reason(state, saw_act3_boss)
        if in_episode and terminal_reason is not None:
            print(f"[flow] terminal state detected ({terminal_reason}) -> running end/recovery flow")
            now_ms = int(time.time() * 1000)
            if now_ms - last_menu_recovery_ms < max(600, int(args.menu_seq_cooldown_ms)):
                time.sleep(0.5)
                continue
            last_menu_recovery_ms = now_ms

            # Hard-coded defeat exit flow requested by user:
            # click twice at fixed position with 1s interval, then wait 3s.
            if terminal_reason == "player_hp_zero":
                window_controller.keep_game_on_top()
                click_fixed_defeat_exit_button()
                time.sleep(DEFEAT_EXIT_POST_WAIT_S)
                continue

            # Fallback recovery when dedicated end button sequence is unavailable.
            window_controller.keep_game_on_top()
            used = run_return_sequence_twice(
                menu_controller=menu_controller,
                window_controller=window_controller,
                sequence_name=args.menu_seq_return,
            )
            if not used:
                window_controller.keep_game_on_top()
                menu_controller.run_sequence(
                    args.menu_seq_continue,
                    min_interval_ms=args.menu_seq_cooldown_ms,
                    window_controller=window_controller,
                )
            time.sleep(1.0)
            continue

        if not in_episode:
            in_episode = True
            episode_start_ms = int(time.time() * 1000)
            hp_start = max(0, int(state.player.hp))
            max_act = act
            max_floor = floor
            same_noncombat_signature_streak = 0
            recent_noncombat_signatures.clear()
            print(f"[dual-rl] episode-start: act={act} floor={floor}")

        now_ms = int(time.time() * 1000)
        remaining = args.min_action_interval_ms - (now_ms - last_action_ms)
        if remaining > 0:
            time.sleep(remaining / 1000.0)

        before = state
        forced = select_forced_noncombat_action(before, card_select_memory, reward_memory, shop_memory)
        action = forced if forced is not None else agent.choose(before, episode_idx)
        window_controller.keep_game_on_top()
        result = executor.execute(action)
        after = wait_state_change(reader, before, timeout_ms=args.action_timeout_ms)
        last_action_ms = int(time.time() * 1000)

        if (
            (before.state_type or "").strip().lower() == "combat_rewards"
            and action.action_type == "claim_reward"
            and action.option_index is not None
        ):
            no_progress = _combat_rewards_key_from_state(before.raw_state) == _combat_rewards_key_from_state(after.raw_state)
            should_mark_failed = (not result.accepted) or no_progress
            if reward_memory.failed_claim_indices is None:
                reward_memory.failed_claim_indices = set()
            if should_mark_failed:
                reward_memory.failed_claim_indices.add(int(action.option_index))

        if (
            (before.state_type or "").strip().lower() == "shop"
            and action.action_type == "shop_purchase"
            and not result.accepted
            and action.option_index is not None
        ):
            if shop_memory.failed_purchase_indices is None:
                shop_memory.failed_purchase_indices = set()
            shop_memory.failed_purchase_indices.add(int(action.option_index))

        transition = summarize_transition(before, after)
        record = {
            "iteration": int(time.time() * 1000),
            "timestamp_ms": int(time.time() * 1000),
            "action": action.model_dump(),
            "result": result.model_dump(),
            "before": compact_state(before),
            "after": compact_state(after),
            "transition": transition,
        }
        append_jsonl(trace_path, record)

        if bool(args.combat_train_every_combat) and bool(transition.get("combat_ended")):
            trained = retrain_combat_model(
                repo_root=repo_root,
                trace_path=trace_path,
                combat_model_out=args.combat_model,
                noncombat_model_out=args.noncombat_model,
                epochs=max(1, int(args.combat_train_epochs)),
                combat_hidden1=int(args.combat_hidden1),
                combat_hidden2=int(args.combat_hidden2),
                noncombat_hidden1=int(args.noncombat_hidden1),
                noncombat_hidden2=int(args.noncombat_hidden2),
                replay_recent_segments=max(0, int(args.replay_recent_segments)),
                replay_new_ratio=float(args.replay_new_ratio),
                replay_max_train_samples=max(0, int(args.replay_max_train_samples)),
            )
            if trained:
                agent.reload_models()
                print("[dual-rl] combat model updated after combat")

        decision = action.action_type
        if action.card_id:
            decision += f"({action.card_id})"
        elif action.option_index is not None:
            decision += f"[{action.option_index}]"
        policy = str(action.metadata.get("policy") or "unknown")
        accepted = "ok" if result.accepted else "reject"
        print(f"[dual-rl] {policy} -> {decision} => {accepted} | {result.message}")

        if action.action_type == "noop" and (not before.in_combat):
            noop_streak += 1
        else:
            noop_streak = 0

        if (not before.in_combat) and (not after.in_combat):
            after_sig = state_signature(after)
            before_sig = state_signature(before)
            if recent_noncombat_signatures and recent_noncombat_signatures[-1] == after_sig:
                same_noncombat_signature_streak += 1
            else:
                same_noncombat_signature_streak = 1
            recent_noncombat_signatures.append(after_sig)

            hit_count = sum(1 for sig in recent_noncombat_signatures if sig == after_sig)
            no_progress = before_sig == after_sig
            soft_loop_triggered = (
                same_noncombat_signature_streak >= max(2, int(args.soft_loop_streak))
                or hit_count >= max(3, int(args.soft_loop_hit_limit))
                or (no_progress and noop_streak >= 3)
            )
            if soft_loop_triggered:
                print(
                    "[flow] soft loop detected "
                    f"state={after.state_type} streak={same_noncombat_signature_streak} hits={hit_count}"
                )
                window_controller.keep_game_on_top()
                used = run_return_sequence_twice(
                    menu_controller=menu_controller,
                    window_controller=window_controller,
                    sequence_name=args.menu_seq_return,
                )
                if not used:
                    window_controller.keep_game_on_top()
                    menu_controller.run_sequence(
                        args.menu_seq_continue,
                        min_interval_ms=args.menu_seq_cooldown_ms,
                        window_controller=window_controller,
                    )
                noop_streak = 0
                same_noncombat_signature_streak = 0
                recent_noncombat_signatures.clear()
                time.sleep(1.0)
                continue
        else:
            same_noncombat_signature_streak = 0
            recent_noncombat_signatures.clear()

        if noop_streak >= 3 and (not before.in_combat):
            print("[flow] non-combat noop streak detected -> forcing return sequence")
            window_controller.keep_game_on_top()
            used = run_return_sequence_twice(
                menu_controller=menu_controller,
                window_controller=window_controller,
                sequence_name=args.menu_seq_return,
            )
            if not used:
                window_controller.keep_game_on_top()
                menu_controller.run_sequence(
                    args.menu_seq_continue,
                    min_interval_ms=args.menu_seq_cooldown_ms,
                    window_controller=window_controller,
                )
            noop_streak = 0
            time.sleep(1.0)
            continue

        time.sleep(max(0, args.step_delay_ms) / 1000.0)


def run_return_sequence_twice(
    menu_controller: MouseMenuController,
    window_controller: GameWindowController,
    sequence_name: str,
) -> bool:
    first = menu_controller.run_sequence(
        sequence_name,
        min_interval_ms=0,
        window_controller=window_controller,
    )
    time.sleep(RETURN_SEQUENCE_REPEAT_INTERVAL_S)
    window_controller.keep_game_on_top()
    second = menu_controller.run_sequence(
        sequence_name,
        min_interval_ms=0,
        window_controller=window_controller,
    )
    return bool(first or second)


def select_forced_noncombat_action(
    state: GameStateSnapshot,
    memory: CardSelectMemory,
    reward_memory: RewardMemory,
    shop_memory: ShopMemory,
) -> ActionCommand | None:
    state_type = str(state.state_type or "").strip().lower()
    raw = state.raw_state if isinstance(state.raw_state, dict) else {}

    # Combat rewards: claim everything first, then proceed.
    if state_type == "combat_rewards":
        rewards = raw.get("rewards") if isinstance(raw.get("rewards"), dict) else {}
        items = rewards.get("items") if isinstance(rewards.get("items"), list) else []
        can_proceed = bool(rewards.get("can_proceed", False))
        key = _combat_rewards_key(rewards)
        if reward_memory.key != key or reward_memory.failed_claim_indices is None:
            reward_memory.reset(key)

        player = rewards.get("player") if isinstance(rewards.get("player"), dict) else {}
        open_slots = _to_int_or_none(player.get("open_potion_slots"))

        indexes = sorted(
            {
                int(item.get("index"))
                for item in items
                if isinstance(item, dict) and isinstance(item.get("index"), int)
            },
            reverse=True,
        )
        for idx in indexes:
            if idx in reward_memory.failed_claim_indices:
                continue
            matched = next((item for item in items if isinstance(item, dict) and int(item.get("index") or -1) == idx), None)
            if matched is not None and str(matched.get("type") or "").lower() == "potion" and open_slots is not None and open_slots <= 0:
                reward_memory.failed_claim_indices.add(idx)
                continue
            return ActionCommand(action_type="claim_reward", option_index=idx, metadata={"policy": "forced_combat_rewards"})

        if can_proceed:
            return ActionCommand(action_type="proceed", metadata={"policy": "forced_combat_rewards"})

    # Shop exit assist: purchases are RL-driven, but avoid noop-stall by forcing proceed when nothing is buyable.
    if state_type == "shop":
        section = raw.get("shop") if isinstance(raw.get("shop"), dict) else {}
        items = section.get("items") if isinstance(section.get("items"), list) else []
        key = f"shop|items:{len(items)}"
        if shop_memory.key != key or shop_memory.failed_purchase_indices is None:
            shop_memory.reset(key)

        buyable = [
            int(item.get("index"))
            for item in items
            if isinstance(item, dict)
            and isinstance(item.get("index"), int)
            and bool(item.get("is_stocked", False))
            and bool(item.get("can_afford", False))
            and int(item.get("index")) not in shop_memory.failed_purchase_indices
        ]
        if not buyable:
            return ActionCommand(action_type="proceed", metadata={"policy": "forced_shop_exit"})

    # Card reward is a strict screen: choose one card or skip.
    if state_type == "card_reward":
        section = raw.get("card_reward") if isinstance(raw.get("card_reward"), dict) else {}
        cards = section.get("cards") if isinstance(section.get("cards"), list) else []
        card_indexes = [
            int(card.get("index"))
            for card in cards
            if isinstance(card, dict) and isinstance(card.get("index"), int)
        ]
        if card_indexes:
            # Follow STS2MCP guidance: use indices from state directly.
            return ActionCommand(action_type="select_card_reward", option_index=card_indexes[0], metadata={"policy": "forced_card_reward"})
        if bool(section.get("can_skip")):
            return ActionCommand(action_type="skip_card_reward", metadata={"policy": "forced_card_reward"})

    # Event room handling: advance dialogue first, then pick a stable option policy.
    if state_type == "event":
        section = raw.get("event") if isinstance(raw.get("event"), dict) else {}
        if bool(section.get("in_dialogue", False)):
            return ActionCommand(action_type="advance_dialogue", metadata={"policy": "forced_event_dialogue"})

        options = section.get("options") if isinstance(section.get("options"), list) else []
        unlocked: list[tuple[int, dict[str, object]]] = []
        unlocked_idx = 0
        for item in options:
            if not isinstance(item, dict):
                continue
            if bool(item.get("is_locked", False)):
                continue
            unlocked.append((unlocked_idx, item))
            unlocked_idx += 1

        if unlocked:
            # Prefer explicit leave/proceed options, then the safest-looking option.
            for idx, item in unlocked:
                if bool(item.get("is_proceed", False)):
                    return ActionCommand(action_type="event_choose", option_index=idx, metadata={"policy": "forced_event_proceed"})

            safest = sorted(unlocked, key=lambda pair: _event_option_risk_score(pair[1]))[0]
            return ActionCommand(action_type="event_choose", option_index=safest[0], metadata={"policy": "forced_event_safe"})

    # Card select follows STS2MCP semantics:
    # - choose screen: select_card(index) immediately resolves
    # - grid screen: select_card until preview/can_confirm, then confirm_selection
    if state_type == "card_select":
        section = raw.get("card_select") if isinstance(raw.get("card_select"), dict) else {}
        screen_type = str(section.get("screen_type") or "").strip().lower()
        prompt = str(section.get("prompt") or "")
        preview_showing = bool(section.get("preview_showing", False))
        can_confirm = bool(section.get("can_confirm", False))
        can_cancel = bool(section.get("can_cancel", False))
        cards = section.get("cards") if isinstance(section.get("cards"), list) else []
        indices = [
            int(card.get("index"))
            for card in cards
            if isinstance(card, dict) and isinstance(card.get("index"), int)
        ]
        required_count = _extract_required_card_count(prompt)
        preferred_indices = _preferred_card_select_indices(cards, prompt)
        if not preferred_indices:
            preferred_indices = list(indices)

        key = f"{screen_type}|{prompt}|{len(indices)}|need:{required_count}"
        if memory.key != key or memory.picked is None:
            memory.reset(key, required_count)

        if screen_type == "choose":
            # STS2MCP choose-a-card overlays often come from potion/event effects.
            # If skip is available, prefer skipping low-value optional choices to avoid soft-lock loops.
            if can_cancel and _should_skip_choose_screen(prompt, cards):
                return ActionCommand(action_type="cancel_selection", metadata={"policy": "forced_card_select_choose_skip"})
            if preferred_indices:
                return ActionCommand(
                    action_type="select_card",
                    option_index=preferred_indices[0],
                    metadata={"policy": "forced_card_select_choose"},
                )
            if can_cancel:
                return ActionCommand(action_type="cancel_selection", metadata={"policy": "forced_card_select_choose"})
            return ActionCommand(action_type="noop", metadata={"policy": "forced_card_select_choose", "reason": "no_choose_cards"})

        if preview_showing or can_confirm:
            return ActionCommand(action_type="confirm_selection", metadata={"policy": "forced_card_select_confirm"})

        # Pick until required count is reached, then wait for confirm to become enabled.
        if len(memory.picked) >= memory.required_count:
            return ActionCommand(action_type="noop", metadata={"policy": "forced_card_select_wait_confirm", "required": memory.required_count})

        for idx in preferred_indices:
            if idx not in memory.picked:
                memory.picked.add(idx)
                return ActionCommand(action_type="select_card", option_index=idx, metadata={"policy": "forced_card_select_pick"})

        # If all known cards were tried but confirm is still unavailable, avoid repeated toggle spam.
        return ActionCommand(action_type="noop", metadata={"policy": "forced_card_select_recover", "reason": "await_confirm_state"})
    return None


def _extract_required_card_count(prompt: str) -> int:
    text = str(prompt or "")
    # English patterns: "Choose 2 cards", "Select 1 card".
    match = re.search(r"(?i)(choose|select)\s+(\d+)\s+card", text)
    if match:
        return max(1, int(match.group(2)))
    # Chinese-like patterns: "选择2张", "选 3 张卡".
    match = re.search(r"(?:选择|选)\s*(\d+)\s*张", text)
    if match:
        return max(1, int(match.group(1)))
    return 1


def _preferred_card_select_indices(cards: list[object], prompt: str) -> list[int]:
    parsed_cards: list[tuple[int, dict[str, object]]] = []
    for item in cards:
        if isinstance(item, dict) and isinstance(item.get("index"), int):
            parsed_cards.append((int(item["index"]), item))
    if not parsed_cards:
        return []

    intent = _card_select_intent(prompt)
    if intent in {"exhaust", "discard", "remove", "transform"}:
        ranked = sorted(
            parsed_cards,
            key=lambda pair: (-_card_select_trash_score(pair[1]), pair[0]),
        )
        return [idx for idx, _ in ranked]

    return [idx for idx, _ in parsed_cards]


def _should_skip_choose_screen(prompt: str, cards: list[object]) -> bool:
    text = str(prompt or "").strip().lower()
    if not text:
        return False

    # Optional potion/event offers are usually safe to skip when cancel is enabled.
    optional_tokens = [
        "potion",
        "药水",
        "colorless",
        "无色",
        "discover",
        "发现",
        "create",
        "获得一张",
        "choose a card",
    ]
    if any(token in text for token in optional_tokens):
        return True

    card_count = sum(1 for item in cards if isinstance(item, dict) and isinstance(item.get("index"), int))
    # Large random offer pools are typically optional; skip to keep run flow stable.
    if card_count >= 5 and ("choose" in text or "选择" in text):
        return True
    return False


def _card_select_intent(prompt: str) -> str:
    text = str(prompt or "").strip().lower()
    if not text:
        return "unknown"
    if any(token in text for token in ["消耗", "exhaust"]):
        return "exhaust"
    if any(token in text for token in ["丢弃", "弃置", "discard"]):
        return "discard"
    if any(token in text for token in ["移除", "删除", "purge", "remove"]):
        return "remove"
    if any(token in text for token in ["转化", "transform"]):
        return "transform"
    return "other"


def _card_select_trash_score(card: dict[str, object]) -> float:
    score = 0.0
    card_type = str(card.get("type") or "").strip().lower()
    card_id = str(card.get("id") or "").strip().lower()
    card_name = str(card.get("name") or "").strip().lower()
    rarity = str(card.get("rarity") or "").strip().lower()

    if card_type in {"curse", "status"}:
        score += 100.0

    bad_tokens = {
        "wound",
        "dazed",
        "burn",
        "void",
        "slimed",
        "regret",
        "injury",
        "clumsy",
        "normality",
        "疑虑",
        "痛苦",
        "受伤",
        "灼伤",
        "淤泥",
        "眩晕",
    }
    haystack = f"{card_id} {card_name}"
    if any(token in haystack for token in bad_tokens):
        score += 80.0

    if any(token in haystack for token in ["strike", "defend", "打击", "防御"]):
        score += 35.0

    if rarity == "common":
        score += 8.0
    elif rarity == "uncommon":
        score += 3.0

    cost = card.get("cost")
    if isinstance(cost, int):
        score += min(4.0, float(max(0, cost)))
    elif isinstance(cost, str) and cost.strip().isdigit():
        score += min(4.0, float(max(0, int(cost.strip()))))

    if bool(card.get("is_upgraded", False)):
        score -= 10.0
    return score


def _event_option_risk_score(option: dict[str, object]) -> float:
    title = str(option.get("title") or "").strip().lower()
    desc = str(option.get("description") or "").strip().lower()
    text = f"{title} {desc}"

    if any(token in text for token in ["leave", "continue", "离开", "继续", "跳过"]):
        return -10.0

    risk = 0.0
    if any(token in text for token in ["lose", "失去", "hp", "生命", "伤害"]):
        risk += 6.0
    if any(token in text for token in ["curse", "诅咒"]):
        risk += 8.0
    if any(token in text for token in ["gold", "金币"]):
        risk += 2.0
    if any(token in text for token in ["fight", "battle", "战斗"]):
        risk += 5.0
    return risk



def is_terminal_loss_state(state: GameStateSnapshot) -> bool:
    state_type = str(state.state_type or "").strip().lower()
    if state_type == "menu":
        return False

    raw = state.raw_state if isinstance(state.raw_state, dict) else {}

    # In combat, hp<=0 is authoritative and should trigger recovery.
    if state.in_combat and state.player.hp <= 0:
        return True

    # Outside combat, only trust explicit raw-state hp fields (not normalized fallback 0 values).
    hp = extract_explicit_player_hp(raw)
    if hp is None:
        return False
    return hp <= 0


def detect_episode_terminal_reason(state: GameStateSnapshot, saw_act3_boss: bool) -> str | None:
    if is_terminal_loss_state(state):
        return "player_hp_zero"
    if is_terminal_win_state(state, saw_act3_boss):
        return "act3_boss_defeated"
    return None


def is_terminal_win_state(state: GameStateSnapshot, saw_act3_boss: bool) -> bool:
    raw = state.raw_state if isinstance(state.raw_state, dict) else {}
    run = raw.get("run") if isinstance(raw.get("run"), dict) else {}

    for key in ["victory", "won", "is_victory", "is_win"]:
        value = run.get(key)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, (int, float)) and int(value) == 1:
            return True

    act = _to_int_or_none(run.get("act")) or 0
    state_type = str(state.state_type or "").strip().lower()
    if saw_act3_boss and (not state.in_combat) and state.player.hp > 0 and act >= 3:
        return state_type in {"menu", "victory", "win", "game_over", "credits"}
    return False


def extract_explicit_player_hp(raw: dict[str, object]) -> int | None:
    # battle.player.hp
    battle = raw.get("battle") if isinstance(raw.get("battle"), dict) else {}
    battle_player = battle.get("player") if isinstance(battle, dict) else {}
    hp = _to_int_or_none(battle_player.get("hp") if isinstance(battle_player, dict) else None)
    if hp is not None:
        return hp

    # top-level player.hp
    top_player = raw.get("player") if isinstance(raw.get("player"), dict) else {}
    hp = _to_int_or_none(top_player.get("hp") if isinstance(top_player, dict) else None)
    if hp is not None:
        return hp

    # multiplayer players[local].hp if available
    players = raw.get("players") if isinstance(raw.get("players"), list) else []
    local_slot = _to_int_or_none(raw.get("local_player_slot"))
    if local_slot is not None and 0 <= local_slot < len(players):
        item = players[local_slot]
        if isinstance(item, dict):
            hp = _to_int_or_none(item.get("hp"))
            if hp is not None:
                return hp
    for item in players:
        if isinstance(item, dict) and bool(item.get("is_local")):
            hp = _to_int_or_none(item.get("hp"))
            if hp is not None:
                return hp

    return None


def _to_int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def retrain_dual_models(
    repo_root: Path,
    trace_path: Path,
    combat_model_out: str,
    noncombat_model_out: str,
    epochs: int,
    combat_hidden1: int,
    combat_hidden2: int,
    noncombat_hidden1: int,
    noncombat_hidden2: int,
    replay_recent_segments: int,
    replay_new_ratio: float,
    replay_max_train_samples: int,
) -> bool:
    cmd = _build_dual_train_cmd(
        repo_root=repo_root,
        trace_path=trace_path,
        combat_model_out=combat_model_out,
        noncombat_model_out=noncombat_model_out,
        epochs=max(1, int(epochs)),
        combat_hidden1=combat_hidden1,
        combat_hidden2=combat_hidden2,
        noncombat_hidden1=noncombat_hidden1,
        noncombat_hidden2=noncombat_hidden2,
        replay_recent_segments=replay_recent_segments,
        replay_new_ratio=replay_new_ratio,
        replay_max_train_samples=replay_max_train_samples,
        mode="dual",
    )
    print("[dual-rl] retrain cmd:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(repo_root))
    return result.returncode == 0


def retrain_combat_model(
    repo_root: Path,
    trace_path: Path,
    combat_model_out: str,
    noncombat_model_out: str,
    epochs: int,
    combat_hidden1: int,
    combat_hidden2: int,
    noncombat_hidden1: int,
    noncombat_hidden2: int,
    replay_recent_segments: int,
    replay_new_ratio: float,
    replay_max_train_samples: int,
) -> bool:
    cmd = _build_dual_train_cmd(
        repo_root=repo_root,
        trace_path=trace_path,
        combat_model_out=combat_model_out,
        noncombat_model_out=noncombat_model_out,
        epochs=max(1, int(epochs)),
        combat_hidden1=combat_hidden1,
        combat_hidden2=combat_hidden2,
        noncombat_hidden1=noncombat_hidden1,
        noncombat_hidden2=noncombat_hidden2,
        replay_recent_segments=replay_recent_segments,
        replay_new_ratio=replay_new_ratio,
        replay_max_train_samples=replay_max_train_samples,
        mode="combat_only",
    )
    print("[dual-rl] combat-train cmd:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(repo_root))
    return result.returncode == 0


def retrain_noncombat_model(
    repo_root: Path,
    trace_path: Path,
    combat_model_out: str,
    noncombat_model_out: str,
    epochs: int,
    combat_hidden1: int,
    combat_hidden2: int,
    noncombat_hidden1: int,
    noncombat_hidden2: int,
    replay_recent_segments: int,
    replay_new_ratio: float,
    replay_max_train_samples: int,
) -> bool:
    cmd = _build_dual_train_cmd(
        repo_root=repo_root,
        trace_path=trace_path,
        combat_model_out=combat_model_out,
        noncombat_model_out=noncombat_model_out,
        epochs=max(1, int(epochs)),
        combat_hidden1=combat_hidden1,
        combat_hidden2=combat_hidden2,
        noncombat_hidden1=noncombat_hidden1,
        noncombat_hidden2=noncombat_hidden2,
        replay_recent_segments=replay_recent_segments,
        replay_new_ratio=replay_new_ratio,
        replay_max_train_samples=replay_max_train_samples,
        mode="noncombat_only",
    )
    print("[dual-rl] noncombat-train cmd:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(repo_root))
    return result.returncode == 0


def _build_dual_train_cmd(
    repo_root: Path,
    trace_path: Path,
    combat_model_out: str,
    noncombat_model_out: str,
    epochs: int,
    combat_hidden1: int,
    combat_hidden2: int,
    noncombat_hidden1: int,
    noncombat_hidden2: int,
    replay_recent_segments: int,
    replay_new_ratio: float,
    replay_max_train_samples: int,
    mode: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "train_dual_rl_models.py"),
        "--trace",
        str(trace_path),
        "--combat-out",
        combat_model_out,
        "--noncombat-out",
        noncombat_model_out,
        "--epochs",
        str(max(1, int(epochs))),
        "--combat-hidden1",
        str(max(64, int(combat_hidden1))),
        "--combat-hidden2",
        str(max(32, int(combat_hidden2))),
        "--noncombat-hidden1",
        str(max(64, int(noncombat_hidden1))),
        "--noncombat-hidden2",
        str(max(32, int(noncombat_hidden2))),
        "--replay-recent-segments",
        str(max(0, int(replay_recent_segments))),
        "--replay-new-ratio",
        str(max(0.0, min(1.0, float(replay_new_ratio)))),
        "--replay-max-train-samples",
        str(max(0, int(replay_max_train_samples))),
    ]
    if mode == "combat_only":
        cmd.append("--train-combat-only")
    elif mode == "noncombat_only":
        cmd.append("--train-noncombat-only")
    return cmd


def report_model_complexity(combat_model_path: str, noncombat_model_path: str) -> None:
    _report_one_model("combat", combat_model_path)
    _report_one_model("noncombat", noncombat_model_path)


def _report_one_model(tag: str, model_path: str) -> None:
    path = Path(model_path)
    if not path.exists():
        print(f"[model] {tag}: missing ({path})")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[model] {tag}: failed to read ({exc})")
        return
    if not isinstance(payload, dict):
        print(f"[model] {tag}: invalid model payload")
        return

    if all(key in payload for key in ["w1", "b1", "w2", "b2", "w3", "b3"]):
        w1 = payload.get("w1") if isinstance(payload.get("w1"), list) else []
        w2 = payload.get("w2") if isinstance(payload.get("w2"), list) else []
        w3 = payload.get("w3") if isinstance(payload.get("w3"), list) else []
        b1 = payload.get("b1") if isinstance(payload.get("b1"), list) else []
        b2 = payload.get("b2") if isinstance(payload.get("b2"), list) else []
        b3 = payload.get("b3") if isinstance(payload.get("b3"), list) else []

        def _size(matrix: list[object]) -> int:
            if not matrix:
                return 0
            if isinstance(matrix[0], list):
                return len(matrix) * len(matrix[0])
            return len(matrix)

        params = _size(w1) + _size(w2) + _size(w3) + len(b1) + len(b2) + len(b3)
        print(
            f"[model] {tag}: nonlinear_mlp input={payload.get('input_dim')} "
            f"h1={payload.get('hidden1')} h2={payload.get('hidden2')} params={params}"
        )
        return

    if str(payload.get("model_type") or "") == "noncombat_transformer_value":
        state_dict = payload.get("state_dict") if isinstance(payload.get("state_dict"), dict) else {}
        total = 0
        for value in state_dict.values():
            if isinstance(value, list):
                total += _nested_list_size(value)
        print(
            f"[model] {tag}: transformer token_buckets={payload.get('token_buckets')} "
            f"seq={payload.get('token_seq_len')} d_model={payload.get('d_model')} "
            f"layers={payload.get('num_layers')} params={total}"
        )
        return

    print(f"[model] {tag}: unknown model format")


def _nested_list_size(value: list[object]) -> int:
    if not value:
        return 0
    if isinstance(value[0], list):
        return sum(_nested_list_size(item) for item in value if isinstance(item, list))
    return len(value)


def wait_state_change(reader: McpApiReader, before: GameStateSnapshot, timeout_ms: int) -> GameStateSnapshot:
    end_at = time.time() + (max(200, timeout_ms) / 1000.0)
    before_sig = state_signature(before)
    latest = before
    while time.time() < end_at:
        latest = reader.read_state()
        if state_signature(latest) != before_sig:
            return latest
        time.sleep(0.2)
    return latest


def state_signature(state: GameStateSnapshot) -> tuple[object, ...]:
    enemy_sig = tuple((enemy.enemy_id, enemy.hp, enemy.block) for enemy in state.enemies)
    hand_sig = tuple(state.player.hand)
    return (
        state.state_type,
        state.in_combat,
        state.in_event,
        state.turn,
        state.player.hp,
        state.player.block,
        state.player.energy,
        hand_sig,
        enemy_sig,
        _screen_state_signature(state.raw_state),
    )


def _screen_state_signature(raw_state: dict[str, object]) -> tuple[object, ...]:
    if not isinstance(raw_state, dict):
        return ("no_raw",)
    state_type = str(raw_state.get("state_type") or "")

    if state_type == "card_select":
        section = raw_state.get("card_select") if isinstance(raw_state.get("card_select"), dict) else {}
        cards = section.get("cards") if isinstance(section.get("cards"), list) else []
        card_sig: list[tuple[object, object]] = []
        for card in cards:
            if isinstance(card, dict):
                card_sig.append((card.get("index"), card.get("id") or card.get("name")))
        return (
            state_type,
            section.get("screen_type"),
            section.get("prompt"),
            bool(section.get("preview_showing", False)),
            bool(section.get("can_confirm", False)),
            bool(section.get("can_cancel", False)),
            tuple(card_sig),
        )

    if state_type == "card_reward":
        section = raw_state.get("card_reward") if isinstance(raw_state.get("card_reward"), dict) else {}
        cards = section.get("cards") if isinstance(section.get("cards"), list) else []
        return (
            state_type,
            bool(section.get("can_skip", False)),
            tuple((card.get("index"), card.get("id") or card.get("name")) for card in cards if isinstance(card, dict)),
        )

    if state_type == "event":
        section = raw_state.get("event") if isinstance(raw_state.get("event"), dict) else {}
        options = section.get("options") if isinstance(section.get("options"), list) else []
        return (
            state_type,
            bool(section.get("in_dialogue", False)),
            tuple(
                (opt.get("index"), bool(opt.get("is_locked", False)), bool(opt.get("was_chosen", False)))
                for opt in options
                if isinstance(opt, dict)
            ),
        )

    if state_type == "combat_rewards":
        section = raw_state.get("rewards") if isinstance(raw_state.get("rewards"), dict) else {}
        items = section.get("items") if isinstance(section.get("items"), list) else []
        item_sig = []
        for item in items:
            if isinstance(item, dict):
                item_sig.append((item.get("index"), item.get("type"), item.get("description")))
        return (
            state_type,
            bool(section.get("can_proceed", False)),
            tuple(item_sig),
        )

    if state_type == "shop":
        section = raw_state.get("shop") if isinstance(raw_state.get("shop"), dict) else {}
        player = section.get("player") if isinstance(section.get("player"), dict) else {}
        items = section.get("items") if isinstance(section.get("items"), list) else []
        item_sig = []
        for item in items:
            if isinstance(item, dict):
                item_sig.append(
                    (
                        item.get("index"),
                        item.get("category"),
                        bool(item.get("can_afford", False)),
                        item.get("cost"),
                    )
                )
        return (
            state_type,
            bool(section.get("can_proceed", False)),
            _to_int_or_none(player.get("gold")),
            tuple(item_sig),
        )

    return (state_type,)


def _combat_rewards_key_from_state(raw_state: object) -> str:
    if not isinstance(raw_state, dict):
        return "combat_rewards|invalid"
    rewards = raw_state.get("rewards") if isinstance(raw_state.get("rewards"), dict) else {}
    return _combat_rewards_key(rewards)


def _combat_rewards_key(rewards: dict[str, object]) -> str:
    items = rewards.get("items") if isinstance(rewards.get("items"), list) else []
    can_proceed = bool(rewards.get("can_proceed", False))
    player = rewards.get("player") if isinstance(rewards.get("player"), dict) else {}
    open_slots = _to_int_or_none(player.get("open_potion_slots"))

    item_sig: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        idx = _to_int_or_none(item.get("index"))
        typ = str(item.get("type") or "")
        potion_id = str(item.get("potion_id") or "")
        desc = str(item.get("description") or "")
        item_sig.append(f"{idx}|{typ}|{potion_id}|{desc}")
    item_sig.sort()
    return f"combat_rewards|can_proceed:{int(can_proceed)}|open_slots:{open_slots}|items:{'||'.join(item_sig)}"


def extract_run_progress(state: GameStateSnapshot) -> tuple[int, int]:
    raw = state.raw_state if isinstance(state.raw_state, dict) else {}
    run = raw.get("run") if isinstance(raw.get("run"), dict) else {}
    return to_int(run.get("act")), to_int(run.get("floor"))


def write_episode_metrics(path: Path, stats: EpisodeStats) -> None:
    payload = {
        "timestamp_ms": int(time.time() * 1000),
        "episode": stats.episode,
        "duration_sec": round(max(0, stats.end_ms - stats.start_ms) / 1000.0, 3),
        "max_act": stats.max_act,
        "max_floor": stats.max_floor,
        "hp_start": stats.hp_start,
        "hp_end": stats.hp_end,
        "hp_loss": stats.hp_loss,
        "died": stats.died,
    }
    append_jsonl(path, payload)


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def ensure_game_running(process_name: str, game_exe: str | None) -> None:
    if find_process(process_name):
        return
    if not game_exe:
        raise RuntimeError(
            f"Game process '{process_name}' not found. Start game manually or pass --game-exe."
        )

    exe_path = Path(game_exe)
    if not exe_path.exists():
        raise FileNotFoundError(f"game exe not found: {exe_path}")

    subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))
    print(f"[dual-rl] launched game: {exe_path}")

    deadline = time.time() + 60.0
    while time.time() < deadline:
        if find_process(process_name):
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for process '{process_name}'")


def find_process(process_name: str):
    lowered = process_name.lower()
    for proc in psutil.process_iter(attrs=["name"]):
        name = str(proc.info.get("name") or "").lower()
        if name == lowered:
            return proc
    return None


def click_fixed_defeat_exit_button() -> None:
    user32 = ctypes.windll.user32
    user32.SetCursorPos(int(DEFEAT_EXIT_X), int(DEFEAT_EXIT_Y))
    for idx in range(DEFEAT_EXIT_CLICKS):
        user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down
        user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up
        if idx < DEFEAT_EXIT_CLICKS - 1:
            time.sleep(DEFEAT_EXIT_CLICK_INTERVAL_MS / 1000.0)


def to_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return 0


if __name__ == "__main__":
    main()

