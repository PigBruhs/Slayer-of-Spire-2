from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from sos2_interface.actions.noop_executor import ActionExecutor
from sos2_interface.contracts.action import ActionCommand, ActionResult
from sos2_interface.contracts.state import GameStateSnapshot
from sos2_interface.policy.action_value_model import load_action_value_model
from sos2_interface.policy.combat_policy_model import load_combat_policy_model
from sos2_interface.policy.noncombat_policy_model import load_noncombat_policy_model
from sos2_interface.policy.segment_simulator import BranchFactorWeights, BranchScore, DeterministicSegmentSimulator
from sos2_interface.policy.trace_utils import compact_state, summarize_transition
from sos2_interface.readers.base import GameReader


MIN_OPERATION_DELAY_MS = 500
MIN_CYCLE_INTERVAL_MS = 500
STATE_POLL_INTERVAL_MS = 250
END_TURN_STABLE_POLLS = 4
SOFT_LOOP_SAME_SIGNATURE_LIMIT = 8
SOFT_LOOP_WINDOW_SIZE = 20
SOFT_LOOP_WINDOW_HIT_LIMIT = 12


@dataclass
class PlanCycleResult:
    iteration: int
    planned_actions: int
    executed_actions: int
    boundary_reason: str
    warnings: list[str]
    decision_summary: str


class RandomBoundaryDetector:
    def action_requires_reobserve(self, action: ActionCommand) -> tuple[bool, str]:
        if bool(action.metadata.get("boundary")):
            reason = str(action.metadata.get("boundary_reason") or "planner boundary")
            return True, reason
        if action.action_type in {"event_choose", "map_choose", "end_turn"}:
            return True, f"action boundary: {action.action_type}"
        if bool(action.metadata.get("random")):
            return True, "policy-marked random action"
        return False, "continue deterministic segment"

    def state_requires_replan(self, before: GameStateSnapshot, after: GameStateSnapshot) -> tuple[bool, str]:
        if after.warnings:
            return True, "state warnings present"
        if before.in_event != after.in_event:
            return True, "scene switched between event and non-event"
        if before.in_combat != after.in_combat:
            return True, "combat state changed"
        if after.player.hp <= 0:
            return True, "player hp depleted"
        return False, "deterministic segment can continue"


class SegmentPlanner:
    def __init__(
        self,
        max_segment_actions: int = 0,
        max_branches: int = 3000,
        value_model_path: str | None = None,
        value_model_weight: float = 0.35,
        combat_model_path: str | None = None,
        combat_model_weight: float = 0.8,
        noncombat_model_path: str | None = None,
        noncombat_model_weight: float = 1.0,
        enable_noncombat_policy: bool = True,
        branch_weight_path: str | None = None,
    ) -> None:
        self._max_segment_actions = int(max_segment_actions)
        self._max_branches = max(64, max_branches)
        self._value_model = load_action_value_model(value_model_path)
        self._value_model_weight = max(0.0, float(value_model_weight))
        self._combat_model = load_combat_policy_model(combat_model_path)
        self._combat_model_weight = max(0.0, float(combat_model_weight))
        self._noncombat_model = load_noncombat_policy_model(noncombat_model_path)
        self._noncombat_model_weight = max(0.0, float(noncombat_model_weight))
        self._enable_noncombat_policy = bool(enable_noncombat_policy)
        self._branch_weights = BranchFactorWeights.from_json(branch_weight_path)

    @property
    def supports_noncombat(self) -> bool:
        return self._enable_noncombat_policy

    def plan(self, state: GameStateSnapshot) -> list[ActionCommand]:
        scripted = self._plan_screen_action(state)
        if scripted is not None:
            return [scripted]

        if not state.in_combat:
            return [ActionCommand(action_type="noop", metadata={"boundary": True, "boundary_reason": "noncombat_no_candidate"})]

        simulator = DeterministicSegmentSimulator(
            state,
            action_value_model=self._value_model,
            model_weight=self._value_model_weight,
            combat_policy_model=self._combat_model,
            combat_model_weight=self._combat_model_weight,
            branch_weights=self._branch_weights,
        )
        explored = {"count": 0}
        best_actions: list[ActionCommand] = []
        best_score: BranchScore | None = None

        def consider(actions: list[ActionCommand], score: BranchScore) -> None:
            nonlocal best_actions, best_score
            if best_score is None or score.total > best_score.total:
                best_actions = [action.model_copy(deep=True) for action in actions]
                best_score = score

        def dfs(current: DeterministicSegmentSimulator, prefix: list[ActionCommand], depth: int) -> None:
            if explored["count"] >= self._max_branches:
                consider(prefix, current.evaluate_branch(ended_turn=False))
                return

            candidates = current.list_candidate_actions()
            if not candidates:
                consider(prefix, current.evaluate_branch(ended_turn=False))
                return

            for candidate in candidates:
                if explored["count"] >= self._max_branches:
                    return
                explored["count"] += 1

                child = current.clone()
                step = child.apply(candidate)
                path = [*prefix, candidate]

                reached_depth = depth + 1 >= self._resolve_depth_limit(state)
                ended_turn = candidate.action_type == "end_turn"
                if reached_depth or step.boundary or not step.applied:
                    if step.boundary or not step.applied:
                        path[-1].metadata["boundary"] = True
                        path[-1].metadata.setdefault("boundary_reason", step.reason)
                    consider(path, child.evaluate_branch(ended_turn=ended_turn))
                else:
                    dfs(child, path, depth + 1)

        dfs(simulator, [], 0)
        if not best_actions:
            return [ActionCommand(action_type="noop", metadata={"boundary": True, "boundary_reason": "no branch produced"})]

        if best_score is not None and best_actions:
            best_actions[0].metadata.update(
                {
                    "branch_score": round(best_score.total, 3),
                    "branch_damage": round(best_score.damage_score, 3),
                    "branch_block": round(best_score.defense_score, 3),
                    "branch_other": round(best_score.utility_score, 3),
                    "branch_incoming": best_score.projected_incoming_damage,
                    "branch_explored": explored["count"],
                }
            )
        return best_actions

    def _resolve_depth_limit(self, state: GameStateSnapshot) -> int:
        if self._max_segment_actions > 0:
            return self._max_segment_actions
        hand_based = max(6, len(state.player.hand) * 2)
        energy_based = max(3, state.player.energy * 3)
        return max(hand_based, energy_based)

    def _plan_screen_action(self, state: GameStateSnapshot) -> ActionCommand | None:
        state_type = (state.state_type or "").strip().lower()
        raw = state.raw_state if isinstance(state.raw_state, dict) else {}

        if state_type in {"monster", "elite", "boss", ""} and state.in_combat:
            return None

        if state_type == "hand_select":
            hand_select = raw.get("hand_select") if isinstance(raw.get("hand_select"), dict) else {}
            selectable = hand_select.get("selectable_cards") if isinstance(hand_select, dict) else []
            if isinstance(selectable, list) and selectable:
                first = selectable[0]
                if isinstance(first, dict):
                    index = first.get("index")
                    if isinstance(index, int):
                        return ActionCommand(action_type="combat_select_card", option_index=index, metadata={"boundary": True})
            if bool(hand_select.get("confirm_enabled", False)):
                return ActionCommand(action_type="combat_confirm_selection", metadata={"boundary": True})
            return ActionCommand(action_type="noop", metadata={"boundary": True, "boundary_reason": "hand_select no selectable card"})

        if not self._enable_noncombat_policy:
            return ActionCommand(
                action_type="noop",
                metadata={"boundary": True, "boundary_reason": f"manual_non_combat:{state_type or 'unknown'}"},
            )

        candidates = self._candidate_screen_actions(state, raw, state_type)
        if not candidates:
            return ActionCommand(
                action_type="noop",
                metadata={"boundary": True, "boundary_reason": f"noncombat_no_candidate:{state_type or 'unknown'}"},
            )

        if self._noncombat_model is None or self._noncombat_model_weight <= 0:
            return candidates[0]

        best_idx = 0
        best_score = float("-inf")
        for idx, candidate in enumerate(candidates):
            score = self._noncombat_model.score(state, candidate) * self._noncombat_model_weight
            if score > best_score:
                best_score = score
                best_idx = idx

        chosen = candidates[best_idx]
        chosen.metadata["noncombat_value"] = round(best_score, 4)
        chosen.metadata["noncombat_candidates"] = len(candidates)
        chosen.metadata["boundary"] = True
        chosen.metadata.setdefault("boundary_reason", f"noncombat_action:{chosen.action_type}")
        return chosen

    def _candidate_screen_actions(
        self,
        state: GameStateSnapshot,
        raw: dict[str, object],
        state_type: str,
    ) -> list[ActionCommand]:
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
            card_reward = raw.get("card_reward") if isinstance(raw.get("card_reward"), dict) else {}
            cards = card_reward.get("cards") if isinstance(card_reward.get("cards"), list) else []
            for card in cards:
                if isinstance(card, dict) and isinstance(card.get("index"), int):
                    actions.append(ActionCommand(action_type="select_card_reward", option_index=int(card["index"])))
            if bool(card_reward.get("can_skip")):
                actions.append(ActionCommand(action_type="skip_card_reward"))

        elif state_type == "event":
            event = raw.get("event") if isinstance(raw.get("event"), dict) else {}
            if bool(event.get("in_dialogue")):
                actions.append(ActionCommand(action_type="advance_dialogue"))
            options = event.get("options") if isinstance(event.get("options"), list) else []
            for item in options:
                if not isinstance(item, dict):
                    continue
                if bool(item.get("is_locked")):
                    continue
                idx = item.get("index")
                if isinstance(idx, int):
                    actions.append(ActionCommand(action_type="event_choose", option_index=idx))

        elif state_type == "map":
            map_state = raw.get("map") if isinstance(raw.get("map"), dict) else {}
            options = map_state.get("next_options") if isinstance(map_state.get("next_options"), list) else []
            for node in options:
                if isinstance(node, dict) and isinstance(node.get("index"), int):
                    actions.append(ActionCommand(action_type="map_choose", option_index=int(node["index"])))

        elif state_type == "rest_site":
            rest = raw.get("rest_site") if isinstance(raw.get("rest_site"), dict) else {}
            options = rest.get("options") if isinstance(rest.get("options"), list) else []
            for item in options:
                if not isinstance(item, dict):
                    continue
                if not bool(item.get("is_enabled")):
                    continue
                idx = item.get("index")
                if isinstance(idx, int):
                    actions.append(ActionCommand(action_type="choose_rest_option", option_index=idx))
            if bool(rest.get("can_proceed")):
                actions.append(ActionCommand(action_type="proceed"))

        elif state_type == "shop":
            shop = raw.get("shop") if isinstance(raw.get("shop"), dict) else {}
            items = shop.get("items") if isinstance(shop.get("items"), list) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                if not bool(item.get("is_stocked", False)):
                    continue
                if not bool(item.get("can_afford", False)):
                    continue
                idx = item.get("index")
                if isinstance(idx, int):
                    actions.append(ActionCommand(action_type="shop_purchase", option_index=idx))
            if bool(shop.get("can_proceed")):
                actions.append(ActionCommand(action_type="proceed"))

        elif state_type == "card_select":
            card_select = raw.get("card_select") if isinstance(raw.get("card_select"), dict) else {}
            cards = card_select.get("cards") if isinstance(card_select.get("cards"), list) else []
            for card in cards:
                if isinstance(card, dict) and isinstance(card.get("index"), int):
                    actions.append(ActionCommand(action_type="select_card", option_index=int(card["index"])))
            if bool(card_select.get("can_confirm")):
                actions.append(ActionCommand(action_type="confirm_selection"))
            if bool(card_select.get("can_cancel")):
                actions.append(ActionCommand(action_type="cancel_selection"))

        elif state_type == "relic_select":
            relic = raw.get("relic_select") if isinstance(raw.get("relic_select"), dict) else {}
            relics = relic.get("relics") if isinstance(relic.get("relics"), list) else []
            for item in relics:
                if isinstance(item, dict) and isinstance(item.get("index"), int):
                    actions.append(ActionCommand(action_type="select_relic", option_index=int(item["index"])))
            if bool(relic.get("can_skip")):
                actions.append(ActionCommand(action_type="skip_relic_selection"))

        elif state_type == "treasure":
            treasure = raw.get("treasure") if isinstance(raw.get("treasure"), dict) else {}
            relics = treasure.get("relics") if isinstance(treasure.get("relics"), list) else []
            for item in relics:
                if isinstance(item, dict) and isinstance(item.get("index"), int):
                    actions.append(ActionCommand(action_type="claim_treasure_relic", option_index=int(item["index"])))
            if bool(treasure.get("can_proceed")):
                actions.append(ActionCommand(action_type="proceed"))

        if not actions and not state.in_combat:
            actions.append(ActionCommand(action_type="noop", metadata={"boundary": True, "boundary_reason": f"no_actions:{state_type or 'unknown'}"}))

        for action in actions:
            action.metadata.setdefault("boundary", True)
            action.metadata.setdefault("boundary_reason", f"screen:{state_type or 'unknown'}")
        return actions


class PlannerLoop:
    def __init__(
        self,
        reader: GameReader,
        executor: ActionExecutor,
        planner: SegmentPlanner | None = None,
        boundary_detector: RandomBoundaryDetector | None = None,
        trace_file: str = "runtime/planner_cycles.jsonl",
        action_trace_file: str | None = "runtime/planner_action_trace.jsonl",
        capture_action_trace: bool = False,
        include_raw_state_in_action_trace: bool = False,
        min_play_card_interval_ms: int = 0,
        wait_for_play_phase_before_card: bool = False,
        combat_ready_timeout_ms: int = 1500,
        wait_action_completion: bool = True,
        action_completion_timeout_ms: int = 4000,
        action_completion_poll_ms: int = 20,
        combat_only: bool = True,
    ) -> None:
        self._reader = reader
        self._executor = executor
        self._planner = planner or SegmentPlanner()
        self._detector = boundary_detector or RandomBoundaryDetector()
        self._trace_file = Path(trace_file)
        self._trace_file.parent.mkdir(parents=True, exist_ok=True)
        self._capture_action_trace = bool(capture_action_trace)
        self._include_raw_state_in_action_trace = bool(include_raw_state_in_action_trace)
        self._min_play_card_interval_ms = max(0, int(min_play_card_interval_ms))
        self._min_action_interval_ms = MIN_OPERATION_DELAY_MS
        self._min_cycle_interval_ms = MIN_CYCLE_INTERVAL_MS
        self._wait_for_play_phase_before_card = bool(wait_for_play_phase_before_card)
        self._combat_ready_timeout_ms = max(0, int(combat_ready_timeout_ms))
        self._wait_action_completion = bool(wait_action_completion)
        self._action_completion_timeout_ms = max(50, int(action_completion_timeout_ms))
        self._combat_only = bool(combat_only)
        # Fixed state polling cadence for stable updates.
        self._action_completion_poll_ms = STATE_POLL_INTERVAL_MS
        self._last_cycle_ms = 0
        self._last_action_ms = 0
        self._last_play_card_ms = 0
        self._last_end_turn_ms = 0
        self._action_trace_file = Path(action_trace_file) if action_trace_file else None
        if self._capture_action_trace and self._action_trace_file is not None:
            self._action_trace_file.parent.mkdir(parents=True, exist_ok=True)
        self._iteration = 0
        self._saw_act3_boss = False
        self._recent_signatures: deque[tuple[object, ...]] = deque(maxlen=SOFT_LOOP_WINDOW_SIZE)
        self._same_signature_streak = 0

    def run_once(self) -> PlanCycleResult:
        self._wait_min_cycle_interval()
        self._wait_post_end_turn_stable_state()
        self._last_cycle_ms = int(time.time() * 1000)
        self._iteration += 1
        observed = self._reader.read_state()
        self._update_episode_markers(observed)

        terminal_reason = self._detect_terminal_reason(observed)
        if terminal_reason is not None:
            cycle = PlanCycleResult(
                iteration=self._iteration,
                planned_actions=0,
                executed_actions=0,
                boundary_reason=terminal_reason,
                warnings=list(observed.warnings),
                decision_summary=f"none({terminal_reason})",
            )
            self._append_trace(observed, [], [], observed, cycle, [])
            return cycle

        if self._combat_only and not observed.in_combat:
            cycle = PlanCycleResult(
                iteration=self._iteration,
                planned_actions=0,
                executed_actions=0,
                boundary_reason=f"manual_non_combat_wait:{(observed.state_type or 'unknown')}",
                warnings=list(observed.warnings),
                decision_summary="none(non_combat)",
            )
            self._append_trace(observed, [], [], observed, cycle, [])
            return cycle

        actions = self._planner.plan(observed)

        executed_actions = 0
        boundary_reason = "no actions produced"
        action_results: list[ActionResult] = []
        current_state = observed
        action_timeline: list[dict[str, object]] = []

        for action in actions:
            before_state = current_state
            self._wait_min_action_interval()
            if action.action_type == "play_card":
                before_state = self._wait_until_play_card_ready(before_state)
                current_state = before_state

            result = self._executor.execute(action)
            action_results.append(result)
            executed_actions += 1
            self._last_action_ms = int(time.time() * 1000)

            if action.action_type == "play_card":
                self._last_play_card_ms = int(time.time() * 1000)
            if action.action_type == "end_turn" and result.accepted:
                self._last_end_turn_ms = int(time.time() * 1000)

            after_state = self._wait_for_action_completion(before_state, action, result)
            current_state = after_state

            if self._capture_action_trace:
                trace_entry = {
                    "iteration": self._iteration,
                    "timestamp_ms": int(time.time() * 1000),
                    "action": action.model_dump(),
                    "result": result.model_dump(),
                    "before": compact_state(before_state),
                    "after": compact_state(after_state),
                    "transition": summarize_transition(before_state, after_state),
                }
                if self._include_raw_state_in_action_trace:
                    trace_entry["before_full"] = before_state.model_dump()
                    trace_entry["after_full"] = after_state.model_dump()
                action_timeline.append(trace_entry)
                self._append_action_trace(trace_entry)

            should_reobserve, reason = self._detector.action_requires_reobserve(action)
            boundary_reason = reason
            if should_reobserve:
                break

        refreshed = current_state if self._capture_action_trace else self._reader.read_state()
        self._update_episode_markers(refreshed)
        should_replan, state_reason = self._detector.state_requires_replan(observed, refreshed)
        if should_replan:
            boundary_reason = state_reason

        soft_loop_reason = self._update_soft_loop_state(observed, refreshed, executed_actions)
        if soft_loop_reason is not None:
            boundary_reason = soft_loop_reason

        warnings = list(dict.fromkeys([*observed.warnings, *refreshed.warnings]))
        cycle = PlanCycleResult(
            iteration=self._iteration,
            planned_actions=len(actions),
            executed_actions=executed_actions,
            boundary_reason=boundary_reason,
            warnings=warnings,
            decision_summary=_summarize_actions(actions, executed_actions),
        )
        self._append_trace(observed, actions, action_results, refreshed, cycle, action_timeline)
        return cycle

    def run_forever(self, interval_ms: int = 300, max_iterations: int | None = None) -> None:
        interval = max(0, interval_ms) / 1000.0
        while True:
            cycle = self.run_once()
            print(
                f"iteration={cycle.iteration} planned={cycle.planned_actions} "
                f"executed={cycle.executed_actions} boundary='{cycle.boundary_reason}' decision='{cycle.decision_summary}'"
            )

            if cycle.boundary_reason in {
                "player hp depleted",
                "episode_terminal:player_hp_zero",
                "episode_terminal:act3_boss_defeated",
            }:
                break

            if cycle.boundary_reason.startswith("soft_loop_detected"):
                break

            if max_iterations is not None and cycle.iteration >= max_iterations:
                break
            if interval > 0:
                time.sleep(interval)

    def _append_trace(
        self,
        observed: GameStateSnapshot,
        actions: list[ActionCommand],
        action_results: list[ActionResult],
        refreshed: GameStateSnapshot,
        cycle: PlanCycleResult,
        action_timeline: list[dict[str, object]],
    ) -> None:
        payload = {
            "timestamp_ms": int(time.time() * 1000),
            "cycle": {
                "iteration": cycle.iteration,
                "planned_actions": cycle.planned_actions,
                "executed_actions": cycle.executed_actions,
                "boundary_reason": cycle.boundary_reason,
                "warnings": cycle.warnings,
                "decision_summary": cycle.decision_summary,
            },
            "observed": observed.model_dump(),
            "planned_actions": [action.model_dump() for action in actions],
            "action_results": [result.model_dump() for result in action_results],
            "action_timeline": action_timeline,
            "refreshed": refreshed.model_dump(),
        }
        with self._trace_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _append_action_trace(self, payload: dict[str, object]) -> None:
        if self._action_trace_file is None:
            return
        with self._action_trace_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _wait_until_play_card_ready(self, baseline: GameStateSnapshot) -> GameStateSnapshot:
        latest = baseline
        if self._min_play_card_interval_ms > 0 and self._last_play_card_ms > 0:
            elapsed = int(time.time() * 1000) - self._last_play_card_ms
            remaining = self._min_play_card_interval_ms - elapsed
            if remaining > 0:
                time.sleep(remaining / 1000.0)

        if not self._wait_for_play_phase_before_card:
            return latest

        deadline = time.time() + (self._combat_ready_timeout_ms / 1000.0)
        while time.time() < deadline:
            if self._is_ready_for_play_card(latest):
                return latest
            time.sleep(0.01)
            latest = self._reader.read_state()
        return latest

    def _wait_for_action_completion(self, before_state: GameStateSnapshot, action: ActionCommand, result: ActionResult) -> GameStateSnapshot:
        if action.action_type == "noop":
            return before_state
        if not result.accepted:
            return self._reader.read_state()
        if not self._wait_action_completion:
            return self._reader.read_state()

        before_signature = _state_signature(before_state)
        deadline = time.time() + (self._action_completion_timeout_ms / 1000.0)
        latest = before_state
        while time.time() < deadline:
            latest = self._reader.read_state()
            if _state_signature(latest) != before_signature:
                return latest
            time.sleep(self._action_completion_poll_ms / 1000.0)
        return latest

    def _wait_min_action_interval(self) -> None:
        if self._last_action_ms <= 0:
            return
        elapsed = int(time.time() * 1000) - self._last_action_ms
        remaining = self._min_action_interval_ms - elapsed
        if remaining > 0:
            time.sleep(remaining / 1000.0)

    def _wait_min_cycle_interval(self) -> None:
        if self._last_cycle_ms <= 0:
            return
        elapsed = int(time.time() * 1000) - self._last_cycle_ms
        remaining = self._min_cycle_interval_ms - elapsed
        if remaining > 0:
            time.sleep(remaining / 1000.0)

    def _wait_post_end_turn_stable_state(self) -> None:
        if self._last_end_turn_ms <= 0:
            return

        latest = self._reader.read_state()
        last_sig = _state_signature(latest)
        stable_polls = 0
        deadline = time.time() + 20.0

        while stable_polls < END_TURN_STABLE_POLLS and time.time() < deadline:
            time.sleep(STATE_POLL_INTERVAL_MS / 1000.0)
            latest = self._reader.read_state()
            current_sig = _state_signature(latest)
            if current_sig == last_sig:
                stable_polls += 1
            else:
                stable_polls = 0
                last_sig = current_sig

        self._last_end_turn_ms = 0

    def _update_episode_markers(self, state: GameStateSnapshot) -> None:
        raw = state.raw_state if isinstance(state.raw_state, dict) else {}
        run = raw.get("run") if isinstance(raw.get("run"), dict) else {}
        act = _to_int_or_none(run.get("act")) or 0
        if str(state.state_type or "").strip().lower() == "boss" and act >= 3:
            self._saw_act3_boss = True

    def _detect_terminal_reason(self, state: GameStateSnapshot) -> str | None:
        if state.player.hp <= 0:
            return "episode_terminal:player_hp_zero"

        explicit_hp = _extract_explicit_player_hp(state.raw_state if isinstance(state.raw_state, dict) else {})
        if explicit_hp is not None and explicit_hp <= 0:
            return "episode_terminal:player_hp_zero"

        if _is_act3_boss_defeated(state, self._saw_act3_boss):
            return "episode_terminal:act3_boss_defeated"
        return None

    def _update_soft_loop_state(
        self,
        observed: GameStateSnapshot,
        refreshed: GameStateSnapshot,
        executed_actions: int,
    ) -> str | None:
        if refreshed.in_combat:
            self._same_signature_streak = 0
            self._recent_signatures.clear()
            return None

        refreshed_sig = _state_signature(refreshed)
        observed_sig = _state_signature(observed)
        if self._recent_signatures and self._recent_signatures[-1] == refreshed_sig:
            self._same_signature_streak += 1
        else:
            self._same_signature_streak = 1
        self._recent_signatures.append(refreshed_sig)

        window_hits = sum(1 for item in self._recent_signatures if item == refreshed_sig)
        no_progress = observed_sig == refreshed_sig
        if (
            self._same_signature_streak >= SOFT_LOOP_SAME_SIGNATURE_LIMIT
            or window_hits >= SOFT_LOOP_WINDOW_HIT_LIMIT
            or (no_progress and executed_actions > 0 and self._same_signature_streak >= 4)
        ):
            state_name = str(refreshed.state_type or "unknown")
            return (
                "soft_loop_detected:"
                f"state={state_name}:streak={self._same_signature_streak}:hits={window_hits}"
            )
        return None

    @staticmethod
    def _is_ready_for_play_card(state: GameStateSnapshot) -> bool:
        if not state.in_combat:
            return True
        raw = state.raw_state if isinstance(state.raw_state, dict) else {}
        battle = raw.get("battle") if isinstance(raw.get("battle"), dict) else None
        if not isinstance(battle, dict):
            return True

        is_play_phase = battle.get("is_play_phase")
        if isinstance(is_play_phase, bool) and not is_play_phase:
            return False
        turn = battle.get("turn")
        if isinstance(turn, str) and turn.strip().lower() not in {"", "player"}:
            return False
        player_actions_disabled = battle.get("player_actions_disabled")
        if isinstance(player_actions_disabled, bool) and player_actions_disabled:
            return False
        return True


def _summarize_actions(actions: list[ActionCommand], executed_actions: int) -> str:
    if not actions:
        return "none"
    selected = actions[: max(1, executed_actions)] if executed_actions > 0 else actions[:1]
    parts: list[str] = []
    for action in selected:
        token = action.action_type
        if action.card_id:
            token += f"({action.card_id})"
        elif action.option_index is not None:
            token += f"[{action.option_index}]"
        parts.append(token)
    return " -> ".join(parts)


def _state_signature(state: GameStateSnapshot) -> tuple[object, ...]:
    enemy_sig = tuple((enemy.enemy_id, enemy.hp, enemy.block) for enemy in state.enemies)
    hand_sig = tuple(state.player.hand)
    warning_sig = tuple(state.warnings)
    return (
        state.state_type,
        state.in_combat,
        state.in_event,
        state.turn,
        state.player.hp,
        state.player.max_hp,
        state.player.block,
        state.player.energy,
        hand_sig,
        state.player.draw_pile_count,
        state.player.discard_pile_count,
        enemy_sig,
        warning_sig,
    )


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


def _extract_explicit_player_hp(raw: dict[str, object]) -> int | None:
    battle = raw.get("battle") if isinstance(raw.get("battle"), dict) else {}
    battle_player = battle.get("player") if isinstance(battle, dict) else {}
    hp = _to_int_or_none(battle_player.get("hp") if isinstance(battle_player, dict) else None)
    if hp is not None:
        return hp

    top_player = raw.get("player") if isinstance(raw.get("player"), dict) else {}
    hp = _to_int_or_none(top_player.get("hp") if isinstance(top_player, dict) else None)
    if hp is not None:
        return hp

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


def _is_act3_boss_defeated(state: GameStateSnapshot, saw_act3_boss: bool) -> bool:
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


