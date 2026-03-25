from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from sos2_interface.actions.noop_executor import ActionExecutor
from sos2_interface.contracts.action import ActionCommand, ActionResult
from sos2_interface.contracts.state import GameStateSnapshot
from sos2_interface.policy.action_value_model import load_action_value_model
from sos2_interface.policy.segment_simulator import BranchScore, DeterministicSegmentSimulator
from sos2_interface.policy.trace_utils import compact_state, summarize_transition
from sos2_interface.readers.base import GameReader


@dataclass
class PlanCycleResult:
    iteration: int
    planned_actions: int
    executed_actions: int
    boundary_reason: str
    warnings: list[str]


class RandomBoundaryDetector:
    """Triggers re-observation only on boundary events, not every action."""

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
    """Enumerates deterministic branches and returns the highest-value action segment."""

    def __init__(
        self,
        max_segment_actions: int = 4,
        max_branches: int = 3000,
        value_model_path: str | None = None,
        value_model_weight: float = 0.35,
    ) -> None:
        self._max_segment_actions = max(1, max_segment_actions)
        self._max_branches = max(64, max_branches)
        self._value_model = load_action_value_model(value_model_path)
        self._value_model_weight = max(0.0, float(value_model_weight))

    def plan(self, state: GameStateSnapshot) -> list[ActionCommand]:
        simulator = DeterministicSegmentSimulator(
            state,
            action_value_model=self._value_model,
            model_weight=self._value_model_weight,
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

                reached_depth = depth + 1 >= self._max_segment_actions
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
    ) -> None:
        self._reader = reader
        self._executor = executor
        self._planner = planner or SegmentPlanner()
        self._detector = boundary_detector or RandomBoundaryDetector()
        self._trace_file = Path(trace_file)
        self._trace_file.parent.mkdir(parents=True, exist_ok=True)
        self._capture_action_trace = bool(capture_action_trace)
        self._include_raw_state_in_action_trace = bool(include_raw_state_in_action_trace)
        self._action_trace_file = Path(action_trace_file) if action_trace_file else None
        if self._capture_action_trace and self._action_trace_file is not None:
            self._action_trace_file.parent.mkdir(parents=True, exist_ok=True)
        self._iteration = 0

    def run_once(self) -> PlanCycleResult:
        self._iteration += 1
        observed = self._reader.read_state()
        actions = self._planner.plan(observed)

        executed_actions = 0
        boundary_reason = "no actions produced"
        action_results: list[ActionResult] = []
        current_state = observed
        action_timeline: list[dict[str, object]] = []

        for action in actions:
            before_state = current_state
            result = self._executor.execute(action)
            action_results.append(result)
            executed_actions += 1

            after_state = before_state
            if self._capture_action_trace:
                after_state = self._reader.read_state()
                current_state = after_state
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
        should_replan, state_reason = self._detector.state_requires_replan(observed, refreshed)
        if should_replan:
            boundary_reason = state_reason

        warnings = list(dict.fromkeys([*observed.warnings, *refreshed.warnings]))
        cycle = PlanCycleResult(
            iteration=self._iteration,
            planned_actions=len(actions),
            executed_actions=executed_actions,
            boundary_reason=boundary_reason,
            warnings=warnings,
        )
        self._append_trace(observed, actions, action_results, refreshed, cycle, action_timeline)
        return cycle

    def run_forever(self, interval_ms: int = 300, max_iterations: int | None = None) -> None:
        interval = max(50, interval_ms) / 1000.0
        while True:
            cycle = self.run_once()
            print(
                f"iteration={cycle.iteration} planned={cycle.planned_actions} "
                f"executed={cycle.executed_actions} boundary='{cycle.boundary_reason}'"
            )

            if max_iterations is not None and cycle.iteration >= max_iterations:
                break
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

