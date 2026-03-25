from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sos2_interface.contracts.action import ActionCommand
from sos2_interface.contracts.state import EnemyIntent, EnemyState, GameStateSnapshot
from sos2_interface.policy.action_value_model import ActionValueModel
from sos2_interface.policy.card_knowledge import (
    CardEffects,
    estimate_card_cost,
    estimate_card_effects,
    is_random_boundary_card,
    normalize_card_id,
)


@dataclass
class BranchScore:
    damage_score: float
    defense_score: float
    utility_score: float
    end_turn_score: float
    incoming_damage_penalty: float
    projected_incoming_damage: int
    total: float


@dataclass
class SimulationStepResult:
    applied: bool
    boundary: bool
    reason: str


class DeterministicSegmentSimulator:
    """Predict deterministic card segments and score candidate branches."""

    def __init__(self, snapshot: GameStateSnapshot, action_value_model: ActionValueModel | None = None, model_weight: float = 0.35) -> None:
        self._state = snapshot.model_copy(deep=True)
        self._state.warnings = []
        self._action_value_model = action_value_model
        self._model_weight = max(0.0, float(model_weight))
        self._damage_dealt = 0
        self._block_gained = 0
        self._other_effect_score = 0.0
        self._learned_value_bonus = 0.0

    @property
    def state(self) -> GameStateSnapshot:
        return self._state

    def clone(self) -> "DeterministicSegmentSimulator":
        copied = DeterministicSegmentSimulator(
            self._state.model_copy(deep=True),
            action_value_model=self._action_value_model,
            model_weight=self._model_weight,
        )
        copied._damage_dealt = self._damage_dealt
        copied._block_gained = self._block_gained
        copied._other_effect_score = self._other_effect_score
        copied._learned_value_bonus = self._learned_value_bonus
        return copied

    def list_candidate_actions(self) -> list[ActionCommand]:
        if self._state.in_event and self._state.event and self._state.event.options:
            return [ActionCommand(action_type="event_choose", option_index=0, metadata={"boundary": True})]

        if not self._state.in_combat:
            return [ActionCommand(action_type="noop", metadata={"boundary": True})]

        actions: list[ActionCommand] = [ActionCommand(action_type="end_turn", metadata={"boundary": True})]
        if self._state.player.energy <= 0:
            return actions

        seen_card_ids: set[str] = set()
        target_id = _first_living_enemy_id(self._state.enemies)
        for raw_card in self._state.player.hand:
            card_id = normalize_card_id(raw_card)
            if not card_id or card_id in seen_card_ids:
                continue
            seen_card_ids.add(card_id)

            cost = estimate_card_cost(card_id)
            if cost is None or cost > self._state.player.energy:
                continue

            actions.append(
                ActionCommand(
                    action_type="play_card",
                    card_id=card_id,
                    target_id=target_id,
                    metadata={"cost": cost, "random": is_random_boundary_card(card_id)},
                )
            )

        return actions

    def evaluate_branch(self, ended_turn: bool) -> BranchScore:
        incoming_damage = _estimate_incoming_damage(self._state.enemies)
        unblocked_damage = max(0, incoming_damage - self._state.player.block)

        damage_score = float(self._damage_dealt)
        defense_score = float(self._block_gained)
        utility_score = float(self._other_effect_score)
        end_turn_score = 2.0 if ended_turn else -0.5
        incoming_penalty = float(unblocked_damage) * 1.2
        total = damage_score + defense_score + utility_score + end_turn_score - incoming_penalty
        total += self._learned_value_bonus

        return BranchScore(
            damage_score=damage_score,
            defense_score=defense_score,
            utility_score=utility_score,
            end_turn_score=end_turn_score,
            incoming_damage_penalty=incoming_penalty,
            projected_incoming_damage=unblocked_damage,
            total=total,
        )

    def apply(self, action: ActionCommand) -> SimulationStepResult:
        if self._action_value_model is not None and self._model_weight > 0:
            self._learned_value_bonus += self._action_value_model.score(self._state, action) * self._model_weight

        if action.action_type in {"end_turn", "event_choose", "map_choose"}:
            return SimulationStepResult(applied=True, boundary=True, reason=f"action boundary: {action.action_type}")

        if action.action_type != "play_card":
            return SimulationStepResult(applied=True, boundary=True, reason=f"unsupported action type: {action.action_type}")

        card_id = normalize_card_id(action.card_id)
        if not card_id:
            return SimulationStepResult(applied=False, boundary=True, reason="missing card_id")

        hand = [normalize_card_id(card) for card in self._state.player.hand]
        if card_id not in hand:
            return SimulationStepResult(applied=False, boundary=True, reason=f"card not in hand: {card_id}")

        cost = _resolve_action_cost(action)
        if cost is None:
            return SimulationStepResult(applied=False, boundary=True, reason=f"unknown card cost: {card_id}")

        if cost > self._state.player.energy:
            return SimulationStepResult(applied=False, boundary=True, reason=f"insufficient energy for {card_id}")

        self._state.player.energy -= cost

        effects = estimate_card_effects(card_id) or CardEffects()
        self._state.player.block += effects.block
        self._block_gained += effects.block
        self._state.player.energy += effects.energy_gain

        if effects.self_hp_loss > 0:
            self._state.player.hp = max(0, self._state.player.hp - effects.self_hp_loss)

        applied_damage = self._apply_damage_to_target(action.target_id, effects.damage)
        self._damage_dealt += applied_damage
        self._other_effect_score += _utility_from_effects(effects)

        # Remove one copy of the played card from hand.
        original_hand = self._state.player.hand
        for index, item in enumerate(original_hand):
            if normalize_card_id(item) == card_id:
                del original_hand[index]
                break

        is_random = bool(action.metadata.get("random")) or is_random_boundary_card(card_id)
        if is_random:
            return SimulationStepResult(applied=True, boundary=True, reason=f"random boundary card: {card_id}")

        return SimulationStepResult(applied=True, boundary=False, reason="deterministic action applied")

    def _apply_damage_to_target(self, target_id: str | None, damage: int) -> int:
        if damage <= 0:
            return 0

        target = _resolve_target_enemy(self._state.enemies, target_id)
        if target is None:
            return 0

        blocked = min(target.block, damage)
        target.block -= blocked
        hp_damage = max(0, damage - blocked)
        actual = min(target.hp, hp_damage)
        target.hp -= actual
        return actual


def _resolve_action_cost(action: ActionCommand) -> int | None:
    metadata_cost = action.metadata.get("cost")
    if isinstance(metadata_cost, int):
        return metadata_cost
    return estimate_card_cost(action.card_id)


def _resolve_target_enemy(enemies: list[EnemyState], target_id: str | None) -> EnemyState | None:
    if target_id:
        for enemy in enemies:
            if enemy.enemy_id == target_id and enemy.hp > 0:
                return enemy
    for enemy in enemies:
        if enemy.hp > 0:
            return enemy
    return None


def _first_living_enemy_id(enemies: Iterable[EnemyState]) -> str | None:
    for enemy in enemies:
        if enemy.hp > 0:
            return enemy.enemy_id
    return None


def _estimate_incoming_damage(enemies: Iterable[EnemyState]) -> int:
    total = 0
    for enemy in enemies:
        if enemy.hp <= 0:
            continue
        for intent in enemy.intents:
            total += _intent_to_damage(intent)
    return max(0, total)


def _intent_to_damage(intent: EnemyIntent) -> int:
    if intent.intent_type not in {"attack", "multi_attack", "unknown"}:
        return 0
    amount = intent.amount
    if amount is None and intent.min_amount is not None and intent.max_amount is not None:
        amount = (intent.min_amount + intent.max_amount) // 2
    elif amount is None and intent.max_amount is not None:
        amount = intent.max_amount
    elif amount is None and intent.min_amount is not None:
        amount = intent.min_amount

    if amount is None:
        return 0

    expected = max(0.0, min(1.0, float(intent.probability))) * float(max(0, amount))
    return int(round(expected * max(1, intent.hits or 1)))


def _utility_from_effects(effects: CardEffects) -> float:
    return (
        effects.draw * 0.6
        + effects.energy_gain * 2.0
        + effects.strength_delta * 2.0
        + effects.dexterity_delta * 1.5
        + effects.vulnerable * 1.0
        + effects.weak * 1.0
        + effects.frail * 0.5
        - effects.self_hp_loss * 1.5
    )


