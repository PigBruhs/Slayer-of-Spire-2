from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
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
from sos2_interface.policy.combat_policy_model import CombatPolicyModel


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


@dataclass
class BranchFactorWeights:
    damage: float = 1.0
    block: float = 0.9
    draw: float = 0.6
    energy_gain: float = 2.0
    strength_delta: float = 2.0
    dexterity_delta: float = 1.5
    vulnerable: float = 1.0
    weak: float = 1.0
    frail: float = 0.5
    self_hp_loss: float = 1.5
    incoming_damage: float = 1.25
    end_turn_bonus: float = 0.6
    no_end_turn_penalty: float = 0.3
    premature_end_turn_penalty: float = 2.0

    @classmethod
    def from_json(cls, path: str | None) -> "BranchFactorWeights":
        if not path:
            return cls()
        file_path = Path(path)
        if not file_path.exists():
            return cls()
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        if not isinstance(payload, dict):
            return cls()

        defaults = cls()
        values: dict[str, float] = {}
        for key, default_value in defaults.__dict__.items():
            raw = payload.get(key, default_value)
            try:
                values[key] = float(raw)
            except (TypeError, ValueError):
                values[key] = float(default_value)
        return cls(**values)


class DeterministicSegmentSimulator:
    def __init__(
        self,
        snapshot: GameStateSnapshot,
        action_value_model: ActionValueModel | None = None,
        model_weight: float = 0.35,
        combat_policy_model: CombatPolicyModel | None = None,
        combat_model_weight: float = 0.8,
        branch_weights: BranchFactorWeights | None = None,
    ) -> None:
        self._state = snapshot.model_copy(deep=True)
        self._state.warnings = []
        self._action_value_model = action_value_model
        self._model_weight = max(0.0, float(model_weight))
        self._combat_policy_model = combat_policy_model
        self._combat_model_weight = max(0.0, float(combat_model_weight))
        self._branch_weights = branch_weights or BranchFactorWeights()

        self._damage_dealt = 0.0
        self._block_gained = 0.0
        self._draw_total = 0.0
        self._energy_gain_total = 0.0
        self._strength_delta_total = 0.0
        self._dexterity_delta_total = 0.0
        self._vulnerable_total = 0.0
        self._weak_total = 0.0
        self._frail_total = 0.0
        self._self_hp_loss_total = 0.0
        self._learned_value_bonus = 0.0
        self._used_potion_slots: set[int] = set()

    @property
    def state(self) -> GameStateSnapshot:
        return self._state

    def clone(self) -> "DeterministicSegmentSimulator":
        copied = DeterministicSegmentSimulator(
            self._state.model_copy(deep=True),
            action_value_model=self._action_value_model,
            model_weight=self._model_weight,
            combat_policy_model=self._combat_policy_model,
            combat_model_weight=self._combat_model_weight,
            branch_weights=self._branch_weights,
        )
        copied._damage_dealt = self._damage_dealt
        copied._block_gained = self._block_gained
        copied._draw_total = self._draw_total
        copied._energy_gain_total = self._energy_gain_total
        copied._strength_delta_total = self._strength_delta_total
        copied._dexterity_delta_total = self._dexterity_delta_total
        copied._vulnerable_total = self._vulnerable_total
        copied._weak_total = self._weak_total
        copied._frail_total = self._frail_total
        copied._self_hp_loss_total = self._self_hp_loss_total
        copied._learned_value_bonus = self._learned_value_bonus
        copied._used_potion_slots = set(self._used_potion_slots)
        return copied

    def list_candidate_actions(self) -> list[ActionCommand]:
        if self._state.in_event and self._state.event and self._state.event.options:
            return [ActionCommand(action_type="event_choose", option_index=0, metadata={"boundary": True})]

        if not self._state.in_combat:
            return [ActionCommand(action_type="noop", metadata={"boundary": True})]

        actions: list[ActionCommand] = [ActionCommand(action_type="end_turn", metadata={"boundary": True})]
        actions.extend(self._list_potion_actions())

        if self._state.player.energy <= 0:
            return actions

        target_id = _first_living_enemy_id(self._state.enemies)
        for index, raw_card in enumerate(self._state.player.hand):
            card_id = normalize_card_id(raw_card)
            if not card_id:
                continue
            cost = estimate_card_cost(card_id)
            if cost is None or cost > self._state.player.energy:
                continue

            actions.append(
                ActionCommand(
                    action_type="play_card",
                    card_id=card_id,
                    target_id=target_id,
                    metadata={"cost": cost, "random": is_random_boundary_card(card_id), "card_index": index},
                )
            )
        return actions

    def evaluate_branch(self, ended_turn: bool) -> BranchScore:
        incoming_damage = _estimate_incoming_damage(self._state.enemies)
        unblocked_damage = max(0, incoming_damage - self._state.player.block)

        w = self._branch_weights
        damage_score = self._damage_dealt * w.damage
        defense_score = self._block_gained * w.block
        utility_score = (
            self._draw_total * w.draw
            + self._energy_gain_total * w.energy_gain
            + self._strength_delta_total * w.strength_delta
            + self._dexterity_delta_total * w.dexterity_delta
            + self._vulnerable_total * w.vulnerable
            + self._weak_total * w.weak
            + self._frail_total * w.frail
            - self._self_hp_loss_total * w.self_hp_loss
        )

        playable_left = _count_playable_cards(self._state.player.hand, self._state.player.energy)
        end_turn_score = w.end_turn_bonus if ended_turn else -w.no_end_turn_penalty
        if ended_turn and playable_left > 0:
            end_turn_score -= w.premature_end_turn_penalty * float(playable_left)

        incoming_penalty = float(unblocked_damage) * w.incoming_damage
        total = damage_score + defense_score + utility_score + end_turn_score - incoming_penalty + self._learned_value_bonus

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
        if self._combat_policy_model is not None and self._combat_model_weight > 0:
            self._learned_value_bonus += self._combat_policy_model.score(self._state, action) * self._combat_model_weight

        if action.action_type in {"end_turn", "event_choose", "map_choose"}:
            return SimulationStepResult(applied=True, boundary=True, reason=f"action boundary: {action.action_type}")
        if action.action_type == "use_potion":
            return self._apply_potion(action)
        if action.action_type != "play_card":
            return SimulationStepResult(applied=True, boundary=True, reason=f"unsupported action type: {action.action_type}")

        card_id = normalize_card_id(action.card_id)
        if not card_id:
            return SimulationStepResult(applied=False, boundary=True, reason="missing card_id")

        hand_index = _metadata_int(action.metadata.get("card_index"))
        hand = [normalize_card_id(card) for card in self._state.player.hand]
        if card_id not in hand:
            return SimulationStepResult(applied=False, boundary=True, reason=f"card not in hand: {card_id}")

        if hand_index is not None:
            if hand_index < 0 or hand_index >= len(self._state.player.hand):
                return SimulationStepResult(applied=False, boundary=True, reason=f"card_index out of range: {hand_index}")
            if normalize_card_id(self._state.player.hand[hand_index]) != card_id:
                return SimulationStepResult(applied=False, boundary=True, reason="card_index/card_id mismatch")

        cost = _resolve_action_cost(action)
        if cost is None:
            return SimulationStepResult(applied=False, boundary=True, reason=f"unknown card cost: {card_id}")
        if cost > self._state.player.energy:
            return SimulationStepResult(applied=False, boundary=True, reason=f"insufficient energy for {card_id}")

        self._state.player.energy -= cost

        effects = estimate_card_effects(card_id) or CardEffects()
        self._state.player.block += effects.block
        self._block_gained += float(effects.block)
        self._state.player.energy += effects.energy_gain
        self._draw_total += float(effects.draw)
        self._energy_gain_total += float(effects.energy_gain)
        self._strength_delta_total += float(effects.strength_delta)
        self._dexterity_delta_total += float(effects.dexterity_delta)
        self._vulnerable_total += float(effects.vulnerable)
        self._weak_total += float(effects.weak)
        self._frail_total += float(effects.frail)
        self._self_hp_loss_total += float(max(0, effects.self_hp_loss))

        if effects.self_hp_loss > 0:
            self._state.player.hp = max(0, self._state.player.hp - effects.self_hp_loss)

        dealt = self._apply_damage_to_target(action.target_id, effects.damage)
        self._damage_dealt += float(dealt)

        if hand_index is not None:
            del self._state.player.hand[hand_index]
        else:
            for idx, item in enumerate(self._state.player.hand):
                if normalize_card_id(item) == card_id:
                    del self._state.player.hand[idx]
                    break

        if bool(action.metadata.get("random")) or is_random_boundary_card(card_id):
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

    def _list_potion_actions(self) -> list[ActionCommand]:
        battle = _battle_state(self._state)
        player = battle.get("player") if isinstance(battle.get("player"), dict) else {}
        potions = player.get("potions") if isinstance(player.get("potions"), list) else []
        target_id = _first_living_enemy_id(self._state.enemies)

        actions: list[ActionCommand] = []
        for fallback_idx, item in enumerate(potions):
            if not isinstance(item, dict):
                continue
            slot = _metadata_int(item.get("slot"))
            if slot is None:
                slot = _metadata_int(item.get("index"))
            if slot is None:
                slot = fallback_idx
            if slot in self._used_potion_slots:
                continue

            target_type = str(item.get("target_type") or "").lower()
            target = target_id if "enemy" in target_type else None
            actions.append(
                ActionCommand(
                    action_type="use_potion",
                    option_index=slot,
                    target_id=target,
                    metadata={"slot": slot, "target_type": target_type, "potion_id": str(item.get("id") or item.get("name") or f"slot_{slot}")},
                )
            )
        return actions

    def _apply_potion(self, action: ActionCommand) -> SimulationStepResult:
        slot = _metadata_int(action.metadata.get("slot"))
        if slot is None:
            slot = action.option_index
        if slot is None:
            return SimulationStepResult(applied=False, boundary=True, reason="missing potion slot")
        if slot in self._used_potion_slots:
            return SimulationStepResult(applied=False, boundary=True, reason=f"potion already used: slot {slot}")

        potion = _find_potion_by_slot(self._state, slot)
        if potion is None:
            return SimulationStepResult(applied=False, boundary=True, reason=f"potion slot not found: {slot}")

        effects = _estimate_potion_effects(potion)
        block = int(effects.get("block", 0))
        heal = int(effects.get("heal", 0))
        damage = int(effects.get("damage", 0))
        self_loss = int(effects.get("self_hp_loss", 0))
        utility = float(effects.get("utility", 0.0))

        if block > 0:
            self._state.player.block += block
            self._block_gained += float(block)
        if heal > 0:
            self._state.player.hp = min(self._state.player.max_hp, self._state.player.hp + heal)
        if self_loss > 0:
            self._state.player.hp = max(0, self._state.player.hp - self_loss)
            self._self_hp_loss_total += float(self_loss)
        if damage > 0:
            dealt = self._apply_damage_to_target(action.target_id, damage)
            self._damage_dealt += float(dealt)

        self._learned_value_bonus += utility
        self._used_potion_slots.add(slot)
        return SimulationStepResult(applied=True, boundary=False, reason=f"potion used: slot {slot}")


def _resolve_action_cost(action: ActionCommand) -> int | None:
    metadata_cost = action.metadata.get("cost")
    if isinstance(metadata_cost, int):
        return metadata_cost
    return estimate_card_cost(action.card_id)


def _metadata_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


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


def _count_playable_cards(hand: list[str], energy: int) -> int:
    count = 0
    for raw_card in hand:
        card_id = normalize_card_id(raw_card)
        if not card_id:
            continue
        cost = estimate_card_cost(card_id)
        if cost is None:
            continue
        if cost <= energy:
            count += 1
    return count


def _battle_state(state: GameStateSnapshot) -> dict[str, object]:
    raw = state.raw_state if isinstance(state.raw_state, dict) else {}
    battle = raw.get("battle")
    return battle if isinstance(battle, dict) else {}


def _find_potion_by_slot(state: GameStateSnapshot, slot: int) -> dict[str, object] | None:
    battle = _battle_state(state)
    player = battle.get("player") if isinstance(battle.get("player"), dict) else {}
    potions = player.get("potions") if isinstance(player.get("potions"), list) else []
    for fallback_idx, item in enumerate(potions):
        if not isinstance(item, dict):
            continue
        item_slot = _metadata_int(item.get("slot"))
        if item_slot is None:
            item_slot = _metadata_int(item.get("index"))
        if item_slot is None:
            item_slot = fallback_idx
        if item_slot == slot:
            return item
    return None


def _estimate_potion_effects(potion: dict[str, object]) -> dict[str, int | float]:
    name = str(potion.get("name") or potion.get("id") or "").lower()
    desc = str(potion.get("description") or "").lower()
    text = f"{name} {desc}"
    values = [int(token) for token in re.findall(r"\d+", text)]
    value = values[0] if values else 0

    damage = 0
    block = 0
    heal = 0
    self_hp_loss = 0
    utility = 0.4

    if any(key in text for key in ["damage", "伤害"]):
        damage = max(damage, value)
        utility += 0.2
    if any(key in text for key in ["block", "格挡"]):
        block = max(block, value)
        utility += 0.15
    if any(key in text for key in ["heal", "恢复", "回复"]):
        heal = max(heal, value)
        utility += 0.2
    if any(key in text for key in ["lose hp", "失去生命", "自损"]):
        self_hp_loss = max(self_hp_loss, value)
        utility -= 0.3

    return {
        "damage": damage,
        "block": block,
        "heal": heal,
        "self_hp_loss": self_hp_loss,
        "utility": utility,
    }

