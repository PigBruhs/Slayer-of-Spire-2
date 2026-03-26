from __future__ import annotations

from sos2_interface.contracts.state import EnemyState, GameStateSnapshot


def compact_state(state: GameStateSnapshot) -> dict[str, object]:
    raw = state.raw_state if isinstance(state.raw_state, dict) else {}
    run = raw.get("run") if isinstance(raw.get("run"), dict) else {}
    return {
        "frame_id": state.frame_id,
        "state_type": state.state_type,
        "in_combat": state.in_combat,
        "in_event": state.in_event,
        "turn": state.turn,
        "player": {
            "hp": state.player.hp,
            "max_hp": state.player.max_hp,
            "block": state.player.block,
            "energy": state.player.energy,
            "hand_count": len(state.player.hand),
            "draw_pile_count": state.player.draw_pile_count,
            "discard_pile_count": state.player.discard_pile_count,
        },
        "run": {
            "act": run.get("act"),
            "floor": run.get("floor"),
            "ascension": run.get("ascension"),
        },
        "enemies": [
            {
                "enemy_id": enemy.enemy_id,
                "hp": enemy.hp,
                "max_hp": enemy.max_hp,
                "block": enemy.block,
            }
            for enemy in state.enemies
        ],
        "warnings": state.warnings,
    }


def summarize_transition(before: GameStateSnapshot, after: GameStateSnapshot) -> dict[str, object]:
    before_enemy_map = _enemy_map(before.enemies)
    after_enemy_map = _enemy_map(after.enemies)
    all_enemy_ids = sorted(set(before_enemy_map.keys()) | set(after_enemy_map.keys()))

    enemy_hp_delta: list[dict[str, object]] = []
    for enemy_id in all_enemy_ids:
        prev = before_enemy_map.get(enemy_id)
        curr = after_enemy_map.get(enemy_id)
        prev_hp = prev["hp"] if prev else 0
        curr_hp = curr["hp"] if curr else 0
        enemy_hp_delta.append(
            {
                "enemy_id": enemy_id,
                "hp_before": prev_hp,
                "hp_after": curr_hp,
                "hp_delta": curr_hp - prev_hp,
                "died": prev_hp > 0 and curr_hp <= 0,
            }
        )

    return {
        "state_type_before": before.state_type,
        "state_type_after": after.state_type,
        "frame_before": before.frame_id,
        "frame_after": after.frame_id,
        "turn_before": before.turn,
        "turn_after": after.turn,
        "in_combat_before": before.in_combat,
        "in_combat_after": after.in_combat,
        "in_event_before": before.in_event,
        "in_event_after": after.in_event,
        "player_hp_delta": after.player.hp - before.player.hp,
        "player_block_delta": after.player.block - before.player.block,
        "player_energy_delta": after.player.energy - before.player.energy,
        "hand_count_delta": len(after.player.hand) - len(before.player.hand),
        "enemy_hp_delta": enemy_hp_delta,
        "combat_started": (not before.in_combat) and after.in_combat,
        "combat_ended": before.in_combat and (not after.in_combat),
        "player_died": before.player.hp > 0 and after.player.hp <= 0,
        "new_warnings": [w for w in after.warnings if w not in before.warnings],
    }


def _enemy_map(enemies: list[EnemyState]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for enemy in enemies:
        result[enemy.enemy_id] = {"hp": enemy.hp, "max_hp": enemy.max_hp, "block": enemy.block}
    return result

