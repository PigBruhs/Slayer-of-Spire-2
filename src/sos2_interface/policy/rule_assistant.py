from __future__ import annotations

from sos2_interface.contracts.action import ActionCommand
from sos2_interface.contracts.state import GameStateSnapshot
from sos2_interface.policy.card_knowledge import estimate_card_cost, is_random_boundary_card


def suggest_actions(state: GameStateSnapshot) -> list[ActionCommand]:
    suggestions: list[ActionCommand] = []
    next_action = suggest_next_action(state)
    suggestions.append(next_action)

    if next_action.action_type == "play_card":
        suggestions.append(ActionCommand(action_type="end_turn"))
    return suggestions


def suggest_next_action(state: GameStateSnapshot) -> ActionCommand:
    if state.in_event and state.event and state.event.options:
        return ActionCommand(action_type="event_choose", option_index=0, metadata={"boundary": True})

    if not state.in_combat:
        return ActionCommand(action_type="noop")

    if state.player.energy <= 0:
        return ActionCommand(action_type="end_turn", metadata={"boundary": True})

    deterministic_candidates: list[ActionCommand] = []
    random_candidates: list[ActionCommand] = []
    target_id = state.enemies[0].enemy_id if state.enemies else None

    for card in state.player.hand:
        cost = estimate_card_cost(card)
        if cost is None or cost > state.player.energy:
            continue

        metadata = {
            "cost": cost,
            "random": is_random_boundary_card(card),
        }
        action = ActionCommand(action_type="play_card", card_id=card, target_id=target_id, metadata=metadata)
        if bool(metadata["random"]):
            random_candidates.append(action)
        else:
            deterministic_candidates.append(action)

    if deterministic_candidates:
        return deterministic_candidates[0]

    if random_candidates:
        action = random_candidates[0]
        action.metadata["boundary"] = True
        action.metadata["boundary_reason"] = "random card effect"
        return action

    return ActionCommand(action_type="end_turn", metadata={"boundary": True, "boundary_reason": "no affordable card"})

