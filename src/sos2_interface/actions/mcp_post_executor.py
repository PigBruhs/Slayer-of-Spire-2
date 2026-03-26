from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request

from sos2_interface.actions.noop_executor import ActionExecutor
from sos2_interface.contracts.action import ActionCommand, ActionResult


@dataclass
class McpPostExecutorConfig:
    host: str = "127.0.0.1"
    port: int = 15526
    mode: str = "singleplayer"
    timeout_seconds: float = 2.0


class McpPostActionExecutor(ActionExecutor):
    """Executes planner actions against STS2MCP POST endpoint."""

    def __init__(self, config: McpPostExecutorConfig, allow_live_actions: bool = False) -> None:
        self._config = config
        self._allow_live_actions = bool(allow_live_actions)

    def execute(self, action: ActionCommand) -> ActionResult:
        if not self._allow_live_actions:
            return ActionResult(
                accepted=False,
                executor="mcp_post",
                action_id=action.action_id,
                message="Live actions are disabled. Pass --enable-live-actions to allow POST execution.",
            )

        if action.action_type == "noop":
            return ActionResult(
                accepted=True,
                executor="mcp_post",
                action_id=action.action_id,
                message="Noop action skipped.",
            )

        payload, mapping_warning = self._to_mcp_payload(action)
        if payload is None:
            return ActionResult(
                accepted=False,
                executor="mcp_post",
                action_id=action.action_id,
                message=mapping_warning or f"Unsupported action_type '{action.action_type}'.",
            )

        response_text, err = self._post_payload(payload)
        if err is not None:
            return ActionResult(
                accepted=False,
                executor="mcp_post",
                action_id=action.action_id,
                message=err,
            )

        if mapping_warning:
            msg = f"{mapping_warning}; {response_text}" if response_text else mapping_warning
        else:
            msg = response_text or "MCP action accepted"

        return ActionResult(
            accepted=True,
            executor="mcp_post",
            action_id=action.action_id,
            message=msg,
        )

    def _to_mcp_payload(self, action: ActionCommand) -> tuple[dict[str, object] | None, str | None]:
        direct_actions = {
            "end_turn": "end_turn",
            "undo_end_turn": "undo_end_turn",
            "combat_confirm_selection": "combat_confirm_selection",
            "skip_card_reward": "skip_card_reward",
            "rewards_skip_card": "skip_card_reward",
            "proceed": "proceed",
            "proceed_to_map": "proceed",
            "confirm_selection": "confirm_selection",
            "deck_confirm_selection": "confirm_selection",
            "cancel_selection": "cancel_selection",
            "deck_cancel_selection": "cancel_selection",
            "skip_relic_selection": "skip_relic_selection",
            "relic_skip": "skip_relic_selection",
            "advance_dialogue": "advance_dialogue",
            "event_advance_dialogue": "advance_dialogue",
        }
        if action.action_type in direct_actions:
            return {"action": direct_actions[action.action_type]}, None

        index_actions = {
            "event_choose": "choose_event_option",
            "choose_event_option": "choose_event_option",
            "map_choose": "choose_map_node",
            "choose_map_node": "choose_map_node",
            "claim_reward": "claim_reward",
            "rewards_claim": "claim_reward",
            "choose_rest_option": "choose_rest_option",
            "rest_choose_option": "choose_rest_option",
            "shop_purchase": "shop_purchase",
            "select_card": "select_card",
            "deck_select_card": "select_card",
            "select_relic": "select_relic",
            "relic_select": "select_relic",
            "claim_treasure_relic": "claim_treasure_relic",
            "treasure_claim_relic": "claim_treasure_relic",
        }
        if action.action_type in index_actions:
            if action.option_index is None:
                return None, f"{action.action_type} requires option_index"
            return {"action": index_actions[action.action_type], "index": int(action.option_index)}, None

        if action.action_type in {"combat_select_card", "select_card_reward", "rewards_pick_card"}:
            if action.option_index is None:
                return None, f"{action.action_type} requires option_index"
            mapped = "combat_select_card" if action.action_type == "combat_select_card" else "select_card_reward"
            return {"action": mapped, "card_index": int(action.option_index)}, None

        if action.action_type == "use_potion":
            slot = _metadata_int(action.metadata, "slot")
            if slot is None:
                slot = action.option_index
            if slot is None:
                return None, "use_potion requires option_index or metadata.slot"

            payload: dict[str, object] = {"action": "use_potion", "slot": int(slot)}
            if action.target_id:
                payload["target"] = action.target_id
            return payload, None

        if action.action_type == "play_card":
            card_index = _metadata_int(action.metadata, "card_index")
            warning: str | None = None
            if card_index is None:
                card_index, warning = self._resolve_card_index(action.card_id)
            if card_index is None:
                return None, warning or "play_card requires metadata.card_index or resolvable card_id"

            payload = {"action": "play_card", "card_index": int(card_index)}
            if action.target_id:
                payload["target"] = action.target_id
            return payload, warning

        return None, None

    def _resolve_card_index(self, card_id: str | None) -> tuple[int | None, str | None]:
        if not card_id:
            return None, "play_card missing card_id"

        current, err = self._fetch_state()
        if err is not None or current is None:
            return None, err or "failed to fetch current state for card index mapping"

        battle = current.get("battle") if isinstance(current.get("battle"), dict) else {}
        player = battle.get("player") if isinstance(battle, dict) else {}
        hand = player.get("hand") if isinstance(player, dict) else []
        if not isinstance(hand, list):
            return None, "current state does not include battle.player.hand"

        normalized = _normalize_card_id(card_id)
        matches: list[int] = []
        for idx, item in enumerate(hand):
            if not isinstance(item, dict):
                continue
            raw_id = item.get("id") or item.get("card_id") or item.get("name")
            if _normalize_card_id(str(raw_id or "")) == normalized:
                matches.append(idx)

        if not matches:
            return None, f"card '{card_id}' not found in current hand"
        if len(matches) > 1:
            return matches[0], f"ambiguous card '{card_id}' matched multiple copies; using first index {matches[0]}"
        return matches[0], None

    def _post_payload(self, payload: dict[str, object]) -> tuple[str | None, str | None]:
        url = self._endpoint_url()
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = request.Request(url=url, data=data, method="POST", headers={"Content-Type": "application/json"})

        try:
            with request.urlopen(req, timeout=self._config.timeout_seconds) as resp:
                text = resp.read().decode("utf-8")
                return _extract_ok_message(text), None
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            detail = _extract_error_message(body) or exc.reason
            if exc.code == 409:
                expected = "singleplayer" if self._config.mode == "singleplayer" else "multiplayer"
                other = "multiplayer" if expected == "singleplayer" else "singleplayer"
                detail = f"{detail}; endpoint mode mismatch (configured={expected}, try={other})"
            return None, f"mcp POST failed HTTP {exc.code}: {detail}"
        except error.URLError as exc:
            return None, f"mcp POST connection error: {exc.reason}"
        except TimeoutError:
            return None, "mcp POST request timed out"

    def _fetch_state(self) -> tuple[dict[str, object] | None, str | None]:
        url = f"{self._endpoint_url()}?format=json"
        req = request.Request(url=url, method="GET")
        try:
            with request.urlopen(req, timeout=self._config.timeout_seconds) as resp:
                text = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            return None, f"mcp GET failed HTTP {exc.code}: {exc.reason}"
        except error.URLError as exc:
            return None, f"mcp GET connection error: {exc.reason}"
        except TimeoutError:
            return None, "mcp GET request timed out"

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"mcp GET invalid json: {exc.msg}"

        if not isinstance(payload, dict):
            return None, "mcp GET payload is not an object"
        if str(payload.get("status", "ok")).lower() == "error":
            return None, str(payload.get("error") or "mcp API returned error status")
        return payload, None

    def _endpoint_url(self) -> str:
        path = "/api/v1/singleplayer" if self._config.mode == "singleplayer" else "/api/v1/multiplayer"
        return f"http://{self._config.host}:{self._config.port}{path}"


def _metadata_int(metadata: dict[str, str | int | float | bool], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _normalize_card_id(text: str) -> str:
    lowered = text.strip().lower().replace("-", "_").replace(" ", "_")
    return "".join(ch for ch in lowered if ch.isalnum() or ch == "_")


def _extract_ok_message(raw: str) -> str | None:
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()
    if isinstance(payload, dict):
        if str(payload.get("status", "ok")).lower() == "error":
            return str(payload.get("error") or "unknown mcp error")
        return str(payload.get("message") or payload.get("status") or "ok")
    return raw.strip()


def _extract_error_message(raw: str) -> str | None:
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()
    if isinstance(payload, dict):
        return str(payload.get("error") or payload.get("detail") or "unknown error")
    return raw.strip()

