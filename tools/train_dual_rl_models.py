from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sos2_interface.policy.combat_policy_model import (
    CombatTransformerRegressor,
    extract_deep_features_from_compact as combat_deep_features,
)
from sos2_interface.policy.noncombat_policy_model import (
    NonCombatTransformerRegressor,
    extract_deep_features_from_compact as noncombat_deep_features,
)
from sos2_interface.policy.tensorboard_logger import TensorboardLogger


@dataclass
class Transition:
    segment_id: int
    before: dict[str, object]
    after: dict[str, object]
    action: dict[str, object]
    transition: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual RL value models (combat + noncombat) from planner action traces")
    parser.add_argument("--trace", default="runtime/planner_action_trace.selfplay.jsonl")
    parser.add_argument("--combat-out", default="runtime/combat_policy_model.json")
    parser.add_argument("--noncombat-out", default="runtime/noncombat_policy_model.json")
    parser.add_argument("--gamma-combat", type=float, default=0.96)
    parser.add_argument("--gamma-noncombat", type=float, default=0.995)
    parser.add_argument("--combat-token-buckets", type=int, default=8192)
    parser.add_argument("--combat-token-seq-len", type=int, default=40)
    parser.add_argument("--combat-numeric-dim", type=int, default=24)
    parser.add_argument("--combat-d-model", type=int, default=192)
    parser.add_argument("--combat-nhead", type=int, default=6)
    parser.add_argument("--combat-layers", type=int, default=3)
    parser.add_argument("--combat-ff-dim", type=int, default=384)
    parser.add_argument("--combat-dropout", type=float, default=0.1)
    parser.add_argument("--noncombat-token-buckets", type=int, default=8192)
    parser.add_argument("--noncombat-token-seq-len", type=int, default=40)
    parser.add_argument("--noncombat-numeric-dim", type=int, default=24)
    parser.add_argument("--noncombat-d-model", type=int, default=192)
    parser.add_argument("--noncombat-nhead", type=int, default=6)
    parser.add_argument("--noncombat-layers", type=int, default=3)
    parser.add_argument("--noncombat-ff-dim", type=int, default=384)
    parser.add_argument("--noncombat-dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-recent-segments", type=int, default=24, help="Latest segments treated as new data for mixed replay")
    parser.add_argument("--replay-new-ratio", type=float, default=0.4, help="Target ratio of samples from recent segments (0-1)")
    parser.add_argument("--replay-max-train-samples", type=int, default=12000, help="Training sample cap after replay mixing; 0 disables cap")
    parser.add_argument("--tb-logdir", default="runtime/tensorboard/dual_rl")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--train-combat-only", action="store_true", help="Train only combat model")
    parser.add_argument("--train-noncombat-only", action="store_true", help="Train only noncombat model")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise FileNotFoundError(f"trace not found: {trace_path}")

    rows = load_transitions(trace_path)
    if not rows:
        raise RuntimeError("no action transitions found in trace")

    combat_rows = [row for row in rows if bool(row.before.get("in_combat"))]
    noncombat_rows = [row for row in rows if not bool(row.before.get("in_combat"))]

    combat_rows = mix_replay_rows(
        rows=combat_rows,
        recent_segments=max(0, int(args.replay_recent_segments)),
        new_ratio=float(args.replay_new_ratio),
        max_train_samples=max(0, int(args.replay_max_train_samples)),
        tag="combat",
    )
    noncombat_rows = mix_replay_rows(
        rows=noncombat_rows,
        recent_segments=max(0, int(args.replay_recent_segments)),
        new_ratio=float(args.replay_new_ratio),
        max_train_samples=max(0, int(args.replay_max_train_samples)),
        tag="noncombat",
    )

    if args.train_combat_only and args.train_noncombat_only:
        raise ValueError("--train-combat-only and --train-noncombat-only are mutually exclusive")

    train_combat = not bool(args.train_noncombat_only)
    train_noncombat = not bool(args.train_combat_only)

    tb_logger = TensorboardLogger(None if args.no_tensorboard else args.tb_logdir)
    if not args.no_tensorboard and not tb_logger.enabled:
        print("tensorboard disabled: install tensorboardX to enable")

    if train_combat and combat_rows:
        tokens_c, numeric_c, y_c = build_transformer_dataset(
            combat_rows,
            feature_fn=combat_deep_features,
            reward_fn=combat_reward,
            gamma=float(args.gamma_combat),
            token_seq_len=int(args.combat_token_seq_len),
            token_buckets=int(args.combat_token_buckets),
            numeric_dim=int(args.combat_numeric_dim),
        )

        combat_path = Path(args.combat_out) if args.resume else None

        model_c = train_transformer(
            existing_model_path=combat_path,
            model_class=CombatTransformerRegressor,
            token_ids=tokens_c,
            numeric=numeric_c,
            y=y_c,
            token_buckets=int(args.combat_token_buckets),
            token_seq_len=int(args.combat_token_seq_len),
            numeric_dim=int(args.combat_numeric_dim),
            d_model=int(args.combat_d_model),
            nhead=int(args.combat_nhead),
            num_layers=int(args.combat_layers),
            ff_dim=int(args.combat_ff_dim),
            dropout=float(args.combat_dropout),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            tb_logger=tb_logger,
            tag="combat",
        )
        save_transformer_json(
            out_path=Path(args.combat_out),
            model=model_c,
            model_type_name="combat_transformer_value",
            trace_path=trace_path,
            token_buckets=int(args.combat_token_buckets),
            token_seq_len=int(args.combat_token_seq_len),
            numeric_dim=int(args.combat_numeric_dim),
            d_model=int(args.combat_d_model),
            nhead=int(args.combat_nhead),
            num_layers=int(args.combat_layers),
            ff_dim=int(args.combat_ff_dim),
            dropout=float(args.combat_dropout),
        )
    elif train_combat:
        print("[dual-rl] skip combat model: no combat rows")

    if train_noncombat and noncombat_rows:
        tokens_n, numeric_n, y_n = build_transformer_dataset(
            noncombat_rows,
            feature_fn=noncombat_deep_features,
            reward_fn=noncombat_reward,
            gamma=float(args.gamma_noncombat),
            token_seq_len=int(args.noncombat_token_seq_len),
            token_buckets=int(args.noncombat_token_buckets),
            numeric_dim=int(args.noncombat_numeric_dim),
        )

        noncombat_path = Path(args.noncombat_out) if args.resume else None

        model_n = train_transformer(
            existing_model_path=noncombat_path,
            model_class=NonCombatTransformerRegressor,
            token_ids=tokens_n,
            numeric=numeric_n,
            y=y_n,
            token_buckets=int(args.noncombat_token_buckets),
            token_seq_len=int(args.noncombat_token_seq_len),
            numeric_dim=int(args.noncombat_numeric_dim),
            d_model=int(args.noncombat_d_model),
            nhead=int(args.noncombat_nhead),
            num_layers=int(args.noncombat_layers),
            ff_dim=int(args.noncombat_ff_dim),
            dropout=float(args.noncombat_dropout),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            tb_logger=tb_logger,
            tag="noncombat",
        )
        save_transformer_json(
            out_path=Path(args.noncombat_out),
            model=model_n,
            model_type_name="noncombat_transformer_value",
            trace_path=trace_path,
            token_buckets=int(args.noncombat_token_buckets),
            token_seq_len=int(args.noncombat_token_seq_len),
            numeric_dim=int(args.noncombat_numeric_dim),
            d_model=int(args.noncombat_d_model),
            nhead=int(args.noncombat_nhead),
            num_layers=int(args.noncombat_layers),
            ff_dim=int(args.noncombat_ff_dim),
            dropout=float(args.noncombat_dropout),
        )
    elif train_noncombat:
        print("[dual-rl] skip noncombat model: no noncombat rows")

    tb_logger.close()


def load_transitions(path: Path) -> list[Transition]:
    transitions: list[Transition] = []
    segment_id = 0
    prev_in_combat: bool | None = None

    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            if not isinstance(raw, dict):
                continue

            before = raw.get("before") if isinstance(raw.get("before"), dict) else {}
            after = raw.get("after") if isinstance(raw.get("after"), dict) else {}
            action = raw.get("action") if isinstance(raw.get("action"), dict) else {}
            trans = raw.get("transition") if isinstance(raw.get("transition"), dict) else {}

            in_combat = bool(before.get("in_combat"))
            if prev_in_combat is None:
                prev_in_combat = in_combat
            elif prev_in_combat != in_combat:
                segment_id += 1
                prev_in_combat = in_combat

            if bool(trans.get("combat_ended")) or bool(trans.get("player_died")):
                # Keep the terminal transition in current segment, then roll segment id.
                transitions.append(Transition(segment_id=segment_id, before=before, after=after, action=action, transition=trans))
                segment_id += 1
                prev_in_combat = None
                continue

            transitions.append(Transition(segment_id=segment_id, before=before, after=after, action=action, transition=trans))

    return transitions


def mix_replay_rows(
    rows: list[Transition],
    recent_segments: int,
    new_ratio: float,
    max_train_samples: int,
    tag: str,
) -> list[Transition]:
    if not rows:
        return rows

    ratio = max(0.0, min(1.0, float(new_ratio)))
    ordered_segments = sorted({int(row.segment_id) for row in rows})
    if recent_segments > 0 and ordered_segments:
        cutoff_segment = ordered_segments[max(0, len(ordered_segments) - recent_segments)]
    else:
        cutoff_segment = ordered_segments[0]

    indexed = list(enumerate(rows))
    new_pool = [(idx, row) for idx, row in indexed if int(row.segment_id) >= cutoff_segment]
    old_pool = [(idx, row) for idx, row in indexed if int(row.segment_id) < cutoff_segment]

    total_cap = max_train_samples if max_train_samples > 0 else len(rows)
    total_cap = max(1, min(total_cap, len(rows)))

    if recent_segments <= 0:
        selected = indexed
        if total_cap < len(selected):
            selected = random.sample(selected, total_cap)
        selected.sort(key=lambda item: item[0])
        print(f"[replay] {tag}: mode=full total={len(rows)} used={len(selected)}")
        return [row for _, row in selected]

    desired_new = min(len(new_pool), int(round(total_cap * ratio)))
    desired_old = min(len(old_pool), max(0, total_cap - desired_new))

    selected: list[tuple[int, Transition]] = []
    if desired_new > 0:
        selected.extend(random.sample(new_pool, desired_new))
    if desired_old > 0:
        selected.extend(random.sample(old_pool, desired_old))

    shortfall = total_cap - len(selected)
    if shortfall > 0:
        chosen_idx = {idx for idx, _ in selected}
        remaining = [item for item in indexed if item[0] not in chosen_idx]
        if remaining:
            selected.extend(random.sample(remaining, min(shortfall, len(remaining))))

    selected.sort(key=lambda item: item[0])
    new_used = sum(1 for idx, _ in selected if idx >= 0 and int(rows[idx].segment_id) >= cutoff_segment)
    print(
        f"[replay] {tag}: total={len(rows)} used={len(selected)} "
        f"new_pool={len(new_pool)} old_pool={len(old_pool)} new_used={new_used} "
        f"recent_segments={recent_segments} target_new_ratio={ratio:.2f}"
    )
    return [row for _, row in selected]


def build_transformer_dataset(
    rows: list[Transition],
    feature_fn,
    reward_fn,
    gamma: float,
    token_seq_len: int,
    token_buckets: int,
    numeric_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[int, list[Transition]] = {}
    for row in rows:
        grouped.setdefault(row.segment_id, []).append(row)

    token_list: list[np.ndarray] = []
    numeric_list: list[np.ndarray] = []
    targets: list[float] = []

    for _, segment_rows in sorted(grouped.items(), key=lambda item: item[0]):
        rewards = [reward_fn(step) for step in segment_rows]
        returns = discounted_returns(rewards, gamma=gamma)
        for step, target in zip(segment_rows, returns):
            tokens, numeric = feature_fn(
                step.before,
                step.action,
                token_seq_len=token_seq_len,
                token_buckets=token_buckets,
                numeric_dim=numeric_dim,
            )
            token_list.append(tokens)
            numeric_list.append(numeric)
            targets.append(target)

    if not token_list:
        return np.zeros((0, token_seq_len), dtype=np.int64), np.zeros((0, numeric_dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.stack(token_list), np.stack(numeric_list), np.array(targets, dtype=np.float32)


def combat_reward(row: Transition) -> float:
    t = row.transition
    hp_delta = float(t.get("player_hp_delta") or 0.0)
    block_delta = float(t.get("player_block_delta") or 0.0)
    energy_delta = float(t.get("player_energy_delta") or 0.0)

    enemy_damage = 0.0
    kills = 0.0
    enemy_rows = t.get("enemy_hp_delta") if isinstance(t.get("enemy_hp_delta"), list) else []
    for item in enemy_rows:
        if not isinstance(item, dict):
            continue
        hp_step = float(item.get("hp_delta") or 0.0)
        enemy_damage += max(0.0, -hp_step)
        if bool(item.get("died")):
            kills += 1.0

    # Main objective: preserve HP while still ending fights efficiently.
    reward = 0.0
    reward += hp_delta * 6.0
    reward += max(0.0, block_delta) * 0.2
    reward += enemy_damage * 0.35
    reward += kills * 1.8
    reward += energy_delta * 0.05

    if bool(t.get("combat_ended")) and not bool(t.get("player_died")):
        reward += 6.0
    if bool(t.get("player_died")):
        reward -= 120.0

    return reward


def noncombat_reward(row: Transition) -> float:
    before = row.before
    after = row.after
    t = row.transition

    before_floor = _extract_floor(before)
    after_floor = _extract_floor(after)
    before_act = _extract_act(before)
    after_act = _extract_act(after)

    floor_delta = float(max(0, after_floor - before_floor))
    act_delta = float(max(0, after_act - before_act))
    hp_delta = float(t.get("player_hp_delta") or 0.0)

    reward = 0.0
    reward += floor_delta * 2.0
    reward += act_delta * 15.0
    reward += hp_delta * 0.4

    action_type = str(row.action.get("action_type") or "")
    if action_type in {"skip_card_reward", "rewards_skip_card", "skip_relic_selection", "relic_skip"}:
        reward -= 0.15

    if bool(t.get("player_died")):
        reward -= 160.0

    return reward


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    out = [0.0] * len(rewards)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + gamma * running
        out[idx] = running
    return out


def train_transformer(
    existing_model_path: Path | None,
    model_class: type,
    token_ids: np.ndarray,
    numeric: np.ndarray,
    y: np.ndarray,
    token_buckets: int,
    token_seq_len: int,
    numeric_dim: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    ff_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    tb_logger: TensorboardLogger,
    tag: str,
):
    if token_ids.shape[0] == 0:
        raise RuntimeError(f"no {tag} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(
        vocab_size=token_buckets,
        token_seq_len=token_seq_len,
        numeric_dim=numeric_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
    )
    if existing_model_path and existing_model_path.exists():
        try:
            import json
            with existing_model_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            raw_state_dict = payload.get("state_dict")
            if isinstance(raw_state_dict, dict):
                state_dict = {key: torch.tensor(value, dtype=torch.float32) for key, value in raw_state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
                print(f"[{tag}] Resumed from {existing_model_path}")
        except Exception as e:
            print(f"[{tag}] Failed to resume from {existing_model_path}: {e}")
    model = model.to(device)

    idxs = np.arange(token_ids.shape[0])
    np.random.shuffle(idxs)
    split = max(1, int(token_ids.shape[0] * 0.9))
    train_idx = idxs[:split]
    valid_idx = idxs[split:]
    if valid_idx.size == 0:
        valid_idx = train_idx[: min(256, train_idx.size)]

    t_train = torch.from_numpy(token_ids[train_idx])
    n_train = torch.from_numpy(numeric[train_idx])
    y_train = torch.from_numpy(y[train_idx]).unsqueeze(1)
    t_valid = torch.from_numpy(token_ids[valid_idx])
    n_valid = torch.from_numpy(numeric[valid_idx])
    y_valid = torch.from_numpy(y[valid_idx]).unsqueeze(1)

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return t_train.shape[0]

        def __getitem__(self, idx):
            return t_train[idx], n_train[idx], y_train[idx]

    loader = DataLoader(_Dataset(), batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_valid_rmse = float("inf")
    best_state_dict = None

    print(f"[{tag}] starting transformer training: {t_train.shape[0]} train, {t_valid.shape[0]} valid")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for b_t, b_n, b_y in loader:
            b_t = b_t.to(device)
            b_n = b_n.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            pred = model(b_t, b_n)
            loss = criterion(pred, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item() * b_t.shape[0])

        train_huber = epoch_loss / t_train.shape[0]
        valid_rmse = evaluate_transformer_rmse(model, t_valid, n_valid, y_valid, device)
        scheduler.step(valid_rmse)

        print(f"[{tag}] epoch {epoch+1:02d}/{epochs} | train_huber: {train_huber:.4f} | valid_rmse: {valid_rmse:.4f}")
        tb_logger.add_scalar(f"loss_train_{tag}", train_huber, epoch)
        tb_logger.add_scalar(f"loss_valid_{tag}", valid_rmse, epoch)

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model.cpu()


def evaluate_transformer_rmse(
    model: nn.Module,
    t_valid: torch.Tensor,
    n_valid: torch.Tensor,
    y_valid: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(t_valid.to(device), n_valid.to(device))
        mse = nn.functional.mse_loss(pred, y_valid.to(device))
    return float(torch.sqrt(mse + 1e-12))


def save_transformer_json(
    out_path: Path,
    model: nn.Module,
    model_type_name: str,
    trace_path: Path,
    token_buckets: int,
    token_seq_len: int,
    numeric_dim: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    ff_dim: int,
    dropout: float,
    checkpoint_epoch: int | None = None,
    train_huber: float | None = None,
    valid_rmse: float | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu().numpy().tolist()
        for key, value in model.state_dict().items()
    }
    payload = {
        "created_at_ms": int(time.time() * 1000),
        "model_type": model_type_name,
        "source_trace": str(trace_path),
        "token_buckets": int(token_buckets),
        "token_seq_len": int(token_seq_len),
        "numeric_dim": int(numeric_dim),
        "d_model": int(d_model),
        "nhead": int(nhead),
        "num_layers": int(num_layers),
        "ff_dim": int(ff_dim),
        "dropout": float(dropout),
        "state_dict": state_dict,
    }
    if checkpoint_epoch is not None:
        payload["checkpoint_epoch"] = int(checkpoint_epoch)
    if train_huber is not None:
        payload["train_huber"] = float(train_huber)
    if valid_rmse is not None:
        payload["valid_rmse"] = float(valid_rmse)

    _atomic_write_json(out_path, payload)
    print(f"model_saved={out_path}")


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def _extract_floor(compact: dict[str, object]) -> int:
    run = compact.get("run") if isinstance(compact.get("run"), dict) else {}
    return _to_int(run.get("floor"))


def _extract_act(compact: dict[str, object]) -> int:
    run = compact.get("run") if isinstance(compact.get("run"), dict) else {}
    return _to_int(run.get("act"))


def _to_int(value: object) -> int:
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

