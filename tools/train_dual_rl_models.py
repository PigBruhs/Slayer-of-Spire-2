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

from sos2_interface.policy.combat_policy_model import extract_feature_vector_from_compact as combat_features
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


class QRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual RL value models (combat + noncombat) from planner action traces")
    parser.add_argument("--trace", default="runtime/planner_action_trace.selfplay.jsonl")
    parser.add_argument("--combat-out", default="runtime/combat_policy_model.json")
    parser.add_argument("--noncombat-out", default="runtime/noncombat_policy_model.json")
    parser.add_argument("--gamma-combat", type=float, default=0.96)
    parser.add_argument("--gamma-noncombat", type=float, default=0.995)
    parser.add_argument("--combat-input-dim", type=int, default=192)
    parser.add_argument("--noncombat-input-dim", type=int, default=224)
    parser.add_argument("--combat-hidden1", type=int, default=768)
    parser.add_argument("--combat-hidden2", type=int, default=512)
    parser.add_argument("--noncombat-hidden1", type=int, default=768)
    parser.add_argument("--noncombat-hidden2", type=int, default=512)
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
    parser.add_argument("--tb-logdir", default="runtime/tensorboard/dual_rl")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--train-combat-only", action="store_true", help="Train only combat model")
    parser.add_argument("--train-noncombat-only", action="store_true", help="Train only noncombat model")
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

    if args.train_combat_only and args.train_noncombat_only:
        raise ValueError("--train-combat-only and --train-noncombat-only are mutually exclusive")

    train_combat = not bool(args.train_noncombat_only)
    train_noncombat = not bool(args.train_combat_only)

    tb_logger = TensorboardLogger(None if args.no_tensorboard else args.tb_logdir)
    if not args.no_tensorboard and not tb_logger.enabled:
        print("tensorboard disabled: install tensorboardX to enable")

    if train_combat and combat_rows:
        x_c, y_c = build_dataset(
            combat_rows,
            feature_fn=lambda before, action: combat_features(before, action, input_dim=args.combat_input_dim),
            reward_fn=combat_reward,
            gamma=float(args.gamma_combat),
        )
        model_c = train_regressor(
            x=x_c,
            y=y_c,
            input_dim=args.combat_input_dim,
            hidden1=args.combat_hidden1,
            hidden2=args.combat_hidden2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            tb_logger=tb_logger,
            tag="combat",
        )
        save_model_json(
            out_path=Path(args.combat_out),
            model=model_c,
            input_dim=args.combat_input_dim,
            hidden1=args.combat_hidden1,
            hidden2=args.combat_hidden2,
            model_name="combat_value_mlp",
            trace_path=trace_path,
        )
    elif train_combat:
        print("[dual-rl] skip combat model: no combat rows")

    if train_noncombat and noncombat_rows:
        tokens_n, numeric_n, y_n = build_noncombat_dataset(
            noncombat_rows,
            reward_fn=noncombat_reward,
            gamma=float(args.gamma_noncombat),
            token_seq_len=int(args.noncombat_token_seq_len),
            token_buckets=int(args.noncombat_token_buckets),
            numeric_dim=int(args.noncombat_numeric_dim),
        )
        model_n = train_noncombat_transformer(
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
        save_noncombat_transformer_json(
            out_path=Path(args.noncombat_out),
            model=model_n,
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


def build_dataset(
    rows: list[Transition],
    feature_fn,
    reward_fn,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[Transition]] = {}
    for row in rows:
        grouped.setdefault(row.segment_id, []).append(row)

    features: list[np.ndarray] = []
    targets: list[float] = []

    for _, segment_rows in sorted(grouped.items(), key=lambda item: item[0]):
        rewards = [reward_fn(step) for step in segment_rows]
        returns = discounted_returns(rewards, gamma=gamma)
        for step, target in zip(segment_rows, returns):
            features.append(feature_fn(step.before, step.action).astype(np.float32))
            targets.append(float(target))

    x = np.stack(features, axis=0) if features else np.zeros((0, 1), dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    return x, y


def build_noncombat_dataset(
    rows: list[Transition],
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
            tokens, numeric = noncombat_deep_features(
                step.before,
                step.action,
                token_seq_len=token_seq_len,
                token_buckets=token_buckets,
                numeric_dim=numeric_dim,
            )
            token_list.append(tokens.astype(np.int64))
            numeric_list.append(numeric.astype(np.float32))
            targets.append(float(target))

    x_tokens = np.stack(token_list, axis=0) if token_list else np.zeros((0, token_seq_len), dtype=np.int64)
    x_numeric = np.stack(numeric_list, axis=0) if numeric_list else np.zeros((0, numeric_dim), dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    return x_tokens, x_numeric, y


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


def train_regressor(
    x: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    hidden1: int,
    hidden2: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    tb_logger: TensorboardLogger,
    tag: str,
) -> QRegressor:
    if x.shape[0] == 0:
        raise RuntimeError(f"no samples for {tag} model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QRegressor(input_dim=input_dim, hidden1=hidden1, hidden2=hidden2).to(device)

    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)
    split = max(1, int(x.shape[0] * 0.9))
    train_idx = idxs[:split]
    valid_idx = idxs[split:]
    if valid_idx.size == 0:
        valid_idx = train_idx[: min(256, train_idx.size)]

    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(y[train_idx]).unsqueeze(1)
    x_valid = torch.from_numpy(x[valid_idx])
    y_valid = torch.from_numpy(y[valid_idx]).unsqueeze(1)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=max(32, batch_size), shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu()) * int(batch_x.shape[0])
            total_count += int(batch_x.shape[0])

        train_loss = total_loss / max(1, total_count)
        valid_rmse = evaluate_rmse(model, x_valid, y_valid, device)
        print(f"[{tag}] epoch={epoch} train_huber={train_loss:.5f} valid_rmse={valid_rmse:.5f}")
        tb_logger.add_scalar(f"dual_rl/{tag}_train_huber", train_loss, epoch)
        tb_logger.add_scalar(f"dual_rl/{tag}_valid_rmse", valid_rmse, epoch)

    return model.cpu()


def train_noncombat_transformer(
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
) -> NonCombatTransformerRegressor:
    if token_ids.shape[0] == 0:
        raise RuntimeError("no noncombat samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NonCombatTransformerRegressor(
        vocab_size=token_buckets,
        token_seq_len=token_seq_len,
        numeric_dim=numeric_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
    ).to(device)

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
            return int(t_train.shape[0])

        def __getitem__(self, idx):
            return t_train[idx], n_train[idx], y_train[idx]

    loader = DataLoader(_Dataset(), batch_size=max(32, batch_size), shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_t, batch_n, batch_y in loader:
            batch_t = batch_t.to(device)
            batch_n = batch_n.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(batch_t, batch_n)
                loss = loss_fn(pred, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * int(batch_t.shape[0])
            total_count += int(batch_t.shape[0])

        train_loss = total_loss / max(1, total_count)
        valid_rmse = evaluate_transformer_rmse(model, t_valid, n_valid, y_valid, device)
        print(f"[{tag}] epoch={epoch} train_huber={train_loss:.5f} valid_rmse={valid_rmse:.5f}")
        tb_logger.add_scalar(f"dual_rl/{tag}_train_huber", train_loss, epoch)
        tb_logger.add_scalar(f"dual_rl/{tag}_valid_rmse", valid_rmse, epoch)

    return model.cpu()


def evaluate_transformer_rmse(
    model: NonCombatTransformerRegressor,
    t_valid: torch.Tensor,
    n_valid: torch.Tensor,
    y_valid: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(t_valid.to(device), n_valid.to(device)).cpu()
        mse = torch.mean((pred - y_valid) ** 2)
    return float(torch.sqrt(mse + 1e-12))


def evaluate_rmse(model: QRegressor, x_valid: torch.Tensor, y_valid: torch.Tensor, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x_valid.to(device)).cpu()
        mse = torch.mean((pred - y_valid) ** 2)
    return float(torch.sqrt(mse + 1e-12))


def save_model_json(
    out_path: Path,
    model: QRegressor,
    input_dim: int,
    hidden1: int,
    hidden2: int,
    model_name: str,
    trace_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    l1 = model.net[0]
    l2 = model.net[2]
    l3 = model.net[4]

    payload = {
        "created_at_ms": int(time.time() * 1000),
        "model_type": model_name,
        "input_dim": input_dim,
        "hidden1": hidden1,
        "hidden2": hidden2,
        "source_trace": str(trace_path),
        "w1": l1.weight.detach().cpu().numpy().T.tolist(),
        "b1": l1.bias.detach().cpu().numpy().tolist(),
        "w2": l2.weight.detach().cpu().numpy().T.tolist(),
        "b2": l2.bias.detach().cpu().numpy().tolist(),
        "w3": l3.weight.detach().cpu().numpy().T.tolist(),
        "b3": l3.bias.detach().cpu().numpy().tolist(),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"model_saved={out_path}")


def save_noncombat_transformer_json(
    out_path: Path,
    model: NonCombatTransformerRegressor,
    trace_path: Path,
    token_buckets: int,
    token_seq_len: int,
    numeric_dim: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    ff_dim: int,
    dropout: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu().numpy().tolist()
        for key, value in model.state_dict().items()
    }
    payload = {
        "created_at_ms": int(time.time() * 1000),
        "model_type": "noncombat_transformer_value",
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
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"model_saved={out_path}")


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


