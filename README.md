# Slayer-of-Spire-2

Only one workflow is maintained: `self play loop -> online training -> dual models`.

## Single Entry

- `tools/start_sampling_training.py`

All retired planner/MLP/offline launcher scripts were removed to keep the project lean.

## Quick Start (Windows PowerShell)

```powershell
Set-Location "E:\Slayer-of-Spire-2"
C:\Users\Ecthelion\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run (Continuous Self-Play)

```powershell
Set-Location "E:\Slayer-of-Spire-2"
.\.venv\Scripts\Activate.ps1
python tools\start_sampling_training.py --menu-mouse-config config\auto_menu_mouse.local.json --episodes 0
```

## Required Menu Config

File: `config/auto_menu_mouse.local.json`

Required sequence keys:

- `start_singleplayer_defect`
- `softlock_troubleshoot`
- `return_to_main_menu`
- `post_run_continue`

Capture workflow is fixed to 6 clicks total:

- start flow: 4 clicks
- softlock troubleshoot: 1 click
- return to main menu: 1 click

## Core Runtime Outputs (Kept)

- `runtime/planner_action_trace.selfplay.jsonl`
- `runtime/planner_cycles.selfplay.jsonl`
- `runtime/selfplay_metrics.jsonl`
- `runtime/combat_policy_model.json`
- `runtime/noncombat_policy_model.json`
- `runtime/tensorboard/dual_rl/`

## Key Settings

`tools/start_sampling_training.py` frequently used options:

- `--episodes` (`0` for continuous run)
- `--menu-mouse-config`
- `--menu-seq-start`, `--menu-seq-return`, `--menu-seq-continue`
- `--menu-seq-cooldown-ms`
- `--retrain-every`
- `--global-train-every-episode`, `--global-train-epochs`
- `--combat-train-every-combat`, `--combat-train-epochs`
- `--epsilon-start`, `--epsilon-end`, `--epsilon-decay-episodes`
- `--soft-loop-streak`, `--soft-loop-window`, `--soft-loop-hit-limit`

## Stop

Press `Ctrl+C` in the terminal.
