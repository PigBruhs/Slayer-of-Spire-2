# Dual RL Pipeline (Single Maintained Flow)

This repository now keeps one training path only:

1. self-play loop
2. online retraining
3. dual model refresh (`combat` + `noncombat`)

Entry point:

- `tools/start_sampling_training.py`

## Start Continuous Training

```powershell
Set-Location "E:\Slayer-of-Spire-2"
.\.venv\Scripts\Activate.ps1
python tools\start_sampling_training.py --menu-mouse-config config\auto_menu_mouse.local.json --episodes 0
```

## Runtime Artifacts

Kept output files:

- `runtime/planner_action_trace.selfplay.jsonl`
- `runtime/planner_cycles.selfplay.jsonl`
- `runtime/selfplay_metrics.jsonl`
- `runtime/combat_policy_model.json`
- `runtime/noncombat_policy_model.json`
- `runtime/tensorboard/dual_rl/`

Old action-value / branch-weight / autobuild artifacts were removed.

## Menu Automation Requirements

`config/auto_menu_mouse.local.json` must include:

- `start_singleplayer_defect`
- `softlock_troubleshoot`
- `return_to_main_menu`
- `post_run_continue`

Mouse capture is fixed to 6 clicks total:

- 4 for `start_singleplayer_defect`
- 1 for `softlock_troubleshoot`
- 1 for `return_to_main_menu`

## Key Training Controls

- run length: `--episodes`
- retrain cadence: `--retrain-every`
- global retrain: `--global-train-every-episode`, `--global-train-epochs`
- combat-only retrain trigger: `--combat-train-every-combat`, `--combat-train-epochs`
- exploration: `--epsilon-start`, `--epsilon-end`, `--epsilon-decay-episodes`
- loop recovery: `--soft-loop-streak`, `--soft-loop-window`, `--soft-loop-hit-limit`

## Optional Offline Refresh (Same Dual Models)

If needed, you can still retrain dual models directly from selfplay traces:

```powershell
Set-Location "E:\Slayer-of-Spire-2"
.\.venv\Scripts\Activate.ps1
python tools\train_dual_rl_models.py --trace runtime\planner_action_trace.selfplay.jsonl --combat-out runtime\combat_policy_model.json --noncombat-out runtime\noncombat_policy_model.json --epochs 16
```
