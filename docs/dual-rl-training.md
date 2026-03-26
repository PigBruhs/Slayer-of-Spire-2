# Dual RL Training (Combat + Non-Combat)

This pipeline trains two separate value models from planner action traces:

- `combat_policy_model.json`: optimized for minimizing HP loss during combat.
- `noncombat_policy_model.json`: optimized for maximizing run progression (act/floor) outside combat.

## 1) Collect traces

Run self-play (or manual play with planner actions enabled) to collect `runtime/planner_action_trace*.jsonl`.

You can now run a single-file self-learning loop directly (sampling + periodic retraining):

```powershell
Set-Location "E:\Slayer-of-Spire-2"
python tools\start_sampling_training.py --episodes 0 --retrain-every 5
```

`--episodes 0` means continuous learning until you stop it.

For full auto-loop (menu -> singleplayer -> Defect -> next run), configure mouse-only sequences:

1. Copy `config/auto_menu_mouse.example.json` to `config/auto_menu_mouse.local.json` (or let capture tool create it).
2. Run coordinate capture workflow:

```powershell
Set-Location "E:\Slayer-of-Spire-2"
python tools\capture_menu_mouse_config.py --input config\auto_menu_mouse.example.json --output config\auto_menu_mouse.local.json --countdown 3.0
```

3. Follow prompts to capture each click point.
   - Defeat-exit button capture is no longer required: script now hard-codes x=853 y=744 and clicks twice with 1s interval, then waits 3s.
4. Start self-learning loop with the captured config.

```powershell
Set-Location "E:\Slayer-of-Spire-2"
python tools\start_sampling_training.py --episodes 0 --retrain-every 5 --menu-mouse-config config\auto_menu_mouse.local.json
```

Optional explicit sequence name override:

```powershell
Set-Location "E:\Slayer-of-Spire-2"
python tools\start_sampling_training.py --episodes 0 --retrain-every 5 --menu-mouse-config config\auto_menu_mouse.local.json --menu-seq-end end_run_to_menu
```

## 2) Train dual models

```powershell
Set-Location "E:\Slayer-of-Spire-2"
python tools\train_dual_rl_models.py --trace runtime\planner_action_trace.selfplay.jsonl --combat-out runtime\combat_policy_model.json --noncombat-out runtime\noncombat_policy_model.json --epochs 16
```

## 3) Use in planner

```powershell
Set-Location "E:\Slayer-of-Spire-2"
python -m sos2_interface.planner_main --reader mcp-api --executor mcp-post --enable-live-actions --combat-model runtime\combat_policy_model.json --noncombat-model runtime\noncombat_policy_model.json --no-combat-only
```

## Notes

- `--combat-only` keeps non-combat decisions manual.
- `--no-combat-only` enables non-combat automation using `noncombat_policy_model.json`.
- Models are trained with discounted returns over contiguous segments (combat and non-combat split).

