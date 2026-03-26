# 读取与接口（当前维护版）

本文档只描述当前维护流程相关内容：

- 唯一训练入口：`tools/start_sampling_training.py`
- 主要状态来源：STS2MCP API（`McpApiReader`）
- 在线产物：selfplay trace + dual models

## 1. 当前流程

1. 读取当前游戏状态（MCP API）。
2. 自博弈循环执行动作。
3. 记录 selfplay trace/metrics。
4. 触发双模型训练（combat + noncombat）。
5. 热加载新模型继续循环。

## 2. 关键状态字段

`GameStateSnapshot` 里训练/决策常用字段：

- `state_type`
- `in_combat`
- `player`（`hp/energy/hand`）
- `enemies`
- `raw_state`（保留 MCP 原始信息，含地图/商店/事件等）

## 3. 动作输出

动作通过 MCP POST 执行器发送，覆盖战斗与非战斗常见操作：

- combat: `play_card`, `use_potion`, `end_turn`
- reward/map/event/shop/select: 对应 `claim/proceed/choose/select` 动作族

## 4. 运行入口

```powershell
Set-Location "E:\Slayer-of-Spire-2"
.\.venv\Scripts\Activate.ps1
python tools\start_sampling_training.py --menu-mouse-config config\auto_menu_mouse.local.json --episodes 0
```

## 5. 菜单自动化

依赖 `config/auto_menu_mouse.local.json` 的序列：

- `start_singleplayer_defect`
- `softlock_troubleshoot`
- `return_to_main_menu`
- `post_run_continue`

鼠标坐标录入固定为 6 次：开始 4 次、排障 1 次、返回主界面 1 次。

## 6. 保留产物命名

只保留以下 runtime 产物：

- `runtime/planner_action_trace.selfplay.jsonl`
- `runtime/planner_cycles.selfplay.jsonl`
- `runtime/selfplay_metrics.jsonl`
- `runtime/combat_policy_model.json`
- `runtime/noncombat_policy_model.json`
- `runtime/tensorboard/dual_rl/`

其余旧命名（action-value、branch-factor、autobuild、test/branchtest）均已清理。
