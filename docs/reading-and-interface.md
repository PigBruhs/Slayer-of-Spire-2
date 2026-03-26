# 读取原理与接口说明

本文档说明 `sos2-interface` 的读取原理、接口协议、以及如何从 Mock 平滑迁移到真实内存读取。

## 1. 目标与边界

- 目标：实时读取对局状态（事件/战斗）并输出可执行动作命令。
- 当前阶段：优先完成“接口层”，不强依赖训练模型。
- 安全落地策略：
  - 先用 `MockReader` 打通端到端。
  - 再接 `MemoryReader`，只替换数据源，不改 API 与策略层。

## 2. 架构总览

- `Reader`：负责“从游戏取状态”。
  - `MemoryReader`：读取进程内存（数值稳定、低延迟）。
  - `ScreenReader`：OCR 读取固定屏幕区域（减少地址维护负担）。
  - `HybridReader`：内存优先 + 屏幕兜底。
  - `ModReader`：由游戏模组主动推送完整状态快照（推荐长期方案）。
  - `McpApiReader`：直接轮询 STS2MCP 的 REST API（`localhost:15526`），保留战斗内外全量原始字段。
- `Runtime`：定时轮询 Reader，维护 `latest_state`。
- `API`：向外暴露状态、建议动作、动作提交接口。
- `ActionExecutor`：负责动作输出（目前为 `noop` 记录日志）。
- `Policy`：基于状态给出建议（目前是规则基线）。

## 3. 读取原理

### 3.1 Mock 读取（可立即验证）

`MockReader` 生成稳定、可重复的战斗/事件状态：

- 每隔若干帧切换到事件。
- 战斗帧中包含玩家能量、手牌、敌人意图。
- 可直接用于验证：
  - 接口字段是否完整。
  - 策略模块是否按预期输出动作。
  - API 与日志链路是否稳定。

### 3.2 内存读取（真实对局）

`MemoryReader` 使用 Windows `ReadProcessMemory`：

1. 按进程名查找 PID（默认 `SlayTheSpire2.exe`）。
2. 按字段读取候选地址（`hp/max_hp/energy`）。
3. 自动锁定当前可用地址，失效时自动回退到其他候选。
4. 组装成统一 `GameStateSnapshot`。
5. 读取失败会写入 `warnings`，不中断服务。

当前实现支持“候选地址自动择优”，你后续可升级为：

- 指针链（base + offsets）
- 签名扫描定位动态基址
- 分模块读取（战斗/事件/地图）

### 3.3 屏幕读取（减少反复扫地址）

`ScreenReader` 对固定 ROI（感兴趣区域）做 OCR，当前支持：

- `hp_region`：解析 `hp/max_hp`
- `energy_region`：解析能量值
- `event_region`：根据关键词判断是否在事件界面

这是一个“轻量、可维护”的视觉方案：

- 不做全屏语义理解，只抓决策关键字段。
- 允许内存字段缺失时仍能提供基本决策输入。
- 可用静态截图调参（`image_path`），减少反复进游戏测试。

### 3.4 混合读取（推荐默认）

`HybridReader` 规则：

- 内存数据可用时优先用内存。
- 内存缺失/异常时回退到屏幕 OCR。
- 统一输出为同一个 `GameStateSnapshot`，上层策略无需改动。

### 3.5 Mod 推送读取（推荐长期）

`ModReader` 不依赖 OCR 或内存地址：

1. 游戏模组通过 `POST /ingest/state` 推送完整 `GameStateSnapshot`。
2. 服务端缓存最新快照。
3. `GET /state`、`GET /suggestions`、planner 直接消费该快照。

这样可以把“读取复杂度”交给官方 mod 接口，接口层只做协议接收与策略。

### 3.6 MCP API 读取（推荐当前联调）

`McpApiReader` 直接拉取 STS2MCP 的官方本地接口：

- `GET /api/v1/singleplayer?format=json`
- （可选）`GET /api/v1/multiplayer?format=json`

输出策略：

1. 归一化关键字段到 `GameStateSnapshot`（`player/enemies/event/in_combat/in_event`）。
2. 把原始返回完整保存在 `raw_state`，用于地图、商店、奖励、遗物选择等战斗外逻辑。
3. 使用 `state_type` 标记当前屏幕（如 `monster/map/shop/event/menu`）。

这样你可以先不丢数据地跑通流程，再按需求逐步把 `raw_state` 拆成更细的强类型结构。

## 4. 接口协议

### 4.1 状态结构：`GameStateSnapshot`

关键字段：

- `source`: `mock | memory | screen | hybrid | mod | mcp_api`
- `frame_id`: 轮询帧号
- `timestamp_ms`: 毫秒时间戳
- `in_combat`, `in_event`: 场景开关
- `turn`: 回合数（已知时）
- `player`: 血量/格挡/能量/手牌/牌堆计数
- `enemies`: 敌人状态 + 意图
- `event`: 事件页面与选项
- `state_type`: MCP 屏幕类型（若可用）
- `raw_state`: MCP 原始响应（保留战斗外完整信息）
- `warnings`: 读取失败或配置问题

### 4.2 动作结构：`ActionCommand`

支持动作类型（MCP 对齐，含别名）：

- 战斗：`play_card`、`use_potion`、`end_turn`、`undo_end_turn`、`combat_select_card`、`combat_confirm_selection`
- 奖励：`claim_reward | rewards_claim`、`select_card_reward | rewards_pick_card`、`skip_card_reward | rewards_skip_card`
- 场景推进：`proceed | proceed_to_map`
- 事件与地图：`event_choose | choose_event_option`、`map_choose | choose_map_node`、`advance_dialogue | event_advance_dialogue`
- 休息点与商店：`choose_rest_option | rest_choose_option`、`shop_purchase`
- 选牌：`select_card | deck_select_card`、`confirm_selection | deck_confirm_selection`、`cancel_selection | deck_cancel_selection`
- 遗物与宝箱：`select_relic | relic_select`、`skip_relic_selection | relic_skip`、`claim_treasure_relic | treasure_claim_relic`
- 其他：`noop`

所有动作包含 `action_id`（用于幂等追踪）。

## 5. API 说明

默认地址 `http://127.0.0.1:8765`。

- `GET /health`：Reader 状态
- `GET /state`：最新状态
- `GET /suggestions`：规则建议动作列表
- `POST /action`：提交动作（当前写入 `runtime/actions.log`）

## 6. 实时调试步骤（Windows）

### 6.1 先验证 Mock 链路

1. 启动服务（Mock）。
2. 打开 `GET /state` 与 `GET /suggestions`。
3. 用 `tools/watch_state.py` 连续打印 JSON。

### 6.2 切换 Memory 链路

1. 复制 `config/memory_map.example.json` 为本地配置。
2. 填入你逆向得到的地址（建议放在 `*_candidates` 列表里）。
3. 使用 `--reader memory --memory-map <path>` 启动。
4. 观察 `warnings` 字段定位读取失败点。
5. 访问 `GET /health`，确认是否出现 `hp_address` / `energy_address`（代表已自动锁定）。

示例：

```json
{
  "process_name": "SlayTheSpire2.exe",
  "hp_candidates": ["0x1F4F5D5AB08", "0x1F4F9045F6C", "0x1F4FDC24480"],
  "max_hp_candidates": [],
  "energy_candidates": ["0x1F4FDC4C9B8"]
}
```

说明：

- 地址字符串支持 `0x` 前缀，也支持你直接粘贴 CE 的十六进制值。
- 若有多个可读候选，读取器会优先沿用上一帧最稳定的地址，减少抖动。

### 6.3 常见问题：`game process not found`

如果游戏已经启动但仍报错，通常是进程名不匹配。先用 PowerShell 查实际进程名：

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match "slay|spire|sts2" } |
  Select-Object Name, ProcessId, ExecutablePath
```

然后把 `config/memory_map.example.json` 里的 `process_name` 改成查到的 `Name`（例如 `SlayTheSpire2.exe`）。

### 6.4 启用 Screen 或 Hybrid

1. 安装 Tesseract OCR（Windows 默认路径通常是 `C:\Program Files\Tesseract-OCR\tesseract.exe`）。
2. 复制 `config/screen_map.example.json` 并根据你的分辨率调整 ROI。
3. 使用 `screen` 或 `hybrid` 模式启动。

```powershell
sos2-interface --reader screen --screen-map config\screen_map.example.json
sos2-interface --reader hybrid --memory-map config\memory_map.example.json --screen-map config\screen_map.example.json
```

4. 访问 `GET /health` 看依赖是否就绪（`has_pytesseract`、`has_cv2` 等）。

ROI 调参建议：

- 先在静态截图上调：把 `image_path` 指向一张游戏截图，反复看 `/state` 输出。
- ROI 尽量小且稳定，避免动画区域。
- 数值区域优先黑白高对比（便于 OCR）。

### 6.5 使用 MCP API 读取（战斗内外全量）

1. 确保 STS2MCP 模组已在游戏内启用，默认监听 `127.0.0.1:15526`。
2. 启动接口层：

```powershell
Set-Location "E:\Slayer-of-Spire-2"
sos2-interface --reader mcp-api --mcp-host 127.0.0.1 --mcp-port 15526 --mcp-mode singleplayer
```

3. 查看状态（注意 `state_type/raw_state`）：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/state" | ConvertTo-Json -Depth 16
```

4. 使用调试脚本连续采样：

```powershell
Set-Location "E:\Slayer-of-Spire-2"
python tools\debug_mcp_state.py --mcp-config config\mcp_api.example.json --interval-ms 200
```

脚本会：

- 打印状态切换（如 `map -> monster -> combat_rewards`）
- 输出关键字段摘要（HP/能量/手牌数/敌人数）
- 持久化每帧 JSON 到 `runtime/mcp_debug.jsonl`（含完整 `raw_state`）

## 7. 后续演进建议

- 先扩展只读字段：敌人意图、事件选项、手牌细节。
- 再做动作执行器：`SendInput`（低侵入）或内存写入（高上限）。
- 在每局日志中保存 `state -> action -> outcome`，为后续策略优化和离线学习做数据闭环。

## 8. 确定性段 Planner（按边界更新）

为了避免“每打一张牌就重读一次状态”，当前新增了 `PlannerLoop`：

1. `observe`：读取当前状态。
2. `plan`：生成一个短动作段（segment）。
3. `execute`：连续执行动作段。
4. `reobserve`：只在边界事件后重读状态。

`SegmentPlanner` 内部使用 `DeterministicSegmentSimulator` 做轻量推演：

- 状态副本：当前 `hand + energy`
- 数值更新：打牌会消耗能量、应用近似效果（伤害/格挡/抽牌等）并从手牌移除
- 成本映射：先用动作 `metadata.cost`，否则走内置卡表（后续可替换为 Spire Codex）
- 随机边界：抽牌/随机效果牌会立刻停止当前段，回到 `reobserve`

当前分支评价（branch scoring）包含：

- `damage_score`：该分支累计造成的伤害
- `defense_score`：该分支累计获得的格挡
- `utility_score`：其他效果分值（如抽牌、能量、状态增益）
- `end_turn_score`：结束回合作为显式决策项
- `incoming_damage_penalty`：根据敌人意图估计回合末承伤惩罚

敌人意图结构已扩展，支持：

- `amount` / `hits`
- `min_amount` / `max_amount`
- `probability` / `is_random`
- `intent_text`

用于估算“预计来伤”时，会结合 `hits` 与 `probability`。

### 8.1 本地卡牌知识文件（手动维护）

模拟器默认读取 `config/card_knowledge.local.json`，你可以手动更新它来提升规划质量。

文件结构：

```json
{
  "costs": {
    "strike": 1,
    "defend": 1,
    "bash": 2
  },
  "random_boundary_cards": [
    "shrug_it_off",
    "pommel_strike"
  ]
}
```

手动更新方式：

1. 结束一局或调试后，把新识别到的 `card_id` 填到 `costs`。
2. 对会抽牌/随机生成/随机目标的牌，加入 `random_boundary_cards`。
3. 保存文件后无需重启服务，下一轮 planner 会自动热更新。
4. 每次更新后运行校验脚本，避免 JSON 或数据类型错误。

```powershell
Set-Location "E:\Slayer-of-Spire-2"
python tools\validate_card_knowledge.py --file config\card_knowledge.local.json
```

可选：如果你想放在别处，用环境变量指定：

```powershell
$env:SOS2_CARD_KNOWLEDGE_PATH = "E:\Slayer-of-Spire-2\config\card_knowledge.local.json"
sos2-planner --reader hybrid --memory-map config\memory_map.example.json --screen-map config\screen_map.example.json
```

边界事件（会触发重观测）：

- `end_turn` / `event_choose` / `map_choose`
- 场景切换（战斗 <-> 事件）
- 状态告警（`warnings`）
- 关键数值变化（例如能量变化）

运行命令：

```powershell
sos2-planner --reader hybrid --memory-map config\memory_map.example.json --screen-map config\screen_map.example.json --iterations 10
```

输出：

- `runtime/planner_actions.jsonl`：执行器动作日志（dry-run）
- `runtime/planner_cycles.jsonl`：每轮的 `observed -> planned_actions -> refreshed` 轨迹
- `runtime/planner_action_trace.jsonl`：逐动作 `before -> action -> after` 转移日志（启用 `--capture-action-trace`）

可调参数：

- `--max-segment-actions`：每轮最多规划的连续动作数（默认 `4`）
- `--capture-action-trace`：记录逐动作状态转移（用于训练样本）
- `--trace-raw-state`：在逐动作日志中写入完整状态快照（数据更全但更大）

### 8.2 可选：接入 MCP 实际执行（自对局采样）

默认建议先 `dry-run` 观测决策质量。若要让模型真实执行动作并采样：

```powershell
Set-Location "E:\Slayer-of-Spire-2"
sos2-planner --reader mcp-api --mcp-config config\mcp_api.example.json --executor mcp-post --enable-live-actions --capture-action-trace --iterations 200
```

说明：

- `mcp-post` 执行器会把 `ActionCommand` 映射为 STS2MCP 的 `POST /api/v1/singleplayer` 动作。
- `play_card` 通过当前手牌自动映射 `card_id -> card_index`；若同名牌多张，默认取第一张并写入告警信息。
- 未加 `--enable-live-actions` 时，执行器会拒绝真实操作（`accepted=false`），避免误触发。
- 执行器现已覆盖 raw API 里的主要动作：战斗、奖励、地图、事件、商店、休息点、选牌/选遗物、宝箱、对话推进，以及多人 `undo_end_turn`。

### 8.3 训练动作价值模型（首版）

从 `planner_action_trace` 生成监督样本并训练线性价值模型：

```powershell
Set-Location "E:\Slayer-of-Spire-2"
python tools\build_action_dataset.py --input runtime\planner_action_trace.jsonl --output runtime\action_value_dataset.jsonl --gamma 0.95
python tools\train_action_value_model.py --dataset runtime\action_value_dataset.jsonl --out runtime\action_value_model.json --epochs 8 --lr 0.03 --target return
```

训练后可在 planner 中启用：

```powershell
sos2-planner --reader mcp-api --mcp-config config\mcp_api.example.json --value-model runtime\action_value_model.json --value-model-weight 0.35 --executor dry-run
```

说明：

- 当前模型是轻量线性回归（SGD），用于快速把动作转移日志转成可学习打分。
- 目标默认 `return`（折扣回报），由 `build_action_dataset.py` 根据转移变化构造。
- 这是收敛管线的起步版，后续可升级到 CQL/IQL 或策略梯度，并接入完整终局胜率标签。

### 8.4 自动训练（自对局 + 周期重训）

自动流程脚本：`tools/auto_train_selfplay.py`

能力：

- 自动检测/拉起游戏进程
- 运行中自动对局；在 `menu` 状态暂停等待人工开启下一局
- 每回合调用 planner + `mcp-post` 执行动作
- 出牌节流：默认每张牌间隔 `50ms`
- 可选等待战斗可行动状态（`is_play_phase/turn/player_actions_disabled`）再打下一张牌
- 每 `N` 局自动执行：`build_action_dataset.py -> train_action_value_model.py`
- 输出收敛监控指标到 `runtime/selfplay_metrics.jsonl`

运行示例：

```powershell
Set-Location "E:\Slayer-of-Spire-2"
python tools\auto_train_selfplay.py --game-exe "D:\Steam\steamapps\common\Slay the Spire 2\SlayTheSpire2.exe" --episodes 50 --retrain-every 5 --play-card-interval-ms 50 --wait-play-phase
```

重要限制：

- STS2MCP raw API 当前没有“主菜单开局/返回主菜单”动作；当前脚本不再做主菜单自动化，需人工开下一局。
- 胜负标签目前是近似判定（是否到达 Act3 Boss 且未观测到玩家死亡），用于先验证收敛速度；建议后续在 mod 侧补充明确的 run outcome 字段。

### 8.5 TensorBoard 收敛可视化

`auto_train_selfplay.py` 与 `train_action_value_model.py` 支持写入 TensorBoard 标量日志。

启动可视化：

```powershell
Set-Location "E:\Slayer-of-Spire-2"
tensorboard --logdir runtime\tensorboard
```

常用曲线：

- `selfplay/win`
- `selfplay/rolling_win_rate`
- `selfplay/max_act`
- `train/train_rmse`
- `train/valid_rmse`

## 9. 全量卡牌映射（离线，本地生成）

你要求“第一次就支持所有卡”，建议直接按 `projet layout.md` 的数据管线走本地全量生成：

1. 用 GDRE + ILSpy 从本机游戏目录提取并反编译。
2. 用本地 parser 产出卡牌 JSON（例如 Spire Codex 的 `data/{lang}/cards*.json`）。
3. 用本仓库脚本把卡牌 JSON 生成 `config/card_knowledge.local.json`，包含：
   - `costs`: `card_id -> 能量`
   - `aliases`: OCR 文本别名 -> `card_id`
   - `random_boundary_cards`: 抽牌/随机/发现等边界牌

### 9.1 生成卡牌映射文件

```powershell
Set-Location "E:\Slayer-of-Spire-2"
python tools\build_card_knowledge_from_codex.py --cards-glob "E:\path\to\spire-codex\data\eng\cards*.json" --out config\card_knowledge.local.json
python tools\validate_card_knowledge.py --file config\card_knowledge.local.json
```

如果你有多个来源文件，也可以重复 `--cards-json`：

```powershell
python tools\build_card_knowledge_from_codex.py --cards-json "E:\codex\data\eng\cards.json" --cards-json "E:\codex\data\zhs\cards.json"
```

### 9.2 手牌识别到 `card_id`

`ScreenReader` 现在支持 `hand_regions`：每个 ROI 对应一个手牌槽位。流程是：

1. OCR 读取每个手牌 ROI 文本。
2. 读取 `config/card_knowledge.local.json` 里的 `aliases` 做标准化映射。
3. 输出 `player.hand=[card_id,...]` 给模拟器和 planner。

注意：

- 若出现 `hand card mapping failed at slot ...`，说明该槽位 OCR 文本未命中别名。
- 可把漏识别文本手动加到 `aliases` 并热更新（无需重启）。

### 9.3 手动选择 OCR 区域（小前端工具）

提供交互式区域选择器：

```powershell
Set-Location "E:\Slayer-of-Spire-2"
python tools\pick_screen_regions.py --screen-map config\screen_map.example.json
```

操作方式：

- 鼠标左键拖动：绘制当前元素区域
- `n`：跳过当前元素
- `u`：撤销上一项
- `s`：保存并退出

输出会写回 `screen_map`，并自动整理 `hand_1..hand_n` 为 `hand_regions`。

