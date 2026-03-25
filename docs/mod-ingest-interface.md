# Mod 推送接口文档

本文档定义本地模组向 `sos2-interface` 推送对局状态的协议。

## 1. 启动方式

使用 `mod` 读取模式启动服务：

```powershell
Set-Location "E:\Slayer-of-Spire-2"
sos2-interface --reader mod --host 127.0.0.1 --port 8765
```

可选安全项（建议）：

```powershell
$env:SOS2_INGEST_TOKEN = "your-local-token"
sos2-interface --reader mod --host 127.0.0.1 --port 8765
```

当设置了 `SOS2_INGEST_TOKEN`，模组必须在请求头带上 `X-SOS2-Token`。

## 2. 接口列表

- `GET /health`
  - 查看服务状态。
  - `ingest_enabled=true` 表示允许外部推送。
- `GET /state`
  - 获取最近一次成功推送的状态快照。
- `POST /ingest/state`
  - 推送完整状态快照（核心接口）。

## 3. 推送协议：`POST /ingest/state`

### 3.1 请求头

- `Content-Type: application/json`
- `X-SOS2-Token: <token>`（仅在服务端设置 `SOS2_INGEST_TOKEN` 时必填）

### 3.2 请求体

请求体是一个完整的 `GameStateSnapshot`。

```json
{
  "source": "mod",
  "frame_id": 1001,
  "timestamp_ms": 1774359000000,
  "in_combat": true,
  "in_event": false,
  "turn": 3,
  "player": {
    "hp": 63,
    "max_hp": 80,
    "block": 12,
    "energy": 2,
    "hand": ["strike", "defend", "bash"],
    "draw_pile_count": 11,
    "discard_pile_count": 6
  },
  "enemies": [
    {
      "enemy_id": "slaver_red",
      "hp": 40,
      "max_hp": 50,
      "block": 0,
      "intents": [
        {
          "enemy_id": "slaver_red",
          "intent_type": "attack",
          "amount": 12,
          "hits": 1,
          "min_amount": 10,
          "max_amount": 14,
          "probability": 1.0,
          "is_random": false,
          "intent_text": "Attack"
        }
      ]
    }
  ],
  "event": null,
  "warnings": []
}
```

### 3.3 返回体

```json
{
  "ok": true,
  "source": "mod",
  "frame_id": 1001,
  "timestamp_ms": 1774359000000
}
```

## 4. 字段说明（关键）

- `frame_id`
  - 模组侧自增帧号。
  - 若推送值小于等于当前帧，服务端会自动修正为当前帧+1。
- `timestamp_ms`
  - 毫秒时间戳。
  - 若传 `<=0`，服务端会自动填充当前时间。
- `player.hand`
  - 请传标准化 `card_id` 列表（例如 `shrug_it_off`）。
- `enemies[].intents[]`
  - 至少建议填 `intent_type + amount + hits`。
  - 支持 `min_amount/max_amount/probability`，用于分支评分中的来伤估计。

## 5. 错误码

- `401 invalid ingest token`
  - `SOS2_INGEST_TOKEN` 已设置，但请求未带正确 `X-SOS2-Token`。
- `409 ingest is disabled for current reader`
  - 服务不是 `--reader mod` 模式。

## 6. 最小联调流程

1. 启动服务（`--reader mod`）。
2. 模组每次决策前（或每回合关键节点）调用一次 `POST /ingest/state`。
3. 调试端使用 `GET /state`、`GET /suggestions`、`POST /action` 联动验证。

示例（PowerShell）：

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://127.0.0.1:8765/state" | ConvertTo-Json -Depth 8
```

