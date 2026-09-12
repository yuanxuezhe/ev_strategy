## Why

策略信号行 (`format_signal_line` hook) 当前两个用法:
- **默认** (`VectorizedStrategy.format_signal_line`): 仅 `ts / sig / side` 三列
- **`channel_deviation`** 已覆写: 多了 `up / dw / low_dev / high_dev / cash / position`,但**缺成交价 + 触发该信号的 K 线 OHLCV**

协作现场定位信号时, "在什么价位的什么 K 线上触发了 BUY/SELL" 是最高频需求; 现状是用户得自己拿 `ts` 去行情里翻, 体验差。`filtered_mr` 走默认 hook 更严重, 信号行基本是裸 sig。

本次只动两个策略 (`filtered_mr` / `channel_deviation`), framework **零改动** — `Engine._process_bucket` / `vectorized_engine._aggregate_loop` 已经把策略 `self._last_info` 透传给 hook, 策略只需在 step 末尾把想要的字段写进 `_last_info` 即可。这是 framework 已有的契约, 没有任何 framework 行为变化。

## What Changes

- **`evtrade/strategies/filtered_mr.py`**:
  - step 末尾 (line 287 之前) 写 `self._last_info = {"side": "BUY"/"SELL", "price": float(c), "o": ..., "h": ..., "l": ..., "c": ..., "v": float(v)}` (从触发该 sig 的 bar 拿)
  - 新增 `format_signal_line(self, ts, sig, info=None) -> str`, 打印: 方向前缀 + ts + 价格 + 4 列 OHLC + 成交量; 与 `channel_deviation` 现有格式保持视觉一致 (单行, 字段用 `|` 分隔)
- **`evtrade/strategies/channel_deviation.py`**:
  - step 末尾 `_last_info` (line 227) 增加 `price` / `o` / `h` / `l` / `c` / `v` 字段 (从当前 finalized bar 拿); 现有 up/dw/low_dev/high_dev/cash/position 字段保留
  - hook 打印格式重排, 把价格/OHLCV 放在更显眼位置; 原 up/dw/dev 字段保留
- **`tests/`**: 新增 `tests/test_format_signal_line_enrich.py`, 锁住两个策略的 hook 输出:
  - `sig != 0` 时格式化的行 MUST 含方向词 (BUY/SELL) + 价格 + OHLCV 全部字段
  - `sig == 0` 时 hook 不会被 framework 调用 (与 spec 一致), 不测
  - `info` 缺字段时 hook MUST NOT 抛 KeyError (`info.get(...)` 防御)
- **`kbs/06-交易策略详解.md`** §信号行打印段: 同步两个策略的当前 hook 输出形态 + 新约定 "info 应含 price + OHLCV"
- **`kbs/09-引擎Engine与主流程.md`**: 无改动 (framework 行为不变)

无 **BREAKING**。改动只是策略展示层, framework / step / state 都不动。

## Capabilities

### New Capabilities
（无）

### Modified Capabilities
（无。原 `Requirement: Strategy display hooks are framework-agnostic` 不动 —— framework 行为确实不变; 新约束落在新增 Requirement 上, 见 New Capabilities）

### New Capabilities
- `evtrade-architecture`: 新增 `Requirement: Strategy signal line shows trigger bar context` —— 锁定策略行为契约: step 末尾 `_last_info` 必含 `side / price / o / h / l / c / v`; hook 用 `info.get(...)`; `sig != 0` 时返回字符串含方向词 + 价格 + 4 个 OHLC 字段名。 4 个 Scenario 覆盖 BUY/SELL 两条 + info 为空不抛 + 字段与触发 bar 一致。

## Impact

- **framework**: 零行改动。`engine.py` / `vectorized_engine.py` 已有的 `getattr(strategy, "_last_info", None)` 路径继续生效。
- **API / 行为**: `format_signal_line(ts, sig, info=None)` 签名不变; `VectorizedStrategy` 基类 hook 不变 (其他策略继续走默认); `_last_info` 字段集是策略私有约定, 不进 spec 关键字。
- **测试**: 现有 `test_filtered_mr.py` / `test_strategy_unified.py` 不应回归 (它们不验 hook 输出); 新增 1 个测试文件 ~30 行。
- **KB / spec**: spec delta 1 个 Scenario; kbs/06 改 1 段。
