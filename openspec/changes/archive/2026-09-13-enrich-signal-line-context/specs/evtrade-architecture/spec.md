## ADDED Requirements

### Requirement: Strategy signal line shows trigger bar context

策略（至少 `filtered_mr` 与 `channel_deviation`）的信号行 hook MUST 在 `sig != 0` 时把
触发该信号的方向、成交价与触发 K 线 OHLCV 全部展示在一行内，方便协作现场直接读出
"在什么价位的什么 K 线上触发了 BUY/SELL"，不再需要拿 `ts` 反查行情。具体契约：

- `step` 末尾 MUST 写 `self._last_info = {"side": "BUY"|"SELL", "price": float, "o": float,
  "h": float, "l": float, "c": float, "v": float|int}`，字段值取自触发该 sig 的 finalized bar
- `format_signal_line` hook MUST 用 `info.get(...)` 而非 `info[...]`，防止 info 缺字段时
  KeyError
- 当 `sig != 0` 时 hook 返回字符串 MUST 包含方向词（`BUY` 或 `SELL`）、价格数字、
  以及 4 个 OHLC 数字；否则视为该策略违反本契约
- `VectorizedStrategy` 基类默认 hook 不变（仅 `ts / sig / side`），不强制所有策略实现本契约

framework 行为（`Engine._process_bucket` / `run_vectorized` 的 `info` 透传机制）不变，
本条 Requirement 只约束策略层。

#### Scenario: filtered_mr 信号行包含方向/价格/OHLCV
- **WHEN** `filtered_mr` 在某个 finalized bar 上触发 `sig=+1`（BUY）
- **THEN** `format_signal_line(ts, sig, info)` 返回字符串 MUST 同时含 `BUY`、价格字段
  （与该 bar 的 `close` 一致）、`o=`、`h=`、`l=`、`c=` 四个 OHLC 字段名；不含这些的视为缺漏

#### Scenario: channel_deviation 信号行扩展后含触发 K 线
- **WHEN** `channel_deviation` 在某个 finalized bar 上触发 `sig=-1`（SELL）
- **THEN** `format_signal_line(ts, sig, info)` 返回字符串 MUST 含 `SELL`、价格、4 个 OHLC
  字段名；原有的 `up` / `dw` / `low_dev` / `high_dev` 字段 MUST 仍然出现（向后兼容）

#### Scenario: hook 用 info.get 不抛 KeyError
- **WHEN** 协作者直接调用 `strategy.format_signal_line(ts, sig, info={})`（info 为空 dict）
- **THEN** hook MUST NOT 抛 `KeyError`；MUST 返回包含 `ts` / `sig` / `BUY`-or-`SELL`-or-空
  的字符串（OHLCV 字段可显示为空或缺失标记，不报错）

#### Scenario: _last_info 字段集与 step 触发 bar 一致
- **WHEN** 协作者断言 `_last_info["c"]` 必须等于触发该 sig 的 bar 的 `close`
- **THEN** MUST 等值（MUST NOT 是上一桶或下一桶的 close）；同理 `o` / `h` / `l` / `v`
  与触发 bar 的同名字段一一对应
