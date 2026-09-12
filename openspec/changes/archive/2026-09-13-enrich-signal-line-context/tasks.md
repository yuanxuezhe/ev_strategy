## 1. filtered_mr: 写 _last_info + 加 hook

- [x] 1.1 在 `evtrade/strategies/filtered_mr.py` step **入口**(bar 字段解包后、所有早返回路径前)写 `self._last_info = {"side": "", "price": float(c), "o": float(o), "h": float(h), "l": float(l), "c": float(c), "v": float(v)}`(side 空串是因为 sig 未知; step 末尾 `return state, sig` 之前覆写为 `"BUY"`/`"SELL"`/`""`)。验证:`grep -n "_last_info" evtrade/strategies/filtered_mr.py` 命中两处(入口 + 末尾覆写)。
  > **实施注**: 原 tasks 描述只写 step 末尾; 但实测发现 filtered_mr 有 early-return 路径(`is_strong_trend or is_high_vol` 时 return 0), 末尾写入会让 early-return 时 `_last_info` 持旧值, 违反 spec Scenario `_last_info 字段集与 step 触发 bar 一致`。改为入口先写一次 + 末尾覆写 side, 所有路径都正确。
- [x] 1.2 同文件新增 `format_signal_line(self, ts, sig, info=None) -> str`, 单行 `|` 分隔, 格式 `SIDE >>xxx> | [ts] | price=... o=... h=... l=... c=... v=...`(空 sig 时 `SIDE` 为空 + 16 空格, 与 channel_deviation 视觉一致); `info.get(...)` 防御。验证:`grep -n "format_signal_line" evtrade/strategies/filtered_mr.py` 命中 1 处。

## 2. channel_deviation: 扩展 _last_info + 改 hook

- [x] 2.1 在 `evtrade/strategies/channel_deviation.py` step 末尾 (line 227 `_last_info = {...}`) 增加 `side / price / o / h / l / c / v` 7 个字段; 原 `up / dw / low_dev / high_dev / cash / position` 保留。验证:`grep -A20 "_last_info" evtrade/strategies/channel_deviation.py` 含 13 个字段。
- [x] 2.2 同文件 `format_signal_line` (line 232) 改格式: 价格 + OHLCV 放在行首/显眼位置, up/dw/dev 字段保留(向后兼容 spec Scenario)。验证:运行 `python -m evtrade backtest --strategy channel_deviation --synthetic-days 30 --device cpu --verbose`, 终端打印含 `BUY` / `SELL` 词 + 价格 + 4 个 OHLC 字段名 + 至少 1 个 `dev` 字段名。
- [x] 2.3 hook 内统一用 `info.get(k, "")`, 杜绝 `info[k]` 直接下标。验证:`grep -n "info\[" evtrade/strategies/channel_deviation.py` 命中 0 行 (`info.get` 可任意)。

## 3. 测试

- [x] 3.1 新建 `tests/test_format_signal_line_enrich.py`:
  - `test_filtered_mr_format_signal_line_buy_contains_all`: 用真实 filtered_mr 实例, 调用 `format_signal_line(ts, +1, {"side":"BUY", ...OHLCV})`, assert `out` 含 `BUY` + `price=` + `o=` + `h=` + `l=` + `c=` + `v=`。
  - `test_filtered_mr_format_signal_line_sell_contains_all`: 同上, sig=-1, 含 `SELL`。
  - `test_filtered_mr_format_signal_line_empty_info_no_keyerror`: `format_signal_line(ts, 1, {})` 不抛 + 返回字符串含 `BUY`(缺字段 → 空字符串渲染, 不报错)。
  - `test_filtered_mr_format_signal_line_no_info_arg_no_keyerror`: `info=None` 也不抛。
  - `test_channel_deviation_format_signal_line_contains_price_and_ohlcv`: 构造 sig=+1 的 info, assert 输出含 `BUY` + `price=` + 4 个 OHLC 字段名 + 至少一个 `dev` 字段(向后兼容)。
  - `test_channel_deviation_format_signal_line_empty_info_no_keyerror`: 同 filtered_mr。
  - `test_filtered_mr_last_info_c_equals_trigger_bar_close`: 跑真实 `filtered_mr.step(state, bar, params)`, 断言 `strategy._last_info["c"] == bar["c"]`(spec Scenario `_last_info 字段与触发 bar 一致`)。
  - `test_hook_no_info_subscript_access[filtered_mr]` / `[channel_deviation]` (参数化): 源扫描 hook 函数体内不能出现 `info[` 下标访问。
  验证:`uv run pytest tests/test_format_signal_line_enrich.py -v` 9 通过。
- [x] 3.2 跑全量 `uv run pytest -q` 验证无回归。验证:166 passed, 1 skipped, 0 failed(原 157 + 新增 9 = 166)。

## 4. spec 落地

- [x] 4.1 把 `openspec/changes/enrich-signal-line-context/specs/evtrade-architecture/spec.md` 的 `ADDED Requirements`(整个 `### Requirement: Strategy signal line shows trigger bar context` 段含 4 个 Scenario)合入 `openspec/specs/evtrade-architecture/spec.md`, 插在 `## Requirements` 末尾(`### Requirement: Market data DB connection has a sane default` 之后)。KB 对应表加 1 行。验证:`grep -n "Strategy signal line shows trigger bar context" openspec/specs/evtrade-architecture/spec.md` 命中 2 处(Requirement + KB 表)。
- [x] 4.2 同步 `kbs/06-交易策略详解.md` §8 策略展示 hook 段: 加 2026-09-13 约定块, 含 `_last_info` 字段约定 + `info.get` 防御 + sig!=0 行内必含方向词 + 价格 + 4 个 OHLC 字段名 + channel_deviation 向后兼容 + 锁定 spec + 测试。验证:`grep -n "_last_info\|side.*price.*o.*h.*l.*c.*v\|BUY.*SELL.*OHLCV" kbs/06-交易策略详解.md` 命中。

## 5. 验证清单

- [x] 5.1 `openspec validate --specs` 通过(主 spec 新 R 已合入)。验证:1 passed, 0 failed。
- [x] 5.2 烟测两个策略(verbose 模式):
  - `python -m evtrade backtest --strategy filtered_mr --synthetic-days 30 --device cpu --verbose | head -10` → 输出含 `BUY >>xxx>` / `SELL >>xxx>` + `price=` + `o=/h=/l=/c=/v=` ✓
  - `python -m evtrade backtest --strategy channel_deviation --synthetic-days 30 --device cpu --verbose | head -5` → 输出含 `BUY/SELL` + 价格 + 5 OHLCV 字段 + 向后兼容字段 `UP=/DW=/low_dev=/high_dev=/cash=/pos=` ✓
- [x] 5.3 `ma_crossover` 走默认 hook 不受影响: `python -m evtrade backtest --strategy ma_crossover --synthetic-days 30 --device cpu --verbose | head -5` → 行内只含 `sig=+1 BUY`(无 `price=`,走 `VectorizedStrategy.format_signal_line` 默认实现)✓

## 6. archive

- [x] 6.1 任务 1-5 全绿; user 决定 baseline commit 自己处理, 我不跑 git。 跑 `openspec archive enrich-signal-line-context --yes` 归档; change 移入 `openspec/changes/archive/`(delta 已合入主 spec)。
