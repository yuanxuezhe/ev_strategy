## Context

framework 已经提供 `Engine._process_bucket` (engine.py:60) 与 `vectorized_engine._aggregate_loop`
(vectorized_engine.py:106) 两条 `print(strategy.format_signal_line(ts, sig, getattr(strategy, "_last_info", None)))`
透传路径。本次不动 framework, 只让两个策略在 step 末尾把价格/OHLCV 写进 `_last_info`, hook 渲染时一并展示。

参考: `proposal.md` Why / What Changes; `specs/evtrade-architecture/spec.md` 新 R `Strategy signal line shows trigger bar context`。

## Goals / Non-Goals

**Goals:**
- 两个策略 (`filtered_mr` / `channel_deviation`) 信号行一行内同时含: 方向 (BUY/SELL) + 价格 + 触发 K 线 OHLCV
- step 与 hook 的字段契约 (`_last_info`) 显式稳定, 测试可锁定
- framework 零改动
- 视觉一致: 两个策略的 hook 输出格式对齐 (单行, 字段用 `|` 分隔, BUY/SELL 用前缀大写字母标记)

**Non-Goals:**
- 不改 framework / `Engine` / `run_vectorized` 任何文件
- 不改 `_last_info` 在 framework 侧的读取方式
- 不改 `signals-out` CSV schema (本就只写 (ts, sig))
- 不动 `VectorizedStrategy` 基类默认 hook (其他策略继续走默认)
- 不加新公共 helper (`fmt` 已存在, 见 `evtrade/primitives.py`, 直接复用)

## Decisions

### D1: hook 格式选 "单行 | 分隔", 不做 ASCII 表格

**为什么**: 现有 `channel_deviation.format_signal_line` 已是单行 | 分隔 (channel_deviation.py:236),
保持视觉一致; ASCII 表格在 verbose 模式下每根都打, 终端窗口一屏看不到几根信号, 不实用。

**备选**: 多行 (方向一行 / 价格一行 / K线一行) → 信息密度低, 浪费屏幕。
**备选**: JSON 单行 → 不可读, verbose 给人的是看不是 parse。

### D2: hook 内 `info.get(...)` 防御, 缺字段显示空字符串

**为什么**: spec Scenario `hook 用 info.get 不抛 KeyError` 强制要求。 `info.get(k, "")` 在
缺字段时返回空串, 渲染成 `o= h= l= c= v=`, 一眼能看出缺数据。

**备选**: `info.get(k, None)` 然后 `fmt(None)` → `fmt` 默认对 None 报异常, 不行。
**备选**: 缺字段抛 RuntimeError → 违反 spec Scenario。

### D3: 价格取 `info["price"]` 字段, 不直接用 `info["c"]`

**为什么**: `channel_deviation` 当前价格 = `close`, 但策略撮合用 close, 与信号触发价同义。
为语义清晰 (跟 hook 的 `price=` 字段名对应), 显式存 `price` 字段而不是让 hook 隐式取 `c`。

**备选**: hook 直接读 `c` → 字段耦合, 后续若策略引入 `vwap / midpoint` 等触发价会破坏现有 hook。
**否决**, 显式存 `price`。

### D4: 不提供 `strategies/util.py` 公共 hook 模板

**为什么**: 两个策略 hook 字段差异明显 (filtered_mr: trend/ADX/ATR; channel_deviation:
up/dw/dev/cash/position), 抽公共模板会让每边都带 if/else 处理对方字段, 反而比各写各的复杂。
两个策略分别覆写, 测试各自锁格式, 比共享模板简单。

**备选**: 共享 `BaseSignalLineRenderer` 基类 → over-engineered, 现在只有 2 个策略用得到, YAGNI。
**否决**。

### D5: 测试只锁 "sig!=0 时含方向词 + 价格 + OHLCV 字段名", 不锁字符串字面值

**为什么**: hook 输出格式允许演进 (调整分隔符 / 加 emoji / 加颜色); 锁字面值会让设计被
测试绑架。锁"必有 BUY/SELL 词"、"必有数字价格"、"必有 o=/h=/l=/c= 字段名"足够稳定。

**备选**: `assert format_signal_line(...) == "BUY >>xxx> | [ts] | price=..."` → 改格式要改测试。
**否决**。

## Risks / Trade-offs

- **`_last_info` 是策略私有属性, framework 通过 `getattr` 读** → 已存在惯例 (channel_deviation.py:227 用了一年); 不引入新 framework 字段。
- **未来加策略时容易忘**写 OHLCV → spec 加了 Scenario 锁, 新策略作者必须读这条 R; 同时 KB §06 同步。
- **verbose=true 时每根 sig 都打印**, 信息密度高; 但本来 sig 频率低 (策略 FSM 锁存), 不会刷屏。
- **测试用真实 strategy 实例 + 合成 bar 跑 step** → 测试比纯字符串断言稍慢, 但能锁住"字段与触发 bar 一致"的契约, 值得。

## Migration Plan

无。改动是策略展示层增量; 旧 verbose 输出 (默认 hook 的 `ts / sig / side`) 对其他策略仍生效。

回退 = `git revert` 单 commit; 无需数据迁移, 无需清缓存。

## Open Questions

（无）
