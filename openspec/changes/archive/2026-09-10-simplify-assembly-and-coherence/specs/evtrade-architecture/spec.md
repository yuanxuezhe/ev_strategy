# Spec Delta: evtrade-architecture

## ADDED Requirements

### Requirement: Sweep grid accepts only effective axes

sweep 的 `--grid` 网格轴 MUST 仅接受两类键：引擎轴（`period / trade_qty / scale /
buy_pct / sell_pct`）与策略 `params_spec` 声明的参数名。MUST NOT 接受"下游从不读取"
的轴（2026-09-10 `all_in` 曾是此类——被解析为 bool 后无任何消费者，静默空转；已删）。
资金模式网格化 MUST 经 `--grid buy_pct=...` / `sell_pct=...` 表达。`--grid all_in=...`
MUST 报 `ValueError: 不支持的网格参数 'all_in'; 可用: [...]`。backtest 子命令的
`--all-in` flag 不受影响（等价 `--buy-pct 1.0 --sell-pct 1.0`）。

#### Scenario: all_in 网格轴被拒
- **WHEN** 执行 `python -m evtrade sweep --grid all_in=true,false ...`
- **THEN** 报 `ValueError`（提示不支持的网格参数及可用列表），MUST NOT 静默跑完

#### Scenario: 资金模式经 buy_pct/sell_pct 网格化
- **WHEN** 执行 `python -m evtrade sweep --grid buy_pct=0.5,1.0 --grid sell_pct=0.5,1.0 ...`
- **THEN** 正常展开笛卡尔积并逐组回测（`all_in` 语义 = buy_pct=sell_pct=1.0 的组合）

### Requirement: reconcile legs receive identical effective funding

`replay --against-ref`（`core.replay.reconcile`）MUST 让 vectorized 腿与 Engine 腿收到
**相同的有效成交参数**。`all_in` 的解析（→ `buy_pct = sell_pct = 1.0`）MUST 在
CLI/调用边界对**两腿统一**生效，MUST NOT 只作用于 Engine 腿（`SimulatedExecutor(all_in=)`
）而让 vectorized 腿停留在原始 `buy_pct / sell_pct`——后者会使两腿成交流不可比、
对账必然 FAIL（2026-09-10 前 bug）。

#### Scenario: all-in 对账可 PASS
- **WHEN** `python -m evtrade replay --log <log> --strategy channel_deviation --against-ref --all-in ...`
  且两腿在统一 buy_pct=sell_pct=1.0 下运行
- **THEN** 成交对账（ts/side/qty/price 逐笔）MUST 不因资金模式不对称而 FAIL

### Requirement: replay_vectorized signal trajectory carries aligned timestamps

`core.replay.replay_vectorized` MUST 返回与 `"sig"` **逐元素对齐**的桶级时间戳键
`"ts"`（与 `trades` 的 ts 同源，即 `_aggregate_buckets` 产出的桶 ts），使
`--signals-out` 能写出真实的 (ts, sig) 行。MUST NOT 以
1m bar 数组索引去取桶级 sig（桶数 < bar 数，越界或错位）——2026-09-10 前
`--signals-out` 在 replay 路径即以 1m bar 数索引桶级 sig 数组（latent bug：
warmup 存在时越界崩溃，无 warmup 时写出大量 sig=0 的错位行）。

#### Scenario: sig 与 ts 等长且对齐
- **WHEN** `replay_vectorized(...)` 返回 `out`
- **THEN** `len(out["ts"]) == len(out["sig"])`；每对 `(ts, sig)` 中 ts 与该笔
  `trades` 用的桶 ts 同源

#### Scenario: replay --signals-out 写出桶级轨迹
- **WHEN** `python -m evtrade replay --log <log> --signals-out <out.csv> --warmup-until <ymd> ...`
- **THEN** 正常完成，`<out.csv>` 行数 = 桶数（非 1m bar 数），无越界异常
