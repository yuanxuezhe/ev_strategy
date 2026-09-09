# Design: unify-metrics-units — metrics 字段单位统一

## 1. 问题复盘

CLI 打印 `最大回撤 (逐bar盯市)   : 15734173.45%` 看似是 `:.2%` 格式错，但根因是 **字段单位在 spec 里没钉死**，三层都按各自理解实现：

| 层 | 当前实现 | 期望实现 |
|---|---|---|
| `vectorized_engine._execute_trades` (line 198) | `max_drawdown = -max_dd` 资金元 | `(-max_dd) / init_equity` 小数 |
| `metrics.summarize` (line 131) | `calmar = cagr / max_drawdown` (cagr=%, mdd=元) | `(cagr/100) / max_drawdown` (无量纲) |
| `cli._run_backtest` (line 240) | `:.2%` 把元当小数 | 字段已改小数，格式保留 |
| `sweep._aggregate` (line 351) | `mdd(元) <= max_mdd(1.0 元)` 永远过 | mdd(小数) ≤ 默认 1.0(=100%=不限) |

证据：`kbs/13-绩效评估与鲁棒选参框架.md:39` 的"实盘建议 0.15"是**小数语义**的铁证——只有 `max_drawdown` 是小数，0.15 才有意义。

## 2. 不挂现有 change 的理由

3 个归档 change 都不合适：
- `remove-dead-imports` — 死代码清理，无关
- `decouple-indicators-from-framework` — 指标与 framework 解耦，已归档
- `unify-strategy-contract` — DSL 收口，已归档；虽然本次也动 `metrics.py`，但**与"统一策略契约"是两个原子变更**，硬挂会破坏原子性原则

本次新建 `unify-metrics-units`，专门处理"指标字段语义"。

## 3. 字段单位约定表（写进 spec）

| 字段 | 单位 | 实现位置 |
|---|---|---|
| `cagr` | % | `metrics.py:109` 现有 `*100.0` 保留 |
| `max_drawdown` | 小数 (0~1.0+) | **新**：`vectorized_engine.py:198` 改 `-max_dd/init_equity` |
| `sharpe_excess` / `sortino_excess` | 无量纲 | `metrics.py:122/127` 保留 |
| `calmar` | 无量纲 | **改**：`metrics.py:131` `cagr/100 / max_drawdown` |
| `max_dd_days` | 自然日 | `metrics.py:141` 保留 |
| `final_equity` / `baseline` / `excess` / `final_cash` / `turnover` | 元 | 保留 |
| `x_mdd` | 小数 | 同 `max_drawdown`，改 `metrics.py:92-94` 也走百分比除法 |

## 4. 代码改动

### 4.1 `evtrade/core/vectorized_engine.py`

`_execute_trades` 末尾：

```python
n = len(close_np)
init_equity = init_cash + init_position * (float(close_np[0]) if n > 0 else 0.0)
# ... 循环里 running_max / dd 不变 ...

return {
    # ... 其它字段不变 ...
    "max_drawdown": ((-max_dd) / init_equity)
        if (max_dd < 0.0 and init_equity > 0) else 0.0,
    "equity_curve": equity_curve,
    "baseline_curve": baseline_curve,
}
```

注意 `init_equity` 在 `n == 0` 时退化为 0.0，对应 `max_drawdown=0.0`（spec 已说"无数据时填 0.0"）。

### 4.2 `evtrade/core/metrics.py`

两处：

**第一处：`x_mdd` 也用百分比语义**（与 `max_drawdown` 一致）：

```python
# line 92-94
running_peak = np.maximum.accumulate(eq)
drawdown = (eq - running_peak) / running_peak   # 改：除以 peak（占 peak 的小数）
is_dd = drawdown < 0.0
max_drawdown = float(-drawdown.min()) if is_dd.any() else 0.0
```

等等 — `metrics.summarize` 自己也算 `max_drawdown`（line 91-94），且与 `vectorized_engine._execute_trades` 计算结果应该一致。改 engine 的归一化后，`metrics.summarize` 也必须改相同公式，否则两个值会**对不上**。

正确做法：
- `vectorized_engine._execute_trades` 现在 line 198 不再算 `max_drawdown`（删除该字段），把它完全交给 `metrics.summarize` 算（因为只有 metrics 有完整 `years / bars_per_year / dd_days` 上下文）。
- `metrics.summarize` line 91-94 用 `(peak - eq) / init_equity` 算，**init_equity = eq[0]**。

实现：

```python
# vectorized_engine.py:195-200
return {"cash": ..., "position": ..., "last_price": ...,
        "n_trades": ..., "n_buy": ..., "n_sell": ..., "turnover": ...,
        "trades": trades,
        # 删 "max_drawdown": -max_dd ...
        "equity_curve": equity_curve, "baseline_curve": baseline_curve}
```

```python
# metrics.py:85-94
if equity_curve is not None and len(equity_curve) >= 2 and years > 0:
    eq = np.asarray(equity_curve, dtype=np.float64)
    init_equity = float(eq[0])  # 期初权益
    running_peak = np.maximum.accumulate(eq)
    drawdown = (eq - running_peak) / init_equity  # 占 init 小数
    is_dd = drawdown < 0.0
    max_drawdown = float(-drawdown.min()) if is_dd.any() else 0.0
    ...
```

这样 `vectorized_engine` 和 `metrics.summarize` 都从 `eq[0]` 拿分母，单位一致。

**第二处：calmar 跨单位修齐**

```python
# metrics.py:130-131 (修)
if max_drawdown > 0:
    calmar = (cagr / 100.0) / max_drawdown  # cagr 是 %, mdd 是小数 -> 无量纲
```

**第三处：x_mdd 也归一化**

```python
# metrics.py: 后续 (excess_rets 累计后)
cum_ex = np.cumsum(excess_rets)              # 累计超额（元单位）
x_mdd_peak = np.maximum.accumulate(cum_ex)
x_mdd_arr = (cum_ex - x_mdd_peak) / max(init_baseline, 1e-9)
x_mdd = float(-x_mdd_arr.min()) if len(x_mdd_arr) else 0.0
```

### 4.3 `evtrade/cli.py`

```python
# line 237-242
print(f"超额 Sharpe           : {s['sharpe_excess']:+.3f}")
print(f"超额 Sortino          : {s['sortino_excess']:+.3f}")
print(f"Calmar (年化/回撤)    : {s['calmar']:+.3f}")            # 无量纲
print(f"最大回撤 (逐bar盯市)   : {s['max_drawdown']:+.2%}")     # 小数 -> %
print(f"最大回撤持续天数       : {s['max_dd_days']:.1f} 天")
```

`max_drawdown` 改小数后 `:.2%` 自动正确。Calmar 直接 `:.3f`（无量纲）。这两行的 label 文案稍调一下让单位显式：

```python
print(f"Calmar (年化%/回撤%)   : {s['calmar']:+.3f}")   # 现在单位都是 %，calmar 无量纲
```

实际 calmar 是无量纲比率，label 写 "(年化/回撤)" 更准确。保留现状即可。

### 4.4 `evtrade/core/sweep.py`

`filter_pass` 默认 1.0 现在语义=100% 回撤（实际不限），**不需要改**。但加注释让未来读者明白：

```python
# sweep.py:351
# 注: max_drawdown 现在是占初始权益的小数; --max-mdd 默认 1.0 = 100% = 不限
# 实盘建议 --max-mdd 0.15 (= 拒回撤 > 15% 的策略)
```

### 4.5 测试 `tests/test_metrics_units.py`

新增 4 个 test：

```python
def test_max_drawdown_unit_is_fraction():
    """max_drawdown 必须是占初始权益的小数,不是元"""
    eq = np.array([200000.0, 180000.0, 120000.0, 150000.0])
    summary = summarize(_state(200000.0, 0.0), 200000.0, 0.0,
                        equity_curve=eq, baseline_curve=eq*1.0,
                        first_ts=..., last_ts=...)
    assert 0.0 <= summary["max_drawdown"] <= 1.5
    assert abs(summary["max_drawdown"] - 0.30) < 1e-6   # (180-120)/200

def test_cagr_is_percent():
    """cagr 必须是百分数,不是小数"""
    eq = np.array([100.0, 110.0, 121.0])  # 2 年翻 1.21 倍 -> 10% CAGR
    summary = summarize(..., equity_curve=eq, years=2.0)
    assert abs(summary["cagr"] - 10.0) < 1e-6   # %, 不是 0.10

def test_calmar_cross_unit_fixed():
    """cagr=10%, mdd=0.20 -> calmar=0.5"""
    # 构造一个 cagr=10%, 最大回撤=0.20 的曲线
    summary = ...
    assert abs(summary["calmar"] - 0.5) < 1e-3

def test_cli_max_drawdown_format():
    """CLI 打印 max_drawdown 必须是合理百分数量级,不是 1.5e7%"""
    # 用 subprocess 跑 backtest 抓 stdout, regex 检查
```

## 5. KB 改动

### 5.1 `kbs/13-绩效评估与鲁棒选参框架.md`

行 23-29 表格后插"字段单位约定表"：

```markdown
**字段单位约定**（2026-09-09 钉死）：

| 字段 | 单位 | 公式形式 |
|---|---|---|
| cagr | % | (eq[-1]/eq[0])^(1/years) - 1，再 ×100 |
| max_drawdown | 占初始权益小数 (0~1.0+) | max(0, peak-trough) / init_equity |
| sharpe_excess / sortino_excess | 无量纲（年化） | mean/std * sqrt(bars_per_year) |
| calmar | 无量纲比率 | cagr(小数) / max_drawdown(小数) |
| max_dd_days | 自然日 | n_dd_bars / bars_per_day |
| final_equity / baseline / excess | 元 | cash + position * price |
```

并改行 39 注释：`--max-mdd 默认 1.0 即不设限` → 强调"1.0 = 100% = 不限"。

### 5.2 `kbs/09-引擎Engine与主流程.md`

第 4 节 `print_summary` 后（行 90 之后）插一段：

```markdown
> **字段单位**（详见 kbs/13 §1）：cagr / max_drawdown / excess_pct / ann_excess_pct 单位为 %（CLI 用 `:.2%`）；
> final_equity / baseline / excess / final_cash / turnover 为元（CLI 用 `:.2f`）；
> calmar 为无量纲比率（CLI 用 `:.3f`）；max_dd_days 为自然日。
```

## 6. 测试矩阵

| 测试 | 锁定目标 |
|---|---|
| `tests/test_metrics_units.py::test_max_drawdown_unit_is_fraction` | mdd 是小数，不是元 |
| `tests/test_metrics_units.py::test_cagr_is_percent` | cagr 是 %，不是小数 |
| `tests/test_metrics_units.py::test_calmar_cross_unit_fixed` | calmar 同单位 |
| `tests/test_metrics_units.py::test_cli_max_drawdown_format` | CLI 输出合理量级 |
| `tests/test_sweep_v2.py` 现有 | sweep filter_pass 与新 mdd 单位兼容 |
| `tests/test_metrics_v2.py` 现有 | metrics 字段集不变 |

## 7. 端到端验证

| 命令 | 期望 |
|---|---|
| `uv run python -m evtrade backtest --strategy channel_deviation --device cpu --period 5m --start 20250101 --end 20260903 --no-sleep` | max_drawdown 打印 `+X.XX%`（合理量级），Calmar 打印 `+X.XXX` |
| `uv run python -m evtrade backtest --strategy ma_crossover --device cpu ...` | 同上 |
| `uv run python -m evtrade sweep --strategy channel_deviation --device cpu --max-mdd 0.15 --grid low1=0.5,1.0 --splits 20250601` | filter_pass 列真正拒掉 >15% mdd 的参数 |
| `uv run pytest -q` | 全部 PASS（含新增 test_metrics_units.py） |

## 8. 风险

| 风险 | 缓解 |
|---|---|
| 改 `max_drawdown` 语义后，旧 sweep 结果的 `filter_pass` 标记可能变 | spec 已说明这是 bug 修复；预期会变，旧 sweep 结果重新跑即可 |
| `calmar` 数值量级变化（10% / 0.20 = 0.5 vs 旧版 10/200000 ≈ 5e-5） | 同上；KB 表格行 26 写"calmar = cagr / max_drawdown"已隐含无单位，修复后符合 KB |
| `x_mdd` 也改成小数后，KB 表格行 29 描述要跟着改 | KB 改动里同步 |