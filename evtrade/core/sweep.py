from __future__ import annotations
"""参数并发扫描 + 鲁棒选参框架 (kbs/13)

流程 (对每组参数):
  1. WFO 多窗回测: train 窗 + K 个滚动 test 窗 (--splits d1,d2,...)
     窗口 i 的数据截断到其结束日, 预热用其开始日之前的全部数据 -> 指标就绪、
     锁存状态全新起算, 与实盘在该日上线的情形一致。
  2. 每窗绩效 (kernel.summarize): 年化扣费超额
     ann_net = excess_pct/years - turnover×fee/baseline/years×100
  3. 聚合: ann_net_min (最差 test 窗, 主判据) / ann_net_mean / pos_ratio /
     sharpe_min / 邻域衰减 S (参数平原: 邻居平均绩效相对自己的衰减, [0,1])
  4. 复合评分: score = ann_net_min / (1 + λ·S)   (ann_net_min<=0 时原样透传)
     λ 由 --score-lambda 控制; 越高越偏向"参数平原中心"。
  5. 帕累托标记: (ann_net_mean, sharpe_min) 双目标非支配点。
  6. 硬过滤标记 filter_pass: 各 test 窗笔数>=min_trades 且 权益回撤<=max_mdd。

并行模型: 一组参数一个独立 KernelState; nogil 线程池 (cpu) 或按窗单 launch (gpu)。
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .kernel import make_state, run_backtest, summarize
from .kernel_dsl import run_one_dsl, strategy_has_dsl
from ..frozen.timeutils import resolve_period_seconds as period_seconds

GRID_KEYS = ("low1", "low2", "high1", "high2", "tf1", "period", "trade_qty", "scale",
             "buy_pct", "sell_pct", "all_in")
# 网格扫描允许的额外 key (按 strategy_name 的 params_spec 动态加入)
_GRID_EXTRA_KEYS: set[str] = set()


def register_grid_keys(keys: set[str]) -> None:
    """新策略注册其允许的 grid key; sweep 解析时合并"""
    _GRID_EXTRA_KEYS.update(keys)


def parse_grid(specs: list, extra_keys: set[str] | None = None) -> list[dict]:
    """["low1=1.0,1.5", "high2=0.3,0.5"] -> 笛卡尔积参数组合列表

    extra_keys: 额外允许的 grid key (由策略的 params_spec 提供)。
                 None 时只允许 GRID_KEYS 白名单; 传非空集合时扩展。
    """
    import itertools
    allowed = set(GRID_KEYS) | set(extra_keys or set()) | _GRID_EXTRA_KEYS
    axes = []
    for s in specs:
        name, _, values = str(s).partition("=")
        name = name.strip()
        if name not in allowed:
            raise ValueError(f"不支持的网格参数 {name!r}; 可用: {sorted(allowed)}")
        vals = [v.strip() for v in values.split(",") if v.strip()]
        if not vals:
            raise ValueError(f"网格参数 {name} 无取值: {s!r}")
        axes.append((name, vals))
    keys = [k for k, _ in axes]
    combos = []
    for values in itertools.product(*[v for _, v in axes]):
        combo = {}
        for k, raw in zip(keys, values):
            if k == "tf1":
                combo[k] = int(raw)
            elif k == "period":
                combo[k] = raw
            elif k == "all_in":
                combo[k] = raw.strip().lower() in ("1", "true", "yes", "y", "t")
            else:
                combo[k] = float(raw)
        combos.append(combo)
    return combos

_EMPTY_SIG = np.empty(0, np.int8)
_EMPTY_F = np.empty(0, np.float64)


def run_one(bars: dict, period: str, warmup_until: int, tf1: int,
            low1: float, low2: float, high1: float, high2: float,
            init_cash: float, init_position: float, trade_qty: float,
            scale: float = 1.0,
            buy_pct: float = 0.0, sell_pct: float = 0.0,
            all_in: bool = False) -> dict:
    """单组参数单窗回测 (内核), 返回 kernel.summarize 口径的绩效字典"""
    st = make_state(period=period, warmup_until=warmup_until, tf1=tf1,
                    low1=low1, low2=low2, high1=high1, high2=high2,
                    init_cash=init_cash, init_position=init_position,
                    trade_qty=trade_qty, scale=scale,
                    buy_pct=buy_pct, sell_pct=sell_pct, all_in=all_in)
    run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"], _EMPTY_SIG, _EMPTY_F, _EMPTY_F)
    return summarize(st)


def _run_window(bars: dict, p: dict, warmup_until: int) -> dict:
    return run_one(bars, p["period"], warmup_until, p["tf1"],
                   p["low1"], p["low2"], p["high1"], p["high2"],
                   p["init_cash"], p["init_position"], p["trade_qty"],
                   p.get("scale", 1.0),
                   buy_pct=p.get("buy_pct", 0.0),
                   sell_pct=p.get("sell_pct", 0.0),
                   all_in=p.get("all_in", False))


def _empty_metrics(init_cash: float, init_position: float) -> dict:
    """空数据时返回零交易占位指标 (避免 flush IndexError)"""
    return {
        "final_price": 0.0, "n_trades": 0, "n_buy": 0, "n_sell": 0,
        "final_cash": init_cash, "final_position": init_position,
        "final_equity": init_cash, "baseline": init_cash,
        "excess": 0.0, "excess_pct": 0.0, "years": 0.0,
        "ann_excess_pct": 0.0,
        "sharpe_excess": 0.0, "sortino_excess": 0.0,
        "cagr": 0.0, "calmar": 0.0,
        "max_dd_days": 0.0, "max_dd_recovered": True,
        "x_mdd": 0.0, "max_drawdown": 0.0, "turnover": 0.0,
    }


def run_one_general(bars: dict, period: str, warmup_until: int,
                     init_cash: float, init_position: float, trade_qty: float,
                     scale: float = 1.0,
                     buy_pct: float = 0.0, sell_pct: float = 0.0,
                     all_in: bool = False,
                     strategy_name: str = "channel_deviation",
                     strategy_params: dict | None = None) -> dict:
    """通用策略单组参数单窗回测 (走参考引擎, 不依赖 kernel._strategy_check)

    与 run_one 的区别:
      - 不依赖 channel_deviation 的 4 个固定参数 (low1/low2/high1/high2/tf1)
      - 任何 strategies/ 子包的策略都可跑 (channel_deviation / breakout / ...)
      - 走参考引擎 (慢约 500x), 不进 numba/CUDA 加速路径
      - 用于参数空间探索 / 新策略验证
    """
    from ..frozen.account import Account
    from ..frozen.aggregator import BarAggregator
    from ..frozen.models import Bar
    from .engine import Engine
    from ..execution.base import SimulatedExecutor
    from ..strategies import get_strategy

    class _ArrFeed:
        def __init__(s, bs): s.bs = bs
        def stream(s):
            for i in range(len(s.bs["stime"])):
                yield Bar(stime=str(int(s.bs["stime"][i])), code="SYN",
                          open=float(s.bs["open"][i]),
                          high=float(s.bs["high"][i]),
                          low=float(s.bs["low"][i]),
                          close=float(s.bs["close"][i]),
                          volume=int(s.bs["volume"][i]))

    account = Account(cash=init_cash, position=init_position)
    executor = SimulatedExecutor(account, qty=trade_qty, verbose=False,
                                  scale=scale,
                                  buy_pct=buy_pct, sell_pct=sell_pct,
                                  all_in=all_in)
    strategy = get_strategy(strategy_name, params=strategy_params or {})
    aggregator = BarAggregator(
        period_seconds(period) if isinstance(period, str) else period,
        on_bars=None,
        warmup_until=str(warmup_until) if warmup_until else None,
    )
    tf1 = (strategy_params or {}).get("tf1", 21)
    eng = Engine(_ArrFeed(bars), aggregator, strategy, executor, tf1=tf1, verbose=False)
    # 空数据安全 (sweep 单窗可能给到空 bars, 已知 flush 会 IndexError, 跳过即可)
    if len(bars["stime"]) == 0:
        return _empty_metrics(init_cash, init_position)
    eng.run()

    final_price = executor.account.last_price
    eq = executor.account.equity(final_price)
    baseline = executor.account.baseline_equity(final_price)
    diff = eq - baseline
    pct = (diff / baseline * 100.0) if baseline else 0.0

    years = 0.0
    if len(bars["stime"]) >= 2:
        from .kernel import encoded_to_epoch
        e0 = encoded_to_epoch(int(bars["stime"][0]))
        e1 = encoded_to_epoch(int(bars["stime"][-1]))
        years = (e1 - e0) / (365.25 * 86400.0)

    ann_excess = (pct / years) if years > 0 else 0.0
    return {
        "final_price": final_price,
        "n_trades": len(account.trades),
        "n_buy": sum(1 for t in account.trades if t["side"] == "BUY"),
        "n_sell": sum(1 for t in account.trades if t["side"] == "SELL"),
        "final_cash": account.cash,
        "final_position": account.position,
        "final_equity": eq,
        "baseline": baseline,
        "excess": diff,
        "excess_pct": pct,
        "years": years,
        "ann_excess_pct": ann_excess,
        "sharpe_excess": 0.0,
        "sortino_excess": 0.0,
        "cagr": 0.0,
        "calmar": 0.0,
        "max_dd_days": 0.0,
        "max_dd_recovered": True,
        "x_mdd": 0.0,
        "max_drawdown": 0.0,
        "turnover": sum(t["qty"] * t["price"] for t in account.trades),
    }


def _ann_net(m: dict, fee_bp: float) -> float:
    """年化扣费超额% = excess_pct/years - turnover×fee/baseline/years×100"""
    fee = fee_bp / 10000.0
    if m.get("years", 0) > 0 and m.get("baseline", 0) > 0:
        return (m["excess_pct"] / m["years"]
                - m["turnover"] * fee / m["baseline"] / m["years"] * 100.0)
    return 0.0


def _neighbor_decay(combos: list[dict], values: np.ndarray) -> np.ndarray:
    """邻域衰减 S ∈ [0,1]: 邻居(每轴±1格)平均绩效相对自身的相对衰减。

    self<=0 或无邻居 -> 0; 邻居均值越低 (孤立尖峰) S 越接近 1。
    """
    n = len(combos)
    out = np.zeros(n)
    if n == 0:
        return out
    keys = [k for k in combos[0]
            if isinstance(combos[0][k], (int, float, str, bool))
            and len({c.get(k) for c in combos}) > 1
            and not isinstance(combos[0][k], dict)]
    if not keys:
        return out
    # 只保留可哈希类型 (排除 dict 等)
    def _safe_items(c):
        return tuple(sorted((k, v) for k, v in c.items()
                           if isinstance(v, (int, float, str, bool))))
    lookup = {_safe_items(c): i for i, c in enumerate(combos)}
    axis_vals = {k: sorted({c[k] for c in combos}) for k in keys}
    for i, c in enumerate(combos):
        if values[i] <= 0:
            continue
        nb = []
        for k in keys:
            vals = axis_vals[k]
            pos = vals.index(c[k])
            for p2 in (pos - 1, pos + 1):
                if 0 <= p2 < len(vals):
                    nc = dict(c)
                    nc[k] = vals[p2]
                    j = lookup.get(_safe_items(nc))
                    if j is not None:
                        nb.append(values[j])
        if nb:
            out[i] = min(max(1.0 - float(np.mean(nb)) / values[i], 0.0), 1.0)
    return out


def _pareto_flag(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """双目标 (均最大化) 非支配标记"""
    order = np.lexsort((-b, -a))          # a 降序, 同 a 时 b 降序
    flags = np.zeros(len(a), dtype=bool)
    best_b = -np.inf
    for idx in order:
        if b[idx] > best_b:
            flags[idx] = True
        best_b = max(best_b, b[idx])
    return flags


def sweep(bars: dict, base: dict, combos: list[dict],
          split_ymd: str = None, n_workers: int = None, verbose: bool = True,
          device: str = "cpu", splits: list = None, fee_bp: float = 5.0,
          lam: float = 1.0, min_trades: int = 30, max_mdd: float = 1.0,
          strategy_name: str = "channel_deviation"):
    """并发扫描 + 鲁棒评分; 返回 pandas.DataFrame (按 score 降序)

    base 键: start/period/trade_qty/init_cash/init_position + strategy_params
    combos:  parse_grid 的输出, 覆盖 base 中的对应键 (策略参数键)。
    splits:  滚动 WFO 分割日列表 ["20260101","20260401"]; 1 个时窗口名为 train/test
             (兼容旧 --split), 多个时为 train/test1..testK。None=单窗 (列名无前缀)。

    strategy_name: 策略 key (默认 channel_deviation)。路径选择 (步骤 3 通用化):
      - channel_deviation                  -> 冻结 numba 内核 (_run_window)
      - 其他带 DSL docstring 的策略        -> DSL 特化 numba 内核 (run_one_dsl);
            device="gpu" 时走通用 CUDA kernel (cuda_sweep_window_generic)
      - 无 DSL 的策略 (如 breakout)        -> 参考引擎 (run_one_general, 慢约 500x)
    """
    import pandas as pd

    if split_ymd and not splits:
        splits = [split_ymd]
    start = base["start"]
    if splits:
        sp = sorted(str(s) for s in splits)
        # 单 split: 1 个 test (从 start 到 split 之后), 列名 train_/test_ 二者皆有
        # 多 splits: train + K 个 test (test1_/test2_/...)
        if len(sp) == 1:
            wins = [("train", None, start), ("test", None, sp[0])]
        else:
            wins = [("train", sp[0], start)]
            for i, s in enumerate(sp):
                end_b = sp[i + 1] if i + 1 < len(sp) else None
                wins.append((f"test{i + 1}", end_b, s))
    else:
        wins = [("full", None, start)]

    def window_bars(end_ymd):
        if end_ymd is None:
            return bars
        mask = bars["stime"] < int(end_ymd) * 1_000_000
        return {k: v[mask] for k, v in bars.items()}

    win_data = [(nm, window_bars(e), int(w) * 1_000_000) for nm, e, w in wins]

    # 通用策略: 把 grid key 覆盖到 base["params"] 上; 缺失的参数继承 base
    #   语义: --params 提供基础参数, --grid 在指定 key 上扫描, 未指定 key 沿用基础值
    if strategy_name != "channel_deviation":
        base_params = dict(base.get("params", {}))
        new_combos = []
        for c in combos:
            # 用临时 dict 合并, 不修改原 c (避免 list(c) + 赋值 c[k] 互相污染)
            merged_params = {**base_params, **c}
            new_c = dict(c)            # copy of grid keys
            new_c["params"] = merged_params
            new_combos.append(new_c)
        combos = new_combos
    params_list = [{**base, **c} for c in combos]

    t0 = time.perf_counter()
    metrics = [None] * len(win_data)
    use_general = (strategy_name != "channel_deviation")
    # DSL 策略 -> numba/CUDA 快路径; 无 DSL -> 参考引擎兜底
    dsl_fast = use_general and strategy_has_dsl(strategy_name)
    if device == "gpu" and not use_general:
        from .gpu import cuda_sweep_window
        for wi, (nm, wb, warm) in enumerate(win_data):
            metrics[wi] = cuda_sweep_window(wb, params_list, warm)
    elif device == "gpu" and dsl_fast:
        from .gpu import cuda_sweep_window_generic
        for wi, (nm, wb, warm) in enumerate(win_data):
            metrics[wi] = cuda_sweep_window_generic(wb, params_list, warm,
                                                    strategy_name=strategy_name)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {}
            for wi, (nm, wb, warm) in enumerate(win_data):
                for ci, p in enumerate(params_list):
                    if use_general and dsl_fast:
                        # DSL 策略: DSL 特化 numba 内核 (与 run_one 同口径)
                        futs[(wi, ci)] = ex.submit(
                            run_one_dsl, wb, p["period"], warm,
                            p["init_cash"], p["init_position"], p["trade_qty"],
                            p.get("tf1", 21),
                            p.get("scale", 1.0),
                            p.get("buy_pct", 0.0), p.get("sell_pct", 0.0),
                            p.get("all_in", False),
                            strategy_name=strategy_name,
                            strategy_params=p.get("params", {}))
                    elif use_general:
                        # 无 DSL 策略: 走参考引擎, 不依赖 low1/low2/... 等固定参数
                        sp = p.get("params", {})
                        futs[(wi, ci)] = ex.submit(
                            run_one_general, wb, p["period"], warm,
                            p["init_cash"], p["init_position"], p["trade_qty"],
                            p.get("scale", 1.0),
                            p.get("buy_pct", 0.0), p.get("sell_pct", 0.0),
                            p.get("all_in", False),
                            strategy_name=strategy_name,
                            strategy_params=sp)
                    else:
                        futs[(wi, ci)] = ex.submit(_run_window, wb, p, warm)
            for wi in range(len(win_data)):
                metrics[wi] = [futs[(wi, ci)].result()
                               for ci in range(len(params_list))]
    dt = time.perf_counter() - t0

    # ---- 行装配: 每窗原始指标 (单窗 full 不加前缀, 兼容旧输出) + 年化扣费 ----
    # 通用策略: row 里把 params 拍平 (lookback/breakout_pct 提到顶层)
    rows = []
    for ci, combo in enumerate(combos):
        row = {**combo}
        if "params" in row:
            # 把基础参数拍平到顶层 (grid 值已存在, 不覆盖)
            for k, v in row["params"].items():
                row.setdefault(k, v)
            # 不再保留 "params" 字段 (避免与拍平后的 key 重复)
            del row["params"]
        for (nm, _, _), ms in zip(win_data, metrics):
            m = ms[ci]
            pre = "" if nm == "full" else f"{nm}_"
            row.update({f"{pre}{k}": v for k, v in m.items()})
            row[f"{pre}ann_net"] = _ann_net(m, fee_bp)
        test_metrics = [ms[ci] for (nm, _, _), ms in zip(win_data, metrics)
                        if nm != "train"]
        anns = [_ann_net(m, fee_bp) for m in test_metrics]
        row["ann_net_min"] = min(anns)
        row["ann_net_mean"] = float(np.mean(anns))
        row["pos_ratio"] = float(np.mean([a > 0 for a in anns]))
        row["sharpe_min"] = min(m["sharpe_excess"] for m in test_metrics)
        row["sortino_min"] = min(m.get("sortino_excess", 0.0) for m in test_metrics)
        row["calmar_max"] = max(m.get("calmar", 0.0) for m in test_metrics)
        row["cagr_max"] = max(m.get("cagr", 0.0) for m in test_metrics)
        row["max_dd_days_max"] = max(m.get("max_dd_days", 0.0) for m in test_metrics)
        row["x_mdd_max"] = max(m["x_mdd"] for m in test_metrics)
        row["filter_pass"] = bool(
            all(m["n_trades"] >= min_trades for m in test_metrics)
            and all(m["max_drawdown"] <= max_mdd for m in test_metrics))
        rows.append(row)

    # ---- 邻域衰减 S + 复合 score + 帕累托 ----
    S = _neighbor_decay(combos, np.array([r["ann_net_min"] for r in rows]))
    for row, s in zip(rows, S):
        row["S"] = float(s)
        base_v = row["ann_net_min"]
        row["score"] = base_v / (1.0 + lam * s) if base_v > 0 else base_v
    pf = _pareto_flag(np.array([r["ann_net_mean"] for r in rows]),
                      np.array([r["sharpe_min"] for r in rows]))
    for row, f in zip(rows, pf):
        row["pareto"] = bool(f)

    df = pd.DataFrame(rows)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    if verbose:
        n_bars = len(bars["stime"])
        n_runs = len(combos) * len(win_data)
        print(f"  -- 扫描 {len(combos)} 组 x {len(win_data)} 窗 x {n_bars} 根 bar "
              f"[{device}] 费率 {fee_bp:.0f}bp λ={lam} "
              f"耗时 {dt:.2f}s ({dt / max(n_runs, 1) * 1000:.2f} ms/次回测)",
              flush=True)
    return df
