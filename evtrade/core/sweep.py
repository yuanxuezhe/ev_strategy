from __future__ import annotations
"""参数并发扫描 + 鲁棒选参框架

WFO 多窗回测 + 复合评分 (score = ann_net_min / (1 + λ·S)) + 帕累托标记 + 硬过滤。
单一执行路径: vectorized_engine.run_vectorized。
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# 引擎级 grid key (框架自带); 策略参数名由各策略 import 时通过
# register_grid_keys() 注入
GRID_KEYS = ("tf1", "period", "trade_qty", "scale",
             "buy_pct", "sell_pct", "all_in")
_GRID_EXTRA_KEYS: set[str] = set()


def register_grid_keys(keys: set[str]) -> None:
    """新策略注册其允许的 grid key; sweep 解析时合并"""
    _GRID_EXTRA_KEYS.update(keys)


def parse_grid(specs: list, extra_keys: set[str] | None = None) -> list[dict]:
    """["k1=1.0,1.5", "k2=0.3,0.5"] -> 笛卡尔积参数组合列表

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


def run_one_from_dict(bars: dict, p: dict, warmup_until: int,
                      strategy_name: str | None = None,
                      device: str = "cpu") -> dict:
    """单组参数单窗回测 (统一入口, dict 形式; 走 vectorized 引擎)

    strategy_name: 必填 (策略 key), 见 evtrade.strategies.available_strategies()
    p 必含键: period / init_cash / init_position / trade_qty / params
    可选:    tf1 (默认 21) / scale / buy_pct / sell_pct / all_in
    """
    if not strategy_name:
        raise ValueError("run_one_from_dict: strategy_name is required")
    from ..strategies import get_strategy
    from .vectorized_engine import run_vectorized

    strategy = get_strategy(strategy_name, params=p.get("params") or {})
    return run_vectorized(
        bars_1m=bars,
        period=p["period"],
        warmup_until=warmup_until,
        strategy=strategy,
        params=strategy.params,
        init_cash=p["init_cash"],
        init_position=p["init_position"],
        trade_qty=p["trade_qty"],
        scale=p.get("scale", 1.0),
        buy_pct=p.get("buy_pct", 0.0),
        sell_pct=p.get("sell_pct", 0.0),
        device=device,
    )["summary"]


def run_one_vectorized(bars: dict, period: str, warmup_until: int,
                       strategy_name: str,
                       strategy_params: dict | None = None,
                       init_cash: float = 200000.0,
                       init_position: float = 200000.0,
                       trade_qty: float = 10000.0,
                       scale: float = 1.0,
                       buy_pct: float = 0.0,
                       sell_pct: float = 0.0,
                       device: str = "cpu") -> dict:
    """单组参数单窗回测 (vectorized 入口; 内部走 run_vectorized)

    strategy_params: 策略参数 dict (会被 _resolve_params 校验)
    """
    from ..strategies import get_strategy
    from .vectorized_engine import run_vectorized
    strategy = get_strategy(strategy_name, params=strategy_params or {})
    return run_vectorized(
        bars_1m=bars, period=period, warmup_until=warmup_until,
        strategy=strategy, params=strategy.params,
        init_cash=init_cash, init_position=init_position,
        trade_qty=trade_qty, scale=scale,
        buy_pct=buy_pct, sell_pct=sell_pct, device=device,
    )["summary"]


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
    order = np.lexsort((-b, -a))
    flags = np.zeros(len(a), dtype=bool)
    best_b = -np.inf
    for idx in order:
        if b[idx] > best_b:
            flags[idx] = True
        best_b = max(best_b, b[idx])
    return flags


def sweep(bars: dict, base: dict, combos: list[dict],
          split_ymd: str = None, n_workers: int | None = None,
          verbose: bool = True,
          device: str = "cpu", splits: list = None, fee_bp: float = 5.0,
          lam: float = 1.0, min_trades: int = 30, max_mdd: float = 1.0,
          strategy_name: str | None = None):
    """并发扫描 + 鲁棒评分; 返回 pandas.DataFrame (按 score 降序)

    strategy_name: 必填 (策略 key); 通过 params_spec 解析所有策略参数。

    base 键: start/period/trade_qty/init_cash/init_position + strategy_params
    combos:  parse_grid 的输出, 覆盖 base 中的对应键 (策略参数键)。
    splits:  滚动 WFO 分割日列表 ["20260101","20260401"]; 1 个时窗口名为 train/test,
             多个时为 train/test1..testK。None=单窗 (列名无前缀)。
    """
    if not strategy_name:
        raise ValueError("sweep: strategy_name is required")
    import pandas as pd

    # 能力探测: requested device + gpu_available; auto 模式下 gpu 不可用时降级 cpu
    from .capability import gpu_available, select_device
    import logging
    _log = logging.getLogger("evtrade.sweep")
    gpu_ok = gpu_available()
    if device == "gpu" and not gpu_ok:
        raise ValueError("请求 device='gpu' 但环境无可用 cupy/CUDA")
    resolved = select_device(strategy_name, device, gpu_ok)
    if device == "auto" and resolved != device:
        _log.warning("device=auto 降级: 请求 %s -> 实际 %s", device, resolved)
    device = resolved

    if split_ymd and not splits:
        splits = [split_ymd]
    start = base["start"]
    if splits:
        sp = sorted(str(s) for s in splits)
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
        cutoff = int(end_ymd) * 1_000_000
        idx = int(np.searchsorted(bars["stime"], cutoff, side="left"))
        return {k: v[:idx] for k, v in bars.items()}

    win_data = [(nm, window_bars(e), int(w) * 1_000_000) for nm, e, w in wins]

    # ---- 解析 strategy_params: 只保留 params_spec 白名单内键 ----
    from ..strategies import get_strategy_class
    spec = get_strategy_class(strategy_name).params_spec or {}
    base_params = dict(base.get("params") or {})

    def _strategy_params(c: dict) -> dict:
        """从 combo 抽策略参数: 同时认 params 包装层与顶层 spec 键"""
        nested = dict(c.get("params") or {})
        merged = {**base_params, **nested, **{k: v for k, v in c.items() if k in spec}}
        return {k: v for k, v in merged.items() if k in spec}

    new_combos = []
    for c in combos:
        new_c = dict(c)
        new_c["params"] = _strategy_params(c)
        new_combos.append(new_c)
    combos = new_combos
    params_list = [{**base, **c} for c in combos]

    t0 = time.perf_counter()
    metrics = [[None] * len(params_list) for _ in win_data]
    if n_workers and n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {}
            for wi, (nm, wb, warm) in enumerate(win_data):
                for ci, p in enumerate(params_list):
                    fut = ex.submit(
                        run_one_vectorized, wb, p["period"], warm,
                        strategy_name=strategy_name,
                        strategy_params=p.get("params", {}),
                        init_cash=p["init_cash"],
                        init_position=p["init_position"],
                        trade_qty=p["trade_qty"],
                        scale=p.get("scale", 1.0),
                        buy_pct=p.get("buy_pct", 0.0),
                        sell_pct=p.get("sell_pct", 0.0),
                        device=device,
                    )
                    futures[fut] = (wi, ci)
            from concurrent.futures import as_completed
            for fut in as_completed(futures):
                wi, ci = futures[fut]
                metrics[wi][ci] = fut.result()
    else:
        for wi, (nm, wb, warm) in enumerate(win_data):
            for ci, p in enumerate(params_list):
                metrics[wi][ci] = run_one_vectorized(
                    wb, p["period"], warm,
                    strategy_name=strategy_name,
                    strategy_params=p.get("params", {}),
                    init_cash=p["init_cash"],
                    init_position=p["init_position"],
                    trade_qty=p["trade_qty"],
                    scale=p.get("scale", 1.0),
                    buy_pct=p.get("buy_pct", 0.0),
                    sell_pct=p.get("sell_pct", 0.0),
                    device=device,
                )
    dt = time.perf_counter() - t0

    # ---- 行装配 ----
    rows = []
    for ci, combo in enumerate(combos):
        row = {**combo}
        if "params" in row:
            for k, v in row["params"].items():
                row.setdefault(k, v)
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
        row["sharpe_min"] = min(m.get("sharpe_excess", 0.0) for m in test_metrics)
        row["sortino_min"] = min(m.get("sortino_excess", 0.0) for m in test_metrics)
        row["calmar_max"] = max(m.get("calmar", 0.0) for m in test_metrics)
        row["cagr_max"] = max(m.get("cagr", 0.0) for m in test_metrics)
        row["max_dd_days_max"] = max(m.get("max_dd_days", 0.0) for m in test_metrics)
        row["x_mdd_max"] = max(m.get("x_mdd", 0.0) for m in test_metrics)
        # max_drawdown 是占初始权益的小数; --max-mdd 默认 1.0 = 100% = 不限
        row["filter_pass"] = bool(
            all(m["n_trades"] >= min_trades for m in test_metrics)
            and all(m.get("max_drawdown", 0.0) <= max_mdd for m in test_metrics))
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

