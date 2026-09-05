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

GRID_KEYS = ("low1", "low2", "high1", "high2", "tf1", "period", "trade_qty", "scale")

_EMPTY_SIG = np.empty(0, np.int8)
_EMPTY_F = np.empty(0, np.float64)


def parse_grid(specs: list) -> list[dict]:
    """["low1=1.0,1.5", "high2=0.3,0.5"] -> 笛卡尔积参数组合列表"""
    import itertools
    axes = []
    for s in specs:
        name, _, values = str(s).partition("=")
        name = name.strip()
        if name not in GRID_KEYS:
            raise ValueError(f"不支持的网格参数 {name!r}, 可用: {GRID_KEYS}")
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
            else:
                combo[k] = float(raw)
        combos.append(combo)
    return combos


def run_one(bars: dict, period: str, warmup_until: int, tf1: int,
            low1: float, low2: float, high1: float, high2: float,
            init_cash: float, init_position: float, trade_qty: float,
            scale: float = 1.0) -> dict:
    """单组参数单窗回测 (内核), 返回 kernel.summarize 口径的绩效字典"""
    st = make_state(period=period, warmup_until=warmup_until, tf1=tf1,
                    low1=low1, low2=low2, high1=high1, high2=high2,
                    init_cash=init_cash, init_position=init_position,
                    trade_qty=trade_qty, scale=scale)
    run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                 bars["close"], bars["volume"], _EMPTY_SIG, _EMPTY_F, _EMPTY_F)
    return summarize(st)


def _run_window(bars: dict, p: dict, warmup_until: int) -> dict:
    return run_one(bars, p["period"], warmup_until, p["tf1"],
                   p["low1"], p["low2"], p["high1"], p["high2"],
                   p["init_cash"], p["init_position"], p["trade_qty"],
                   p.get("scale", 1.0))


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
    keys = [k for k in combos[0] if len({c.get(k) for c in combos}) > 1]
    if not keys:
        return out
    lookup = {tuple(sorted(c.items())): i for i, c in enumerate(combos)}
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
                    j = lookup.get(tuple(sorted(nc.items())))
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
          lam: float = 1.0, min_trades: int = 30, max_mdd: float = 1.0):
    """并发扫描 + 鲁棒评分; 返回 pandas.DataFrame (按 score 降序)

    base 键: start/period/tf1/low1/low2/high1/high2/trade_qty/init_cash/init_position
    combos:  parse_grid 的输出, 覆盖 base 中的对应键。
    splits:  滚动 WFO 分割日列表 ["20260101","20260401"]; 1 个时窗口名为 train/test
             (兼容旧 --split), 多个时为 train/test1..testK。None=单窗 (列名无前缀)。
    """
    import pandas as pd

    if split_ymd and not splits:
        splits = [split_ymd]
    start = base["start"]
    if splits:
        sp = sorted(str(s) for s in splits)
        wins = [("train", sp[0], start)]
        for i, s in enumerate(sp):
            end_b = sp[i + 1] if i + 1 < len(sp) else None
            nm = "test" if len(sp) == 1 else f"test{i + 1}"
            wins.append((nm, end_b, s))
    else:
        wins = [("full", None, start)]

    def window_bars(end_ymd):
        if end_ymd is None:
            return bars
        mask = bars["stime"] < int(end_ymd) * 1_000_000
        return {k: v[mask] for k, v in bars.items()}

    win_data = [(nm, window_bars(e), int(w) * 1_000_000) for nm, e, w in wins]
    params_list = [{**base, **c} for c in combos]

    t0 = time.perf_counter()
    metrics = [None] * len(win_data)
    if device == "gpu":
        from .gpu import cuda_sweep_window
        for wi, (nm, wb, warm) in enumerate(win_data):
            metrics[wi] = cuda_sweep_window(wb, params_list, warm)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {}
            for wi, (nm, wb, warm) in enumerate(win_data):
                for ci, p in enumerate(params_list):
                    futs[(wi, ci)] = ex.submit(_run_window, wb, p, warm)
            for wi in range(len(win_data)):
                metrics[wi] = [futs[(wi, ci)].result()
                               for ci in range(len(params_list))]
    dt = time.perf_counter() - t0

    # ---- 行装配: 每窗原始指标 (单窗 full 不加前缀, 兼容旧输出) + 年化扣费 ----
    rows = []
    for ci, combo in enumerate(combos):
        row = {**combo}
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
