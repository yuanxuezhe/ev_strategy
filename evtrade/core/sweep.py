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

from .kernel_dsl import _EMPTY_F, _EMPTY_SIG, run_one_dsl, strategy_has_dsl

# 引擎级 grid key (框架自带); 策略参数名由各策略 import 时通过
# register_grid_keys() 注入 (见 evtrade/strategies/<name>.py 末尾)
GRID_KEYS = ("tf1", "period", "trade_qty", "scale",
             "buy_pct", "sell_pct", "all_in")
# 网格扫描允许的额外 key (按 strategy_name 的 params_spec 动态加入)
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
                      strategy_name: str | None = None) -> dict:
    """单组参数单窗回测 (统一入口, dict 形式; 任意 DSL 策略)

    strategy_name: 必填 (策略 key), 见 evtrade.strategies.available_strategies()
    p 必含键: period / init_cash / init_position / trade_qty / params
    可选:    tf1 (默认 21) / scale / buy_pct / sell_pct / all_in
    """
    if not strategy_name:
        raise ValueError("run_one_from_dict: strategy_name is required")
    return run_one_dsl(bars, p["period"], warmup_until,
                       p["init_cash"], p["init_position"], p["trade_qty"],
                       tf1=p.get("tf1", 21), scale=p.get("scale", 1.0),
                       buy_pct=p.get("buy_pct", 0.0), sell_pct=p.get("sell_pct", 0.0),
                       all_in=p.get("all_in", False),
                       strategy_name=strategy_name,
                       strategy_params=p.get("params") or {})


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
                     strategy_name: str | None = None,
                     strategy_params: dict | None = None) -> dict:
    """通用策略单组参数单窗回测 (走参考引擎, 任意 strategies/ 子包策略)

    strategy_name: 必填 (策略 key); 通过策略的 params_spec 解析所有参数,
                   不绑定任何具体策略的参数名。
    走参考引擎 (慢约 500x), 不进 numba/CUDA 加速路径; 用于参数空间探索 /
    新策略验证。
    """
    if not strategy_name:
        raise ValueError("run_one_general: strategy_name is required")
    from .engine import build_engine
    from ._harness import NumpyDictFeed

    feed = NumpyDictFeed(bars)

    tf1 = (strategy_params or {}).get("tf1", 21)
    eng = build_engine(feed, period=period,
                       warmup_until=str(warmup_until) if warmup_until else None,
                       strategy_name=strategy_name,
                       strategy_params=strategy_params or {},
                       init_cash=init_cash, init_position=init_position,
                       trade_qty=trade_qty, scale=scale,
                       buy_pct=buy_pct, sell_pct=sell_pct, all_in=all_in,
                       tf1=tf1, verbose=False)
    # 空数据安全 (sweep 单窗可能给到空 bars, 已知 flush 会 IndexError, 跳过即可)
    if len(bars["stime"]) == 0:
        return _empty_metrics(init_cash, init_position)
    eng.run()

    account = eng.executor.account
    final_price = account.last_price
    eq = account.equity(final_price)
    baseline = account.baseline_equity(final_price)
    diff = eq - baseline
    pct = (diff / baseline * 100.0) if baseline else 0.0

    # 年数口径与 kernel.summarize / GPU _collect_gpu_results 一致:
    # 首个 mark=1 (>= warmup_until) 的 bar 到末根, 不含预热段。
    from .kernel import encoded_to_epoch
    stime = bars["stime"]
    years = 0.0
    idx0 = int(np.searchsorted(stime, int(warmup_until)))
    if idx0 < len(stime) and stime[-1] > stime[idx0]:
        years = ((encoded_to_epoch(int(stime[-1])) - encoded_to_epoch(int(stime[idx0])))
                 / (365.25 * 86400.0))

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
        # 参考引擎未追踪逐日权益曲线, 以下指标仅 kernel/GPU 路径计算;
        # 此处占位 0.0/True, 与 _empty_metrics 一致 (口径缺口, 待参考层补权益序列后统一)。
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
          strategy_name: str | None = None):
    """并发扫描 + 鲁棒评分; 返回 pandas.DataFrame (按 score 降序; 任意 DSL 策略)

    strategy_name: 必填 (策略 key); 通过 params_spec 解析所有策略参数。

    base 键: start/period/trade_qty/init_cash/init_position + strategy_params
    combos:  parse_grid 的输出, 覆盖 base 中的对应键 (策略参数键)。
    splits:  滚动 WFO 分割日列表 ["20260101","20260401"]; 1 个时窗口名为 train/test
             (兼容旧 --split), 多个时为 train/test1..testK。None=单窗 (列名无前缀)。

    路径选择 (三端同源):
      - 带 DSL docstring 的策略 -> numba 特化内核 (run_one_dsl);
        device="gpu" 时走通用 CUDA kernel (cuda_sweep_window_generic)
      - 无 DSL 的策略 -> 参考引擎 (run_one_general, 慢约 500x)

    参数传递 (统一路径): 策略参数以 params dict 为唯一事实源
    (--params > _defaults 落盘 > params_spec 默认)。
    """
    if not strategy_name:
        raise ValueError("sweep: strategy_name is required")
    import pandas as pd

    # 能力探测: requested device + 策略参数上限 + gpu_available
    # auto 模式下 gpu 不可用或策略不兼容时, 降级到 cpu 并 warn
    from .capability import gpu_available, select_device
    import logging
    _log = logging.getLogger("evtrade.sweep")
    gpu_ok = gpu_available()
    if device == "gpu" and not gpu_ok:
        raise ValueError("请求 device='gpu' 但环境无可用 cupy/CUDA")
    resolved = select_device(strategy_name, device, gpu_ok)
    if device == "auto" and resolved != device:
        _log.warning("device=auto 降级: 请求 %s -> 实际 %s", device, resolved)
    device = resolved  # 后续分支统一用 device (cpu/gpu)

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
        """按 end_ymd 截取 bars (numpy slice, 一次 searchsorted + 视图切片,
        避免对每个 key 都做一次 Python boolean mask 拷贝)。

        end_ymd=None 时返回原 bars 引用 (上游只在 wfo=单窗时用一次, 拷贝无意义)。
        """
        if end_ymd is None:
            return bars
        cutoff = int(end_ymd) * 1_000_000
        # np.searchsorted(side='left') 找第一个 >= cutoff 的下标, 与原语义
        # 'stime < cutoff' 等价; 但 searchsorted 是 C 实现的 O(log n) 标量,
        # 然后用 slice 拿到原数组的视图 (no copy)
        idx = int(np.searchsorted(bars["stime"], cutoff, side="left"))
        return {k: v[:idx] for k, v in bars.items()}

    win_data = [(nm, window_bars(e), int(w) * 1_000_000) for nm, e, w in wins]

    # 统一参数路径: 策略参数以 params dict 为唯一事实源 (--params > _defaults 落盘);
    # 所有 GPU/CPU 路径 (含 channel_deviation) 都读 p["params"]。
    base_params = dict(base.get("params") or {})
    # 网格 key 覆盖到 base["params"] 上; 缺失的参数继承基础值
    #   语义: --params 提供基础参数, --grid 在指定 key 上扫描, 未指定 key 沿用基础值
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
    # 路径选择: dsl_fast -> numba 内核; 无 DSL -> 参考引擎兜底。
    # GPU 一律走通用 CUDA kernel (cuda_sweep_window_generic, 所有 DSL 策略同路径)。
    dsl_fast = strategy_has_dsl(strategy_name)
    metrics = [None] * len(win_data)
    if device == "gpu" and dsl_fast:
        from .gpu import cuda_sweep_window_generic
        for wi, (nm, wb, warm) in enumerate(win_data):
            metrics[wi] = cuda_sweep_window_generic(wb, params_list, warm,
                                                    strategy_name=strategy_name)
    else:
        # 主线程预热: 把首次 numba specialization 串行化在主线程, 避免并发
        # worker 同时第一次调用 dsl_kernel(name).run_backtest 时各自触发
        # numba type specialization, 浪费 CPU 且扭曲冷启动延迟。
        if dsl_fast:
            try:
                from .kernel_dsl import dsl_kernel, make_state_general
                _first_p = params_list[0]
                _kmod = dsl_kernel(strategy_name)
                _probe_st = make_state_general(
                    strategy_name, _first_p["period"], warmup_until=0,
                    tf1=_first_p.get("tf1", 21),
                    init_cash=_first_p["init_cash"],
                    init_position=_first_p["init_position"],
                    trade_qty=_first_p["trade_qty"],
                    strategy_params=_first_p.get("params", {}))
                # 取第一窗口前 32 个 bar 当探针 (覆盖 warmup 即可, 避免预热开销)
                _wb = win_data[0][1]
                _n = min(32, len(_wb["stime"]))
                _kmod.run_backtest(_probe_st,
                                   _wb["stime"][:_n], _wb["open"][:_n],
                                   _wb["high"][:_n], _wb["low"][:_n],
                                   _wb["close"][:_n], _wb["volume"][:_n],
                                   _EMPTY_SIG, _EMPTY_F, _EMPTY_F)
                _kmod.summarize(_probe_st)
            except Exception as _e:
                # 预热失败不应中断 sweep —— 后面的 run_one_dsl 会再次触发
                # 并给出原始错误; 这里只 log warning 让用户感知
                import logging
                logging.getLogger("evtrade.sweep").warning(
                    "numba kernel warm-up 失败 (将走冷启动): %s", _e)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            # 把 (wi, ci) 全部 flatten 一次性提交, 让窗口切片与 kernel 执行
            # 真正重叠; 用 as_completed 流式收集到预分配 metrics 缓冲,
            # 哪个先完先写对应槽位 —— 不再按 wi 顺序 barrier 等待整窗完成。
            metrics = [[None] * len(params_list) for _ in win_data]
            futures = {}
            for wi, (nm, wb, warm) in enumerate(win_data):
                for ci, p in enumerate(params_list):
                    if dsl_fast:
                        # DSL 策略: numba 内核
                        fut = ex.submit(
                            run_one_dsl, wb, p["period"], warm,
                            p["init_cash"], p["init_position"], p["trade_qty"],
                            p.get("tf1", 21),
                            p.get("scale", 1.0),
                            p.get("buy_pct", 0.0), p.get("sell_pct", 0.0),
                            p.get("all_in", False),
                            strategy_name=strategy_name,
                            strategy_params=p.get("params", {}))
                    else:
                        # 无 DSL 策略: 走参考引擎, 通用 params dict 透传
                        sp = p.get("params", {})
                        fut = ex.submit(
                            run_one_general, wb, p["period"], warm,
                            p["init_cash"], p["init_position"], p["trade_qty"],
                            p.get("scale", 1.0),
                            p.get("buy_pct", 0.0), p.get("sell_pct", 0.0),
                            p.get("all_in", False),
                            strategy_name=strategy_name,
                            strategy_params=sp)
                    futures[fut] = (wi, ci)

            # 流式收集: as_completed 按完成顺序返回; 立即写 metrics[wi][ci]
            # 这样无需等待整窗; 异常也会立刻 re-raise。
            from concurrent.futures import as_completed
            for fut in as_completed(futures):
                wi, ci = futures[fut]
                metrics[wi][ci] = fut.result()
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
