from __future__ import annotations
"""参数并发扫描

2026-09-13 重构: framework 不再汇总 metrics; sweep 仅按策略 params_spec 网格
跑策略 step, 返回每组的 final_state 透传 + 邻域 S + pareto 标记。
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# 引擎级 grid key (仅 period; funding/撮合 由策略 params 承担)
GRID_KEYS = ("period",)


def parse_grid(specs: list, extra_keys: set[str] | None = None,
                type_hints: dict[str, type] | None = None) -> list[dict]:
    """["k1=1.0,1.5", "k2=0.3,0.5"] -> 笛卡尔积参数组合列表

    extra_keys: 额外允许的 grid key (由策略的 params_spec 提供)。
                 None 时只允许 GRID_KEYS 白名单; 传非空集合时扩展。
    type_hints:  按 key 给类型提示; 用于决定 grid value 是否转 float。
                 例如 `{"higher_period": str}` 让 "30m"/"1h" 保持字符串。
                 缺省: int 转 int, float 转 float, str / None 保持 str。
    """
    import itertools
    allowed = set(GRID_KEYS) | set(extra_keys or set())
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
            # 类型决策: type_hints 优先; 缺省 (向后兼容) 按"period 留字符串, 其它 float"
            if k == "period":
                combo[k] = raw
                continue
            t = (type_hints or {}).get(k)
            if t is str:
                combo[k] = raw
            elif t is int:
                combo[k] = int(raw)
            else:
                # float / 缺省 / 未知类型: 全部按 float 转 (向后兼容旧 test)
                combo[k] = float(raw)
        combos.append(combo)
    return combos


def run_one_vectorized(bars: dict, period: str, warmup_until: int,
                       strategy_name: str,
                       strategy_params: dict | None = None) -> dict:
    """单组参数单窗回测 (vectorized 入口; 内部走 run_vectorized)

    strategy_params: 策略参数 dict (会被 _resolve_params 校验)
    返回: {"final_state": ..., "sig": ndarray, "buckets": dict} 透传 run_vectorized
    """
    from ..strategies import get_strategy
    from .vectorized_engine import run_vectorized
    strategy = get_strategy(strategy_name, params=strategy_params or {})
    return run_vectorized(
        bars_1m=bars, period=period, warmup_until=warmup_until,
        strategy=strategy, params=strategy.params,
    )


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
          device: str = "cpu", splits: list = None,
          lam: float = 1.0,
          strategy_name: str | None = None):
    """并发扫描; 返回 pandas.DataFrame (按 score 降序)

    2026-09-13 重构: framework 不再汇总 metrics; 只按 params 网格跑策略 step,
    透传 final_state, 用 final_state 中策略自定的标量字段做评分。
    若策略未声明标量, score=0 (排序 fallback)。

    strategy_name: 必填 (策略 key); 通过 params_spec 解析所有策略参数。
    base 键: start + strategy_params
    combos:  parse_grid 的输出, 覆盖 base 中的对应键 (策略参数键)。
    splits:  滚动 WFO 分割日列表 ["20260101","20260401"]; 1 个时窗口名为 train/test,
             多个时为 train/test1..testK。None=单窗 (列名无前缀)。
    """
    if not strategy_name:
        raise ValueError("sweep: strategy_name is required")
    import pandas as pd

    # 设备解析
    from ..backends import gpu_available, resolve_device
    import logging
    _log = logging.getLogger("evtrade.sweep")
    gpu_ok = gpu_available()
    resolved = resolve_device(device, gpu_ok)
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

    # 把策略 spec 字段平铺到 combo 顶层 (避免下游 setdefault + del 二次处理)
    combos = [{k: v for k, v in c.items() if k != "params"}
              | _strategy_params(c)
              for c in combos]

    # ---- 预校验: 跨字段 validators 失败的 combo 跳过 (不污染 sweep_results.csv) ----
    strategy_cls_for_validate = get_strategy_class(strategy_name)
    valid_combos = []
    skipped_combos = []
    for c in combos:
        try:
            strategy_cls_for_validate._resolve_params(
                {k: v for k, v in c.items() if k in spec})
            valid_combos.append(c)
        except ValueError as e:
            skipped_combos.append((c, str(e)))
    if skipped_combos:
        n_skip = len(skipped_combos)
        n_total = len(combos)
        print(f"  -- 跳过 {n_skip}/{n_total} 个违反 validator 的 combo "
              f"(如 {next(iter(skipped_combos))[1]})", flush=True)
    combos = valid_combos
    if not combos:
        raise ValueError(
            f"sweep: 所有 {n_total} 个 combo 都违反策略 validators; "
            f"检查 grid 与 params_spec")
    params_list = [{**base, **c} for c in combos]

    def _run_one(wb, p, warm):
        # 抽策略 spec 字段 (per-combo 已 flat 合并到 p 顶层) 当 strategy_params
        spec_keys_set = set(spec.keys())
        sp_params = {k: p[k] for k in spec_keys_set if k in p}
        return run_one_vectorized(
            wb, p["period"], warm,
            strategy_name=strategy_name,
            strategy_params=sp_params,
        )

    # ---- 路由: batched (opt-in) vs ThreadPool ----
    strategy_cls = strategy_cls_for_validate
    # "真覆写" 检测: 类自身 __dict__ 里有 batched_step (排除继承来的基类默认)
    use_batched = (
        "batched_step" in vars(strategy_cls)
        and device != "cpu"
        and len(params_list) >= 32
        and gpu_available()
    )

    t0 = time.perf_counter()
    results = [[None] * len(params_list) for _ in win_data]
    if use_batched:
        if verbose:
            print(f"  -- batched 路径 (device={device}, n_combos={len(params_list)}); "
                  f"--workers 忽略", flush=True)
        from .batched_sweep import run_batched
        try:
            results = run_batched(
                bars, base, params_list,
                strategy_cls=strategy_cls, device=device,
                splits=splits,
                split_ymd=split_ymd,
            )
        except RuntimeError as e:
            # CUDA OOM 自动 fallback ThreadPool
            if "out of memory" in str(e).lower() or "OutOfMemoryError" in type(e).__name__:
                print(f"  -- batched CUDA OOM -> 降级 ThreadPool: {e}", flush=True)
            else:
                raise
            use_batched = False   # 落入下方 ThreadPool 分支
    if not use_batched:
        if n_workers and n_workers > 1:
            from concurrent.futures import as_completed
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = {
                    ex.submit(_run_one, wb, p, warm): (wi, ci)
                    for wi, (_, wb, warm) in enumerate(win_data)
                    for ci, p in enumerate(params_list)
                }
                for fut in as_completed(futures):
                    wi, ci = futures[fut]
                    results[wi][ci] = fut.result()
        else:
            for wi, (_, wb, warm) in enumerate(win_data):
                for ci, p in enumerate(params_list):
                    results[wi][ci] = _run_one(wb, p, warm)
    dt = time.perf_counter() - t0

    # ---- 行装配 ----
    # 2026-09-13: framework 不再汇总 metrics; final_state 由策略持有, framework 仅
    # 透传 + 提供 score 计算用的标量 'primary_score' (策略 final_state 中应包含此字段;
    # 若无则 score=0)。策略可自己定义 final_state 字段; framework 不读其它字段。
    rows = []
    for ci, combo in enumerate(combos):
        row = {**combo}
        for (nm, _, _), res in zip(win_data, results):
            r = res[ci]
            final = r.get("final_state") if r else None
            pre = "" if nm == "full" else f"{nm}_"
            primary = _extract_primary_score(final)
            row[f"{pre}primary_score"] = primary
            row[f"{pre}final_state"] = final  # 策略自管的 dataclass / dict 透传
        test_scores = []
        for (nm, _, _), res in zip(win_data, results):
            if nm == "train":
                continue
            r = res[ci]
            final = r.get("final_state") if r else None
            test_scores.append(_extract_primary_score(final))
        row["primary_score_min"] = min(test_scores) if test_scores else 0.0
        row["primary_score_mean"] = float(np.mean(test_scores)) if test_scores else 0.0
        rows.append(row)

    # ---- 邻域衰减 S + 复合 score + 帕累托 ----
    S = _neighbor_decay(combos, np.array([r["primary_score_min"] for r in rows]))
    for row, s in zip(rows, S):
        row["S"] = float(s)
        base_v = row["primary_score_min"]
        row["score"] = base_v / (1.0 + lam * s) if base_v > 0 else base_v
    pf = _pareto_flag(np.array([r["primary_score_mean"] for r in rows]),
                      np.array([r["primary_score_min"] for r in rows]))
    for row, f in zip(rows, pf):
        row["pareto"] = bool(f)

    df = pd.DataFrame(rows)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    if verbose:
        n_bars = len(bars["stime"])
        n_runs = len(combos) * len(win_data)
        print(f"  -- 扫描 {len(combos)} 组 x {len(win_data)} 窗 x {n_bars} 根 bar "
              f"[{device}] λ={lam} "
              f"耗时 {dt:.2f}s ({dt / max(n_runs, 1) * 1000:.2f} ms/次回测)",
              flush=True)
    return df


def _extract_primary_score(final_state) -> float:
    """从策略 final_state 提 score 用标量 (framework 不识别具体字段语义)

    优先级: dataclass/dict['primary_score'] -> final_state['final_score'] -> 0
    """
    if final_state is None:
        return 0.0
    if isinstance(final_state, dict):
        return float(final_state.get("primary_score",
                                     final_state.get("final_score", 0.0)))
    return float(getattr(final_state, "primary_score",
                         getattr(final_state, "final_score", 0.0)))