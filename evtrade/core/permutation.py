from __future__ import annotations
"""蒙特卡洛置换检验: 排除"好绩效是运气"

================================================================
⚠️  冻结层模块  ⚠️
================================================================
置换方式 (日块打乱, 不是逐 bar 打乱) 是 kbs/13 的核心方法论, 已被
tests/test_sweep.py::test_permutation_sanity 锁定。
逐 bar 打乱会人工制造跳变, null 被打穿到 -60%/年, 完全失去参考价值。
================================================================

原理 (日块自助置换, stationary bootstrap 的离散版):
  以自然日为块整日打乱价格路径的顺序 —— 破坏多日结构、保留日内微观结构与
  价格分布。同一组参数在打乱后的数据上重跑 N 次, 得到"无多日结构"null 下的
  年化超额分布; 真实绩效在该分布中的位置即 p 值:
      p = (随机绩效 >= 真实绩效的次数 + 1) / (N + 1)
  p < 0.05 => 策略在多日维度上的择时方向显著异于运气。
"""

import numpy as np

from .sweep import run_one_from_dict


def permutation_test(bars: dict, params: dict, warmup_until: int,
                     n: int = 500, fee_bp: float = 5.0, seed: int = 42,
                     verbose: bool = False) -> dict:
    """对单组参数做置换检验 (日块自助置换), 返回真实年化超额 (费前) 与 p 值。

    置换方式: 以**自然日为块**整日打乱顺序 (日内 OHLC 行情原样保留, stime 槽位
    不变) —— 破坏多日结构、保留日内微观结构。不能用逐 bar 打乱: 那会制造剧烈的
    人工跳变, 使触发条件 (H 回到下轨/L 回到上轨) 在随机数据上系统性"买贵卖贱",
    null 被机械泄漏打穿, 失去参考价值 (实测 -60%/年, 见 kbs/13)。

    统计量用**费前**年化超额: 费用影响由确定性评分 (ann_net, sweep 层) 单独衡量,
    置换检验只回答"多日维度的择时方向是否异于运气"。
    """
    real_m = run_one_from_dict(bars, params, warmup_until)
    real = real_m.get("ann_excess_pct", 0.0)

    stime = bars["stime"]
    nb = len(stime)
    day = stime // 1_000_000
    day_start_idx = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])  # 每日首根下标
    n_days = len(day_start_idx)
    if n_days < 3:
        return {"real_ann_net": real, "p_value": 1.0, "null_mean": real,
                "null_p95": real, "null_max": real, "n": 0, "years": real_m.get("years", 0.0)}

    price_keys = ("open", "high", "low", "close", "volume")
    day_lens = np.diff(np.r_[day_start_idx, nb])
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    ge = 0
    for j in range(n):
        perm_days = rng.permutation(n_days)          # 整日整日地打乱
        new_idx = np.concatenate(
            [np.arange(day_start_idx[d], day_start_idx[d] + day_lens[d])
             for d in perm_days])
        shuffled = dict(bars)
        for k in price_keys:
            shuffled[k] = bars[k][new_idx]
        m = run_one_from_dict(shuffled, params, warmup_until)
        vals[j] = m.get("ann_excess_pct", 0.0)
        if vals[j] >= real:
            ge += 1
        if verbose and (j + 1) % 100 == 0:
            print(f"    ... {j + 1}/{n}", flush=True)

    p = (ge + 1) / (n + 1)
    return {
        "real_ann_net": real,
        "p_value": p,
        "null_mean": float(np.mean(vals)),
        "null_p95": float(np.percentile(vals, 95)),
        "null_max": float(np.max(vals)),
        "n": n,
        "years": real_m.get("years", 0.0),
    }
