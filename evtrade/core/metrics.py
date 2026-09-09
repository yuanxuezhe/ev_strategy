from __future__ import annotations
"""绩效汇总与数据打包工具 (从 core/kernel.py 迁入; DSL 删除后统一放 metrics)

================================================================
✅  可改层  ✅
================================================================
bars_to_arrays   list[Bar] -> numpy dict (vectorized 引擎输入)
summarize        终态 -> 绩效字典 (与 ref engine.print_summary 同口径)
trades_to_list   成交记录 -> list[dict]
bucket_table     全轨迹 -> 每根周期K线表格 (framework 层, 无指标列)
bundle_per_bar   per-bar 数组 -> 通用契约 dict
"""

import numpy as np

from .timeutils import encoded_to_epoch


def bars_to_arrays(bars) -> dict:
    """list[Bar] -> 内核输入数组 (stime 转 14 位整数)"""
    n = len(bars)
    stime = np.empty(n, np.int64)
    o = np.empty(n, np.float64)
    h = np.empty(n, np.float64)
    l = np.empty(n, np.float64)
    c = np.empty(n, np.float64)
    v = np.empty(n, np.float64)
    for i, b in enumerate(bars):
        stime[i] = int(b.stime)
        o[i] = b.open
        h[i] = b.high
        l[i] = b.low
        c[i] = b.close
        v[i] = b.volume
    return {"stime": stime, "open": o, "high": h, "low": l, "close": c, "volume": v}


def summarize(final_state: dict, init_cash: float, init_position: float,
              equity_curve=None, baseline_curve=None,
              first_ts: int = 0, last_ts: int = 0) -> dict:
    """终态 dict + equity 序列 -> 绩效字典 (16 字段)

    equity_curve / baseline_curve 由 vectorized_engine 在逐 bar 执行中累积:
      equity_curve[i]   = cash_i + position_i * close_i
      baseline_curve[i] = init_cash + init_position * close_i

    若 caller 未提供 equity 序列, 用终态 + 序列长度 1 退化, 此时 cagr/sortino/
    calmar/max_dd_days 等时间序列指标全部退化为 0.0。

    final_state: {"cash", "position", "last_price", "n_trades", "n_buy",
                  "n_sell", "turnover", "peak_equity", "max_drawdown", ...}
    (vectorized_engine 的 exec_state 或 ref Account 的等价 dict)

    返回字段 (16):
      final_price, final_cash, final_position, final_equity
      baseline, excess, excess_pct, years
      n_trades, n_buy, n_sell, turnover
      cagr, sharpe_excess, sortino_excess, calmar
      max_dd_days, max_dd_recovered, x_mdd, max_drawdown
    """
    cash = final_state["cash"]
    position = final_state["position"]
    last_price = final_state["last_price"]
    baseline = init_cash + init_position * last_price
    equity = cash + position * last_price
    diff = equity - baseline
    pct = (diff / baseline * 100) if baseline else 0.0

    years = 0.0
    if first_ts and last_ts > first_ts:
        years = ((encoded_to_epoch(last_ts) - encoded_to_epoch(first_ts))
                 / (365.25 * 86400.0))
    ann_excess_pct = (pct / years) if years > 0 else 0.0

    # === 基于 equity_curve 的全套 16 字段 ===
    cagr = 0.0
    sharpe_excess = 0.0
    sortino_excess = 0.0
    calmar = 0.0
    max_dd_days = 0.0
    max_dd_recovered = 0
    x_mdd = final_state.get("max_drawdown", 0.0)
    max_drawdown = x_mdd

    if equity_curve is not None and len(equity_curve) >= 2 and years > 0:
        eq = np.asarray(equity_curve, dtype=np.float64)
        bl = (np.asarray(baseline_curve, dtype=np.float64)
              if baseline_curve is not None else None)

        # === 1. max_drawdown / max_dd_days / max_dd_recovered ===
        running_peak = np.maximum.accumulate(eq)
        drawdown = eq - running_peak                     # ≤ 0
        is_dd = drawdown < 0.0
        max_drawdown = float(-drawdown.min()) if is_dd.any() else 0.0

        # 最大回撤首次触底 + 之后恢复到前高的天数
        if is_dd.any():
            trough_idx = int(np.argmin(drawdown))
            peak_idx = int(np.argmax(eq[:trough_idx + 1]))
            if baseline_curve is not None:
                # 用交易日 / 时间戳比例估算"回撤天数"; 在没有逐 bar ts 的场景下
                # 直接用 (bars 数 / 总 bars) * 365 * years 作 fallback
                pass
            max_dd_recovered = len(eq) - trough_idx     # 离末尾多少 bar 才恢复

        # === 2. CAGR ===
        if eq[0] > 0:
            cagr = (eq[-1] / eq[0]) ** (1.0 / years) - 1.0
            cagr *= 100.0  # 百分比

        # === 3. Sharpe / Sortino (excess over baseline) ===
        if bl is not None and len(bl) == len(eq) and bl[0] > 0:
            rets_eq = np.diff(eq) / eq[:-1]
            rets_bl = np.diff(bl) / bl[:-1]
            excess_rets = rets_eq - rets_bl
            if len(excess_rets) > 1:
                mean_ex = float(np.mean(excess_rets))
                std_ex = float(np.std(excess_rets, ddof=1))
                downside = excess_rets[excess_rets < 0.0]
                if std_ex > 0.0:
                    bars_per_year = len(excess_rets) / years if years > 0 else 1.0
                    sharpe_excess = mean_ex / std_ex * (bars_per_year ** 0.5) * 100.0
                if len(downside) > 0:
                    down_std = float(np.sqrt(np.mean(downside ** 2)))
                    if down_std > 0.0:
                        sortino_excess = (mean_ex / down_std
                                          * (bars_per_year ** 0.5) * 100.0)

            # === 4. Calmar = CAGR / max_dd (取 %) ===
            if max_drawdown > 0:
                calmar = cagr / max_drawdown

        # === 5. max_dd_days (估算; 真实时间戳下用 peak_idx→trough_idx→恢复 周期数) ===
        if is_dd.any():
            trough_idx = int(np.argmin(drawdown))
            peak_idx = int(np.argmax(eq[:trough_idx + 1]))
            n_dd_bars = trough_idx - peak_idx
            if years > 0 and len(eq) > 1:
                bars_per_day = len(eq) / (years * 365.25)
                if bars_per_day > 0:
                    max_dd_days = n_dd_bars / bars_per_day

    return {
        "final_price": last_price,
        "n_trades": final_state.get("n_trades", 0),
        "n_buy": final_state.get("n_buy", 0),
        "n_sell": final_state.get("n_sell", 0),
        "final_cash": cash,
        "final_position": position,
        "final_equity": equity,
        "baseline": baseline,
        "excess": diff,
        "excess_pct": pct,
        "years": years,
        "ann_excess_pct": ann_excess_pct,
        "turnover": final_state.get("turnover", 0.0),
        # 16 字段扩展
        "cagr": cagr,
        "sharpe_excess": sharpe_excess,
        "sortino_excess": sortino_excess,
        "calmar": calmar,
        "max_dd_days": max_dd_days,
        "max_dd_recovered": max_dd_recovered,
        "x_mdd": x_mdd,
        "max_drawdown": max_drawdown,
    }


def trades_to_list(trades: list) -> list[dict]:
    """成交记录 list -> list[dict] (规范化; 已是 list[dict] 时直通)"""
    if not trades:
        return []
    if isinstance(trades[0], dict):
        return trades
    # 兼容 tuple 形式 (ts, side_int, qty, price)
    out = []
    for t in trades:
        ts, side_int, qty, price = t
        out.append({"ts": int(ts), "side": "BUY" if side_int == 1 else "SELL",
                    "qty": float(qty), "price": float(price)})
    return out


def bucket_table(stime: np.ndarray, sig: np.ndarray,
                 ts_out, o_out, h_out, l_out, c_out, v_out) -> dict:
    """从全轨迹构建"每根周期K线"表格 (纯 numpy 向量化, framework 层)。

    每行 = 一个周期桶在闭合时点的状态: ts/open/high/low/close/volume/count/sig/n_sig。
    框架只输出行情 + 信号轨迹; 指标列由策略 hook 拼接。
    """
    n = len(ts_out)
    new_bucket = np.r_[True, ts_out[1:] != ts_out[:-1]]
    first_idx = np.flatnonzero(new_bucket)
    last_idx = np.flatnonzero(np.r_[new_bucket[1:], True])
    count = np.diff(np.r_[first_idx, n])
    n_sig = np.add.reduceat(np.abs(sig).astype(np.int64), first_idx)

    return {
        "ts": ts_out[last_idx],
        "open": o_out[last_idx],
        "high": h_out[last_idx],
        "low": l_out[last_idx],
        "close": c_out[last_idx],
        "volume": v_out[last_idx],
        "count": count,
        "sig": sig[last_idx],
        "n_sig": n_sig,
    }


def bundle_per_bar(sig, ts_arr, o_arr, h_arr, l_arr, c_arr, v_arr) -> dict:
    """把 per-bar 数组打包为通用契约 dict

    返回: {"sig": ..., "per_bar": {"ts", "o", "h", "l", "c", "v"}}
    (框架层不出现指标键; 策略 hook 按需从 per_bar 取值算指标)
    """
    return {
        "sig": sig,
        "per_bar": {
            "ts": ts_arr, "o": o_arr, "h": h_arr,
            "l": l_arr, "c": c_arr, "v": v_arr,
        },
    }
