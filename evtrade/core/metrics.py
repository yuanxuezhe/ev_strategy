from __future__ import annotations
"""绩效汇总与数据打包工具

公开 API:
  bars_to_arrays       list[Bar] -> numpy dict (vectorized 引擎输入)
  summarize            终态 + equity 序列 -> 25 字段绩效字典
  trades_to_list       成交记录 -> list[dict] (规范化)
  bucket_table         全轨迹 -> 每根周期K线表格 (framework 层, 无指标列)
  bundle_per_bar       per-bar 数组 -> 通用契约 dict

summarize 字段集 (25):
  终态 (5):     final_price / final_cash / final_position / final_equity / baseline
  交易 (5):     n_trades / n_buy / n_sell / turnover / excess_pct
  时间 (2):     years / cagr_excess
  风险调整 (5): cagr / sharpe_excess / sortino_excess / calmar / ir
  回撤 (3):     max_drawdown / max_dd_days / max_dd_recovered
  持仓行为 (7): win_rate / profit_factor / avg_pnl / max_consecutive_wins
                / max_consecutive_losses / avg_hold_bars / max_hold_bars
  基准对比 (2): baseline_max_dd / dd_excess
"""

import math

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


# ============ 持仓行为统计 ============

def _round_trip_stats(trades: list, bucket_seconds: int = 300) -> dict:
    """相邻 BUY/SELL 配对算 PnL + 持仓周期

    bucket_seconds: 桶周期秒数 (5m=300, 15m=900, 1h=3600)。用于把
        ts 时间差转换为"桶数"。caller 已知 period 时应传入; 默认 5m。

    返回: {pnls: list[float], hold_bars: list[int], trade_pairs: int}
    未配对的开仓/平仓忽略。
    """
    pnls: list[float] = []
    hold_bars: list[int] = []
    open_qty = 0.0
    open_cost = 0.0
    open_ts = 0
    for t in trades:
        side = t.get("side")
        qty = float(t.get("qty", 0.0))
        price = float(t.get("price", 0.0))
        ts = int(t.get("ts", 0))
        if side == "BUY":
            open_qty += qty
            open_cost += qty * price
            if open_ts == 0:
                open_ts = ts
        elif side == "SELL":
            if open_qty > 0:
                matched = min(qty, open_qty)
                pnl = matched * (price - open_cost / open_qty) if open_qty > 0 else 0.0
                pnls.append(pnl)
                # 持仓周期: ts 是 encoded (YYYYMMDDHHmmss), 转 epoch 后差分得秒数
                if open_ts and ts > open_ts and bucket_seconds > 0:
                    sec_diff = encoded_to_epoch(ts) - encoded_to_epoch(open_ts)
                    hold_bars.append(max(1, sec_diff // bucket_seconds))
                open_qty -= matched
                if open_qty <= 1e-9:
                    open_qty = 0.0
                    open_cost = 0.0
                    open_ts = 0
    return {"pnls": pnls, "hold_bars": hold_bars, "trade_pairs": len(pnls)}


def _max_consecutive(pnls: list[float], kind: str) -> int:
    """最长连续盈利/亏损笔数

    kind: "win" (pnl > 0) 或 "loss" (pnl < 0)
    """
    best = 0
    cur = 0
    cond = (lambda p: p > 0) if kind == "win" else (lambda p: p < 0)
    for p in pnls:
        if cond(p):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _max_drawdown_recovered(eq: np.ndarray, trough_idx: int) -> int:
    """trough 之后首个恢复 >= 前高 的 bar 数; 未恢复时 -1"""
    n = len(eq)
    if trough_idx >= n:
        return -1
    peak_at_trough = float(eq[:trough_idx + 1].max())
    for i in range(trough_idx + 1, n):
        if eq[i] >= peak_at_trough:
            return int(i - trough_idx)
    return -1  # sentinel: 从未恢复


# ============ 主入口 ============

def summarize(final_state: dict, init_cash: float, init_position: float,
              equity_curve=None, baseline_curve=None,
              first_ts: int = 0, last_ts: int = 0,
              trades: list | None = None,
              bucket_seconds: int = 300) -> dict:
    """终态 + equity 序列 + 成交记录 -> 25 字段绩效字典

    equity_curve / baseline_curve 由 vectorized_engine 在逐 bar 执行中累积:
      equity_curve[i]   = cash_i + position_i * close_i
      baseline_curve[i] = init_cash + init_position * close_i

    final_state: {"cash", "position", "last_price", "n_trades", "n_buy",
                  "n_sell", "turnover"} (vectorized_engine 的 exec_state)
    trades: list[dict] with keys {ts, side, qty, price} (BUY/SELL 配对)

    若 caller 未提供 equity 序列, 时间序列指标全部退化为 0.0。
    若 caller 未提供 trades, 持仓行为指标退化为 0.0 / inf 哨兵。
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

    # 持仓行为 (不依赖 equity 序列; 即使 equity 为 None 也能算)
    # 持仓行为 (不依赖 equity 序列; 即使 equity 为 None 也能算)
    pnls: list[float] = []
    hold_bars: list[int] = []
    if trades:
        stats = _round_trip_stats(trades, bucket_seconds=bucket_seconds)
        pnls = stats["pnls"]
        hold_bars = stats["hold_bars"]

    n_pairs = len(pnls)
    n_win = sum(1 for p in pnls if p > 0)
    n_loss = sum(1 for p in pnls if p < 0)
    total_win = sum(p for p in pnls if p > 0)
    total_loss = sum(p for p in pnls if p < 0)

    win_rate = (n_win / n_pairs) if n_pairs > 0 else 0.0
    profit_factor = (total_win / abs(total_loss)) if total_loss < 0 else (
        math.inf if total_win > 0 else 0.0)
    avg_pnl = (sum(pnls) / n_pairs) if n_pairs > 0 else 0.0
    max_consec_wins = _max_consecutive(pnls, "win")
    max_consec_losses = _max_consecutive(pnls, "loss")
    avg_hold_bars = (sum(hold_bars) / len(hold_bars)) if hold_bars else 0.0
    max_hold_bars = max(hold_bars) if hold_bars else 0

    # === 基于 equity_curve 的全套字段 ===
    cagr = 0.0
    sharpe_excess = 0.0
    sortino_excess = 0.0
    calmar = 0.0
    ir = 0.0
    max_dd_days = 0.0
    max_dd_recovered = -1
    max_drawdown = 0.0
    baseline_max_dd = 0.0
    cagr_excess = 0.0
    x_mdd = 0.0  # 累计超额曲线回撤

    if equity_curve is not None and len(equity_curve) >= 2:
        eq = np.asarray(equity_curve, dtype=np.float64)
        bl = (np.asarray(baseline_curve, dtype=np.float64)
              if baseline_curve is not None else None)

        # === 1. max_drawdown (策略; 占当时 peak 的小数) ===
        running_peak = np.maximum.accumulate(eq)
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdown = np.where(running_peak > 0,
                                (eq - running_peak) / running_peak,
                                0.0)
        is_dd = drawdown < 0.0
        if is_dd.any():
            trough_idx = int(np.argmin(drawdown))
            max_drawdown = float(-drawdown.min())
            max_dd_recovered = _max_drawdown_recovered(eq, trough_idx)
            peak_idx = int(np.argmax(eq[:trough_idx + 1]))
            if years > 0:
                n_dd_bars = trough_idx - peak_idx
                bars_per_day = len(eq) / (years * 365.25)
                if bars_per_day > 0:
                    max_dd_days = n_dd_bars / bars_per_day

        # === 2. baseline_max_dd (单位与 max_drawdown 同式) ===
        if bl is not None and len(bl) == len(eq):
            bl_peak = np.maximum.accumulate(bl)
            with np.errstate(divide="ignore", invalid="ignore"):
                bl_dd = np.where(bl_peak > 0,
                                 (bl - bl_peak) / bl_peak,
                                 0.0)
            if (bl_dd < 0.0).any():
                baseline_max_dd = float(-bl_dd.min())

        # === 3. CAGR ===
        if eq[0] > 0 and years > 0:
            safe_years = max(years, 1e-9)
            cagr = (eq[-1] / eq[0]) ** (1.0 / safe_years) - 1.0
            cagr *= 100.0

        # === 4. Sharpe / Sortino / IR / Calmar (基于超额收益率) ===
        if bl is not None and len(bl) == len(eq) and bl[0] > 0 and years > 0:
            rets_eq = np.diff(eq) / eq[:-1]
            rets_bl = np.diff(bl) / bl[:-1]
            excess_rets = rets_eq - rets_bl
            if eq[0] > 0 and bl[0] > 0:
                safe_years = max(years, 1e-9)
                cum_excess = (eq[-1] / eq[0]) / (bl[-1] / bl[0]) - 1.0
                cagr_excess = ((1.0 + cum_excess) ** (1.0 / safe_years) - 1.0) * 100.0
            if len(excess_rets) > 1:
                mean_ex = float(np.mean(excess_rets))
                std_ex = float(np.std(excess_rets, ddof=1))
                downside = excess_rets[excess_rets < 0.0]
                bars_per_year = len(excess_rets) / years
                # x_mdd: 累计超额曲线最大回撤 (占初始 baseline 的小数)
                cum_ex_arr = np.cumsum(excess_rets)
                ex_peak = np.maximum.accumulate(cum_ex_arr)
                x_mdd = float((ex_peak - cum_ex_arr).max()) if len(cum_ex_arr) else 0.0
                if std_ex > 0.0:
                    sharpe_excess = (mean_ex / std_ex
                                     * (bars_per_year ** 0.5) * 100.0)
                    # IR = 年化超额 / 年化跟踪误差; rf=0 时数值上 ≡ sharpe_excess
                    ir = sharpe_excess
                if len(downside) > 0:
                    down_std = float(np.sqrt(np.mean(downside ** 2)))
                    if down_std > 0.0:
                        sortino_excess = (mean_ex / down_std
                                          * (bars_per_year ** 0.5) * 100.0)

            # Calmar = cagr(%) / max_dd(小数); 同单位相除
            if max_drawdown > 0:
                calmar = (cagr / 100.0) / max_drawdown

    # dd_excess = 策略回撤 - 基准回撤 (正值=策略比基准更深)
    dd_excess = max_drawdown - baseline_max_dd

    return {
        # 终态 (5)
        "final_price": last_price,
        "final_cash": cash,
        "final_position": position,
        "final_equity": equity,
        "baseline": baseline,
        # 交易 (5)
        "n_trades": final_state.get("n_trades", 0),
        "n_buy": final_state.get("n_buy", 0),
        "n_sell": final_state.get("n_sell", 0),
        "turnover": final_state.get("turnover", 0.0),
        "excess_pct": pct,
        # 时间 (2)
        "years": years,
        "cagr_excess": cagr_excess,
        # 风险调整 (5)
        "cagr": cagr,
        "sharpe_excess": sharpe_excess,
        "sortino_excess": sortino_excess,
        "calmar": calmar,
        "ir": ir,
        # 回撤 (3)
        "max_drawdown": max_drawdown,
        "max_dd_days": max_dd_days,
        "max_dd_recovered": max_dd_recovered,
        "x_mdd": x_mdd,
        # 持仓行为 (7)
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_pnl": avg_pnl,
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "avg_hold_bars": avg_hold_bars,
        "max_hold_bars": max_hold_bars,
        # 基准对比 (2)
        "baseline_max_dd": baseline_max_dd,
        "dd_excess": dd_excess,
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
