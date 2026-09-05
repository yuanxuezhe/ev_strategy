from __future__ import annotations
"""命令行入口

  python mysql_analyze_demo.py --period 5m --start 20250101 --end 20260903 --no-sleep
      等价于 python -m evtrade backtest ... (默认 --engine kernel, 与 ref 逐笔等价)
  python -m evtrade sweep --grid low1=1.0,1.5,2.0 --grid high2=0.3,0.5,0.8 --split 20260101
"""

import argparse
import time

import numpy as np

from .config import INIT_CASH, INIT_POSITION, INTERVAL, TF1, TRADE_QTY
from .timeutils import resolve_period_seconds


def _period_type(s: str) -> str:
    resolve_period_seconds(s)          # 校验格式, 非法直接报 argparse 错
    return s


# ============ backtest 子命令 ============

def build_backtest_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="minute_bars 周期合并 + 通达信通道轨 (回测/实盘统一)")
    ap.add_argument("--period", default="5m", type=_period_type,
                    help="K线周期, 任意 数字+m/h/d: 5m/7m/15m/30m/90m/2h/4h/6h/1d/3d ...")
    ap.add_argument("--start", default="20250101", help="策略起始日期 YYYYMMDD")
    ap.add_argument("--end", default="20260903", help="策略结束日期 YYYYMMDD")
    ap.add_argument("--step-days", type=int, default=7, help="[ref] 分段查询天数(闭区间)")
    ap.add_argument("--tf1", type=int, default=TF1, help="通道轨 EMA 周期")
    ap.add_argument("--no-sleep", action="store_true",
                    help="[ref] 去掉每根 bar 的 sleep; kernel 引擎本就全速")
    ap.add_argument("--trade-qty", type=float, default=TRADE_QTY, help="每次信号交易股数")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="倍投系数: 连续同向信号数量=上次×scale (首次=trade_qty, "
                         "反向重置); 1.0=关闭, 如 2.0")
    ap.add_argument("--code", default="159992.SZ", help="证券代码 (如 159992.SZ / 513120.SH)")
    ap.add_argument("--low1", type=float, default=1.5, help="下轨极端偏离阈值(百分比)")
    ap.add_argument("--low2", type=float, default=1.0, help="下轨回撤触发阈值(百分比)")
    ap.add_argument("--high1", type=float, default=1.5, help="上轨极端偏离阈值(百分比)")
    ap.add_argument("--high2", type=float, default=0.5, help="上轨回撤触发阈值(百分比)")
    ap.add_argument("--engine", default="kernel", choices=["kernel", "ref"],
                    help="kernel=numba流式内核(默认,与ref逐笔等价) / ref=原Python实现")
    ap.add_argument("--warmup-days", type=int, default=365,
                    help="[kernel] 预热天数 (拉取 start 之前的行情供指标就绪)")
    ap.add_argument("--data-cache", default=None,
                    help="[kernel] 行情 npz 缓存目录 (命中后不访问数据库)")
    ap.add_argument("--show-bars", action="store_true",
                    help="打印每根周期K线 (桶闭合时点) 的 OHLCV、EMA 上下轨与四个偏离值")
    ap.add_argument("--bars-out", default=None,
                    help="同 --show-bars 内容输出 CSV (大数据量建议用这个)")
    return ap


def _f4(v) -> str:
    """数值格式化: NaN -> ----"""
    try:
        if v != v:
            return "----"
    except TypeError:
        return "----"
    return f"{v:.4f}"


def _run_kernel(args):
    from .data import load_bars
    from .kernel import (bucket_table, make_state, run_backtest,
                         run_backtest_trace, summarize, trades_to_list)

    bars = load_bars(args.code, args.start, args.end,
                     warmup_days=args.warmup_days, cache_dir=args.data_cache)
    n = len(bars["stime"])
    show_bars = args.show_bars or args.bars_out
    print(f"证券: {args.code}  周期: {args.period}  策略日期: {args.start}~{args.end}  "
          f"预热: {args.warmup_days}天  TF1={args.tf1}  "
          f"low1/low2={args.low1}/{args.low2} high1/high2={args.high1}/{args.high2}  "
          f"scale={args.scale}  引擎: kernel (numba)\n", flush=True)

    st = make_state(period=args.period, warmup_until=int(args.start) * 1_000_000,
                    tf1=args.tf1, low1=args.low1, low2=args.low2,
                    high1=args.high1, high2=args.high2,
                    init_cash=INIT_CASH, init_position=INIT_POSITION,
                    trade_qty=args.trade_qty, scale=args.scale,
                    record_trades=True, trade_cap=n)
    t0 = time.perf_counter()
    if show_bars:
        sig_out = np.zeros(n, np.int8)
        up_out = np.full(n, np.nan)
        dw_out = np.full(n, np.nan)
        ts_out = np.zeros(n, np.int64)
        o_out = np.zeros(n)
        h_out = np.zeros(n)
        l_out = np.zeros(n)
        c_out = np.zeros(n)
        v_out = np.zeros(n)
        run_backtest_trace(st, bars["stime"], bars["open"], bars["high"],
                           bars["low"], bars["close"], bars["volume"],
                           sig_out, up_out, dw_out,
                           ts_out, o_out, h_out, l_out, c_out, v_out)
        tab = bucket_table(bars["stime"], sig_out, up_out, dw_out,
                           ts_out, o_out, h_out, l_out, c_out, v_out)
        # 只保留策略期 (--start 起) 的桶; 预热期仅用于指标准备, 不输出
        mask = tab["ts"] >= int(args.start) * 1_000_000
        tab = {k: v[mask] for k, v in tab.items()}
    else:
        run_backtest(st, bars["stime"], bars["open"], bars["high"], bars["low"],
                     bars["close"], bars["volume"],
                     np.empty(0, np.int8), np.empty(0), np.empty(0))
    dt = time.perf_counter() - t0

    for t in trades_to_list(st):
        side = "BUY " if t["side"] == "BUY" else "SELL"
        arrow = ">>" if t["side"] == "BUY" else ">>"
        print(f"        {arrow} {side} {t['qty']:.0f}股 @ {t['price']:.4f}  "
              f"[{t['ts']}]  剩余资金 {t['cash_after']:.2f}", flush=True)

    s = summarize(st)
    print("\n" + "=" * 60)
    print("回测盈亏汇总")
    print("=" * 60)
    print(f"期末价 (最后一根close) : {s['final_price']:.4f}")
    print(f"交易次数              : {s['n_trades']} (BUY {s['n_buy']} / SELL {s['n_sell']})")
    print(f"期初资金 / 期初持仓    : {st.init_cash:.0f} / {st.init_position:.0f}股")
    print(f"期末资金 / 期末持仓    : {st.cash:.2f} / {st.position:.0f}股")
    print(f"期末持仓市值           : {st.position * st.last_price:.2f}")
    print(f"策略总资产 (资金+市值) : {s['final_equity']:.2f}")
    print(f"不操作基线 (资金+市值) : {s['baseline']:.2f}")
    print(f"盈亏差额 (策略-基线)   : {s['excess']:+.2f}")
    print(f"盈亏比例              : {s['excess_pct']:+.2f}%")
    print(f"最大回撤 (逐bar盯市)   : {s['max_drawdown']:.2%}")
    print(f"成交额合计            : {s['turnover']:.0f}")
    print(f"内核耗时              : {dt * 1000:.1f} ms ({n} 根 1m bar)")
    print("=" * 60)

    if show_bars:
        if args.show_bars:
            print("\n周期K线明细 (每行 = 一个桶在闭合时点; ts 为右端点; sig 为该桶最后一根 bar 的信号):")
            ts_l = tab["ts"].tolist()
            cols = {k: tab[k].tolist() for k in
                    ("open", "high", "low", "close", "volume", "count", "up", "dw",
                     "low_dev", "high_dev", "low_dev_h", "high_dev_l", "sig", "n_sig")}
            for i in range(len(ts_l)):
                print(f"[{ts_l[i]}] O:{_f4(cols['open'][i])} H:{_f4(cols['high'][i])} "
                      f"L:{_f4(cols['low'][i])} C:{_f4(cols['close'][i])} "
                      f"V:{cols['volume'][i]:.0f} x{cols['count'][i]} | "
                      f"UP={_f4(cols['up'][i])} DW={_f4(cols['dw'][i])} | "
                      f"low_dev={_f4(cols['low_dev'][i])} high_dev={_f4(cols['high_dev'][i])} "
                      f"low_dev_h={_f4(cols['low_dev_h'][i])} high_dev_l={_f4(cols['high_dev_l'][i])} | "
                      f"sig={cols['sig'][i]} n_sig={cols['n_sig'][i]}", flush=True)
        if args.bars_out:
            with open(args.bars_out, "w", encoding="utf-8-sig") as f:
                f.write("ts,open,high,low,close,volume,count,up,dw,"
                        "low_dev,high_dev,low_dev_h,high_dev_l,signal,n_sig\n")
                for i in range(len(tab["ts"])):
                    f.write(f"{tab['ts'][i]},{tab['open'][i]},{tab['high'][i]},"
                            f"{tab['low'][i]},{tab['close'][i]},{tab['volume'][i]},"
                            f"{tab['count'][i]},{tab['up'][i]},{tab['dw'][i]},"
                            f"{tab['low_dev'][i]},{tab['high_dev'][i]},"
                            f"{tab['low_dev_h'][i]},{tab['high_dev_l'][i]},"
                            f"{tab['sig'][i]},{tab['n_sig'][i]}\n")
            print(f"\nK线明细已保存: {args.bars_out} ({len(tab['ts'])} 行)")


def _run_ref(args):
    """原 Python 实现路径 (保留逐根 sleep / 分段拉数的原始行为)"""
    from .account import Account
    from .aggregator import BarAggregator
    from .engine import Engine
    from .execution import SimulatedExecutor
    from .feeds import MySQLBacktestFeed
    from .strategy import ChannelDeviationStrategy

    delay = 0 if args.no_sleep else INTERVAL
    feed = MySQLBacktestFeed(code=args.code, start_ymd=args.start, end_ymd=args.end,
                             step_days=args.step_days, delay=delay, verbose=True)
    account = Account(cash=INIT_CASH, position=INIT_POSITION)
    executor = SimulatedExecutor(account, qty=args.trade_qty, verbose=True,
                                 scale=args.scale)
    aggregator = BarAggregator(resolve_period_seconds(args.period), on_bars=None,
                               warmup_until=feed.warmup_until)
    strategy = ChannelDeviationStrategy(low1=args.low1, low2=args.low2,
                                        high1=args.high1, high2=args.high2)
    engine = Engine(feed, aggregator, strategy, executor, tf1=args.tf1, verbose=True)
    print(f"证券: {args.code}  周期: {args.period}  策略日期: {args.start}~{args.end}  "
          f"预热起点: {feed.warmup_start}  分段: {args.step_days}天/段(闭区间)  "
          f"TF1={args.tf1}  sleep={'OFF' if args.no_sleep else 'ON'}  "
          f"low1/low2={args.low1}/{args.low2} high1/high2={args.high1}/{args.high2}  "
          f"引擎: ref  Ctrl+C 停止\n")
    engine.run()
    engine.print_summary()


def backtest_main(argv=None):
    args = build_backtest_parser().parse_args(argv)
    if args.engine == "ref":
        _run_ref(args)
    else:
        _run_kernel(args)


# ============ sweep 子命令 ============

def build_sweep_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="参数并发扫描 (numba 内核, 线程池并行)")
    ap.add_argument("--code", default="159992.SZ")
    ap.add_argument("--start", default="20250101", help="策略起始日期 (预热另计)")
    ap.add_argument("--end", default="20260903")
    ap.add_argument("--period", default="5m", type=_period_type,
                    help="K线周期, 任意 数字+m/h/d")
    ap.add_argument("--tf1", type=int, default=TF1)
    ap.add_argument("--low1", type=float, default=1.5)
    ap.add_argument("--low2", type=float, default=1.0)
    ap.add_argument("--high1", type=float, default=1.5)
    ap.add_argument("--high2", type=float, default=0.5)
    ap.add_argument("--trade-qty", type=float, default=TRADE_QTY)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="倍投系数 (连续同向信号数量累乘, 反向重置; 1.0=关闭)")
    ap.add_argument("--grid", action="append", default=[],
                    help="参数网格, 可多次: --grid low1=1.0,1.5,2.0 "
                         "(支持 low1/low2/high1/high2/tf1/period/trade_qty)")
    ap.add_argument("--split", default=None,
                    help="单分割日 YYYYMMDD (等价 --splits 该值; 窗口名 train/test)")
    ap.add_argument("--splits", default=None,
                    help="滚动 WFO 分割日, 逗号分隔: --splits 20260101,20260401,20260701 "
                         "-> train + test1..test3, score 取最差 test 窗")
    ap.add_argument("--fee-bp", type=float, default=5.0,
                    help="费率 (万分比, 单边) 用于年化扣费; 默认 5bp")
    ap.add_argument("--score-lambda", type=float, default=1.0,
                    help="邻域衰减惩罚 λ: score = 最差窗年化扣费超额 / (1+λ·S)")
    ap.add_argument("--min-trades", type=int, default=30,
                    help="硬过滤: 各 test 窗最少成交笔数")
    ap.add_argument("--max-mdd", type=float, default=1.0,
                    help="硬过滤: 各 test 窗权益最大回撤上限 (0.15=15%%)")
    ap.add_argument("--mc", type=int, default=0,
                    help="对 score 前 --mc-top 名做蒙特卡洛置换检验 (打乱行情 N 次)")
    ap.add_argument("--mc-top", type=int, default=5)
    ap.add_argument("--warmup-days", type=int, default=365)
    ap.add_argument("--workers", type=int, default=None, help="并发线程数 (默认=CPU核数)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"],
                    help="cpu=numba内核+线程池 / gpu=CUDA单launch (需 cupy, 见 gpu.py)")
    ap.add_argument("--data-cache", default=None, help="行情 npz 缓存目录")
    ap.add_argument("--synthetic-days", type=int, default=0,
                    help=">0 时用合成数据 (不连库); 数值=截至 --end 的数据天数, "
                         "应覆盖预热需求")
    ap.add_argument("--out", default="sweep_results.csv", help="结果 CSV 路径")
    ap.add_argument("--top", type=int, default=20, help="控制台展示前 N 组")
    return ap


def sweep_main(argv=None):
    args = build_sweep_parser().parse_args(argv)
    from .data import load_bars, synthetic_bars
    from .kernel import bars_to_arrays
    from .sweep import GRID_KEYS, parse_grid, sweep as run_sweep

    if args.synthetic_days > 0:
        from datetime import datetime, timedelta
        d_end = datetime.strptime(args.end, "%Y%m%d")
        d_start = d_end - timedelta(days=args.synthetic_days)
        print(f"生成合成数据: {d_start:%Y%m%d} ~ {args.end} "
              f"({args.synthetic_days} 天, seed=42)", flush=True)
        bars = bars_to_arrays(synthetic_bars(days=args.synthetic_days,
                                             start_ymd=f"{d_start:%Y%m%d}"))
    else:
        bars = load_bars(args.code, args.start, args.end,
                         warmup_days=args.warmup_days, cache_dir=args.data_cache)

    base = {"start": args.start, "period": args.period, "tf1": args.tf1,
            "low1": args.low1, "low2": args.low2, "high1": args.high1,
            "high2": args.high2, "trade_qty": args.trade_qty,
            "scale": args.scale,
            "init_cash": INIT_CASH, "init_position": INIT_POSITION}
    combos = parse_grid(args.grid) or [{}]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()] if args.splits else None
    if splits:
        print(f"扫描 {len(combos)} 组参数 (滚动 WFO {len(splits)} 窗: "
              f"train + test1..test{len(splits)}) ...\n", flush=True)
    else:
        print(f"扫描 {len(combos)} 组参数 (单窗) ...\n", flush=True)
    df = run_sweep(bars, base, combos, split_ymd=args.split,
                   n_workers=args.workers, device=args.device,
                   splits=splits, fee_bp=args.fee_bp, lam=args.score_lambda,
                   min_trades=args.min_trades, max_mdd=args.max_mdd)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    cols = [c for c in df.columns]
    show = [c for c in df.columns
            if c in ("score", "ann_net_min", "ann_net_mean", "pos_ratio",
                     "sharpe_min", "x_mdd_max", "S", "pareto", "filter_pass")
            or c in GRID_KEYS]
    print(df[show].head(args.top).to_string(index=False))
    print(f"\n全部结果已保存: {args.out} ({len(df)} 行; 列: {cols})")

    if args.mc > 0:
        from .permutation import permutation_test
        print(f"\n蒙特卡洛置换检验 (前 {args.mc_top} 名, 各打乱 {args.mc} 次, "
              f"统计量=费前年化超额):")
        warm = int(base["start"]) * 1_000_000
        for _, row in df.head(args.mc_top).iterrows():
            combo = {k: row[k] for k in GRID_KEYS if k in row}
            params = {**base, **combo}
            r = permutation_test(bars, params, warm, n=args.mc,
                                 fee_bp=args.fee_bp)
            print(f"  {' '.join(f'{k}={row[k]}' for k in combo)}  "
                  f"真实年化超额 {r['real_ann_net']:+.2f}%/年  "
                  f"p={r['p_value']:.3f}  (随机分布均值 {r['null_mean']:+.2f}, "
                  f"95分位 {r['null_p95']:+.2f})", flush=True)


# ============ replay 子命令 ============

def build_replay_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="录制回放对账: bar 日志 -> 内核信号轨迹 (可选: 与参考引擎对账)")
    ap.add_argument("--log", required=True,
                    help="bar 日志 (CSV: stime,code,open,high,low,close,volume; "
                         "由 append_bar/write_bars_log 产生)")
    ap.add_argument("--period", default="5m", type=_period_type,
                    help="K线周期, 任意 数字+m/h/d")
    ap.add_argument("--tf1", type=int, default=TF1)
    ap.add_argument("--low1", type=float, default=1.5)
    ap.add_argument("--low2", type=float, default=1.0)
    ap.add_argument("--high1", type=float, default=1.5)
    ap.add_argument("--high2", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--warmup-until", default=None,
                    help="预热阈值 YYYYMMDD (可选; 日志从更早开始时用于只回放策略期)")
    ap.add_argument("--against-ref", action="store_true",
                    help="同时用参考 Python 引擎回放并逐 bar 对账")
    ap.add_argument("--signals-out", default=None, help="信号轨迹输出 CSV")
    return ap


def replay_main(argv=None):
    args = build_replay_parser().parse_args(argv)
    from .replay import (read_bars_log, reconcile, replay_kernel, write_bars_log)

    bars = read_bars_log(args.log)
    if not bars:
        raise SystemExit("日志为空")
    warm = int(args.warmup_until) * 1_000_000 if args.warmup_until else 0
    print(f"回放: {len(bars)} 根 bar [{bars[0].stime} ~ {bars[-1].stime}]  "
          f"period={args.period} tf1={args.tf1} "
          f"low1/low2={args.low1}/{args.low2} high1/high2={args.high1}/{args.high2} "
          f"scale={args.scale}\n", flush=True)

    k = replay_kernel(bars, args.period, warm, args.tf1, args.low1, args.low2,
                      args.high1, args.high2, scale=args.scale)
    s = k["summary"]
    print(f"信号 {int((k['sig'] != 0).sum())} 个 (BUY {s['n_buy']} / SELL {s['n_sell']}), "
          f"成交 {s['n_trades']} 笔, 期末总资产 {s['final_equity']:,.2f} "
          f"(基线 {s['baseline']:,.2f}, 超额 {s['excess_pct']:+.2f}%)")

    if args.signals_out:
        with open(args.signals_out, "w", encoding="utf-8-sig") as f:
            f.write("stime,signal,up,dw\n")
            for b, sig, up, dw in zip(bars, k["sig"].tolist(), k["up"].tolist(),
                                      k["dw"].tolist()):
                f.write(f"{b.stime},{sig},{up},{dw}\n")
        print(f"信号轨迹已保存: {args.signals_out}")

    if args.against_ref:
        print()
        reconcile(bars, args.period, warm, args.tf1, args.low1, args.low2,
                  args.high1, args.high2, scale=args.scale)


def main(argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("backtest", "sweep", "replay"):
        cmd = args.pop(0)
        if cmd == "sweep":
            return sweep_main(args)
        if cmd == "replay":
            return replay_main(args)
    return backtest_main(args)


if __name__ == "__main__":
    main()
