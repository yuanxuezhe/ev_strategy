from __future__ import annotations
"""命令行入口

================================================================
✅  可改层模块  ✅  (用户面的主要修改点)
================================================================
本文件定义三个子命令: backtest / sweep / replay, 是用户面的主入口。

常见修改:

  1. 加新参数 (例: --max-position):
     - 在 build_backtest_parser() / build_sweep_parser() 加 add_argument
     - 在 _run_kernel / _run_ref 中读取并透传给 make_state 或 executor
     - 同步给内核加 jitclass 字段 (参照阶段 2 的 buy_pct/sell_pct 加法)
     - tests/test_xxx.py 加测试锁定

  2. 改输出格式:
     - _run_kernel() 末尾的 print 段 (汇总 / 信号行)
     - 保留 "=== 60 字符等号 ===" 包裹风格, 便于日志检索

  3. 加新子命令 (例: live 启动实盘):
     - 在 main() 的 if/elif 链加一项
     - 写一个 _run_xxx() 函数, 调用现有 Engine/ChainedFeed/BrokerExecutor

  4. 别忘了:
     - evtrade/__main__.py 是 `python -m evtrade` 入口, 委托 main()
     - evtrade/__init__.py 顶层导出符号 (新模块要在此处加 from)
================================================================
"""

import argparse
import time

import numpy as np

from .core.config import INIT_CASH, INIT_POSITION, INTERVAL, TF1, TRADE_QTY
from .frozen.timeutils import resolve_period_seconds


def _parse_params(spec: str) -> dict:
    """'k1:v1;k2:v2' -> dict (类型自动推导: int / float / bool / str)

    用例:
      --params "low1:1.5;low2:1.0;tf1:21"
      --params "all_in:true;buy_pct:0.5"
    """
    out: dict = {}
    if not spec:
        return out
    for kv in spec.split(";"):
        kv = kv.strip()
        if not kv:
            continue
        if ":" not in kv:
            raise ValueError(f"参数格式错误 {kv!r}; 应为 'key:value'")
        k, _, v = kv.partition(":")
        k = k.strip()
        v = v.strip()
        out[k] = _auto_cast(v)
    return out


def _auto_cast(s: str):
    """字符串 -> int / float / bool / str"""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _period_type(s: str) -> str:
    resolve_period_seconds(s)          # 校验格式, 非法直接报 argparse 错
    return s


def _resolve_strategy_params(strategy_name: str, params_arg: str,
                              legacy_low1: float | None = None,
                              legacy_low2: float | None = None,
                              legacy_high1: float | None = None,
                              legacy_high2: float | None = None) -> dict:
    """解析策略参数, 优先级 (高 -> 低):
      1. CLI --params 显式传入 (params_arg 非空)
      2. evtrade/strategies/_defaults/<strategy_name>.json 落盘默认
      3. 旧 CLI kwargs (low1/low2/high1/high2) — channel_deviation 兼容
      4. 空 dict (后续 StrategyBase 用 params_spec 默认值)

    返回 dict; 顶层键与 StrategyBase.params_spec 对齐。
    """
    # 1. CLI --params 显式
    if params_arg:
        return _parse_params(params_arg)
    # 2. 默认参数落盘
    from .strategies._defaults_loader import exists, load
    if exists(strategy_name):
        data = load(strategy_name)
        return dict(data.get("params") or {})
    # 3. 旧 CLI kwargs 兼容 (channel_deviation 历史接口)
    if strategy_name == "channel_deviation" and legacy_low1 is not None:
        return {"low1": legacy_low1, "low2": legacy_low2,
                "high1": legacy_high1, "high2": legacy_high2}
    # 4. 空
    return {}


# ============ backtest 子命令 ============

def build_backtest_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="minute_bars 周期合并 + 通达信通道轨 (回测/实盘统一)")
    ap.add_argument("--period", default="5m", type=_period_type,
                    help="K线周期, 任意 数字+m/h/d: 5m/7m/15m/30m/90m/2h/4h/6h/1d/3d ...")
    ap.add_argument("--strategy", default="channel_deviation",
                    help="策略 key (来自 evtrade.strategies.available_strategies())")
    ap.add_argument("--params", default="",
                    help="策略参数 (通用 dict 形式): 'k1:v1;k2:v2' (分号分隔 kv, "
                         "类型自动推导 int/float/bool/str)。"
                         "例: --params 'low1:1.5;low2:1.0;high1:1.5;high2:0.5'")
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
    ap.add_argument("--all-in", action="store_true",
                    help="[阶段 2] 全仓模式: BUY 吃满现金 / SELL 清光持仓 (便捷开关, "
                         "等价 --buy-pct 1.0 --sell-pct 1.0)")
    ap.add_argument("--buy-pct", type=float, default=0.0,
                    help="[阶段 2] BUY 时按当前现金的该比例下注 (0=关闭走 --trade-qty, "
                         "0.5=半仓, 1.0=全仓)")
    ap.add_argument("--sell-pct", type=float, default=0.0,
                    help="[阶段 2] SELL 时按当前持仓的该比例卖 (0=关闭走 --trade-qty, "
                         "1.0=清仓)")
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
    # 策略参数: --params 显式 > _defaults 落盘 > 旧 CLI kwargs 兼容
    strategy_params = _resolve_strategy_params(
        args.strategy, args.params,
        legacy_low1=args.low1, legacy_low2=args.low2,
        legacy_high1=args.high1, legacy_high2=args.high2)

    from .core.data import load_bars
    from .core.kernel import bucket_table, make_state
    from .core.kernel_dsl import dsl_kernel, make_state_general, strategy_has_dsl

    # 无 DSL docstring 的策略不能进内核路径 (numba/CUDA 都由 DSL 渲染)
    if args.strategy != "channel_deviation" and not strategy_has_dsl(args.strategy):
        print(f"[警告] 策略 {args.strategy!r} 没有 DSL compute_signal docstring; "
              f"kernel 引擎不可用, 请用 --engine ref")
        return

    bars = load_bars(args.code, args.start, args.end,
                     warmup_days=args.warmup_days, cache_dir=args.data_cache)
    n = len(bars["stime"])
    show_bars = args.show_bars or args.bars_out
    print(f"证券: {args.code}  周期: {args.period}  策略日期: {args.start}~{args.end}  "
          f"预热: {args.warmup_days}天  TF1={args.tf1}  "
          f"策略: {args.strategy}  "
          f"scale={args.scale}  "
          f"资金模式: {'ALL-IN' if args.all_in else f'buy={args.buy_pct}/sell={args.sell_pct}'}  "
          f"引擎: kernel (numba)\n", flush=True)

    if args.strategy == "channel_deviation":
        # 冻结内核路径 (low1..high2 别名; 72 项差分锁定)
        st = make_state(period=args.period, warmup_until=int(args.start) * 1_000_000,
                        tf1=args.tf1, low1=args.low1, low2=args.low2,
                        high1=args.high1, high2=args.high2,
                        init_cash=INIT_CASH, init_position=INIT_POSITION,
                        trade_qty=args.trade_qty, scale=args.scale,
                        buy_pct=args.buy_pct, sell_pct=args.sell_pct,
                        all_in=args.all_in,
                        record_trades=True, trade_cap=n)
    else:
        # DSL 策略: 按策略 params_spec 顺序填 p0..pN, 内核段由 DSL 渲染
        st = make_state_general(args.strategy, period=args.period,
                                warmup_until=int(args.start) * 1_000_000,
                                tf1=args.tf1, init_cash=INIT_CASH,
                                init_position=INIT_POSITION,
                                trade_qty=args.trade_qty, scale=args.scale,
                                buy_pct=args.buy_pct, sell_pct=args.sell_pct,
                                all_in=args.all_in,
                                strategy_params=strategy_params,
                                record_trades=True, trade_cap=n)
    kmod = dsl_kernel(args.strategy)   # channel_deviation -> 冻结 kernel 本尊
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
        kmod.run_backtest_trace(st, bars["stime"], bars["open"], bars["high"],
                                bars["low"], bars["close"], bars["volume"],
                                sig_out, up_out, dw_out,
                                ts_out, o_out, h_out, l_out, c_out, v_out)
        tab = bucket_table(bars["stime"], sig_out, up_out, dw_out,
                           ts_out, o_out, h_out, l_out, c_out, v_out)
        # 只保留策略期 (--start 起) 的桶; 预热期仅用于指标准备, 不输出
        mask = tab["ts"] >= int(args.start) * 1_000_000
        tab = {k: v[mask] for k, v in tab.items()}
    else:
        kmod.run_backtest(st, bars["stime"], bars["open"], bars["high"],
                          bars["low"], bars["close"], bars["volume"],
                          np.empty(0, np.int8), np.empty(0), np.empty(0))
    dt = time.perf_counter() - t0

    for t in kmod.trades_to_list(st):
        side = "BUY " if t["side"] == "BUY" else "SELL"
        arrow = ">>" if t["side"] == "BUY" else ">>"
        print(f"        {arrow} {side} {t['qty']:.0f}股 @ {t['price']:.4f}  "
              f"[{t['ts']}]  剩余资金 {t['cash_after']:.2f}", flush=True)

    s = kmod.summarize(st)
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
    print(f"年化超额 (择时贡献)   : {s['ann_excess_pct']:+.2f}%/年")
    print(f"年化复合 CAGR         : {s['cagr']:+.2f}%/年")
    print(f"超额 Sharpe           : {s['sharpe_excess']:+.3f}")
    print(f"超额 Sortino          : {s['sortino_excess']:+.3f}")
    print(f"Calmar (年化/回撤)    : {s['calmar']:+.3f}")
    print(f"最大回撤 (逐bar盯市)   : {s['max_drawdown']:.2%}")
    print(f"最大回撤持续天数       : {s['max_dd_days']:.1f} 天  (恢复={s['max_dd_recovered']})")
    print(f"超额曲线最大回撤       : {s['x_mdd']:.2%}")
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
    from .frozen.account import Account
    from .frozen.aggregator import BarAggregator
    from .core.engine import Engine
    from .execution.base import SimulatedExecutor
    from .feeds.mysql_history import MySQLBacktestFeed
    from .strategies import get_strategy
    from .strategies import get_strategy as _gs

    delay = 0 if args.no_sleep else INTERVAL
    feed = MySQLBacktestFeed(code=args.code, start_ymd=args.start, end_ymd=args.end,
                             step_days=args.step_days, delay=delay, verbose=True)
    account = Account(cash=INIT_CASH, position=INIT_POSITION)
    executor = SimulatedExecutor(account, qty=args.trade_qty, verbose=True,
                                 scale=args.scale,
                                 buy_pct=args.buy_pct, sell_pct=args.sell_pct,
                                 all_in=args.all_in)
    aggregator = BarAggregator(resolve_period_seconds(args.period), on_bars=None,
                               warmup_until=feed.warmup_until)
    # 策略参数: --params 显式 > _defaults 落盘 > 旧 CLI kwargs 兼容
    strategy_params = _resolve_strategy_params(
        args.strategy, args.params,
        legacy_low1=args.low1, legacy_low2=args.low2,
        legacy_high1=args.high1, legacy_high2=args.high2)
    strategy = _gs(args.strategy, params=strategy_params)
    engine = Engine(feed, aggregator, strategy, executor, tf1=args.tf1, verbose=True)
    print(f"证券: {args.code}  周期: {args.period}  策略: {args.strategy}  "
          f"策略日期: {args.start}~{args.end}  "
          f"预热起点: {feed.warmup_start}  分段: {args.step_days}天/段(闭区间)  "
          f"TF1={args.tf1}  sleep={'OFF' if args.no_sleep else 'ON'}  "
          f"low1/low2={args.low1}/{args.low2} high1/high2={args.high1}/{args.high2}  "
          f"scale={args.scale}  "
          f"资金模式: {'ALL-IN' if args.all_in else f'buy={args.buy_pct}/sell={args.sell_pct}'}  "
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
    ap.add_argument("--strategy", default="channel_deviation",
                    help="策略 key (来自 evtrade.strategies.available_strategies())")
    ap.add_argument("--params", default="",
                    help="基础策略参数: 'k1:v1;k2:v2' (与 --grid 笛卡尔积叠加)")
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
    ap.add_argument("--all-in", action="store_true",
                    help="全仓模式 (等价 --buy-pct 1.0 --sell-pct 1.0)")
    ap.add_argument("--buy-pct", type=float, default=0.0,
                    help="BUY 时按当前现金的该比例下注 (0=关闭)")
    ap.add_argument("--sell-pct", type=float, default=0.0,
                    help="SELL 时按当前持仓的该比例卖 (0=关闭)")
    ap.add_argument("--grid", action="append", default=[],
                    help="参数网格, 可多次: --grid low1=1.0,1.5,2.0 "
                         "(支持 low1/low2/high1/high2/tf1/period/trade_qty, "
                         "及该策略 params_spec 声明的参数名)")
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
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "gpu"],
                    help="auto=优先 GPU (可用且策略兼容), 否则 cpu (默认); "
                         "cpu=numba内核+线程池 / gpu=CUDA单launch (需 cupy, 见 gpu.py)")
    ap.add_argument("--data-cache", default=None, help="行情 npz 缓存目录")
    ap.add_argument("--synthetic-days", type=int, default=0,
                    help=">0 时用合成数据 (不连库); 数值=截至 --end 的数据天数, "
                         "应覆盖预热需求")
    ap.add_argument("--out", default="sweep_results.csv", help="结果 CSV 路径")
    ap.add_argument("--top", type=int, default=20, help="控制台展示前 N 组")
    ap.add_argument("--save-defaults", action="store_true", default=False,
                    help="扫描后自动选最优 (filter_pass 优先 / 否则 score 第一行), "
                         "写入 evtrade/strategies/_defaults/<strategy>.json, "
                         "并尝试单独 git commit (中文 message 含选择原因)。"
                         "不传本参数 = 仅写 CSV, 不动默认参数文件。")
    ap.add_argument("--no-save-defaults", dest="save_defaults",
                    action="store_false",
                    help="明确跳过默认参数落盘 (与不传 --save-defaults 等价)。")
    return ap


def sweep_main(argv=None):
    args = build_sweep_parser().parse_args(argv)
    from .data import load_bars, synthetic_bars
    from .kernel import bars_to_arrays
    from .sweep import GRID_KEYS, parse_grid, sweep as run_sweep
    from .strategies import get_strategy_param_spec

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

    # 基础策略参数: --params 显式 > _defaults 落盘 > 旧 CLI kwargs 兼容 > 空
    base_params = _resolve_strategy_params(
        args.strategy, args.params,
        legacy_low1=args.low1, legacy_low2=args.low2,
        legacy_high1=args.high1, legacy_high2=args.high2)

    base = {"start": args.start, "period": args.period, "tf1": args.tf1,
            "low1": args.low1, "low2": args.low2, "high1": args.high1,
            "high2": args.high2, "trade_qty": args.trade_qty,
            "scale": args.scale,
            "buy_pct": args.buy_pct, "sell_pct": args.sell_pct,
            "all_in": args.all_in,
            "init_cash": INIT_CASH, "init_position": INIT_POSITION,
            "params": base_params}
    # 网格 key = 内置 GRID_KEYS + 该策略 params_spec 的参数名 (--grid lookback=10,20)
    spec_keys = set(get_strategy_param_spec(args.strategy))
    combos = parse_grid(args.grid, extra_keys=spec_keys) or [{}]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()] if args.splits else None
    if splits:
        print(f"扫描 {len(combos)} 组参数 (滚动 WFO {len(splits)} 窗: "
              f"train + test1..test{len(splits)}) ...\n", flush=True)
    else:
        print(f"扫描 {len(combos)} 组参数 (单窗) ...\n", flush=True)
    df = run_sweep(bars, base, combos, split_ymd=args.split,
                   n_workers=args.workers, device=args.device,
                   splits=splits, fee_bp=args.fee_bp, lam=args.score_lambda,
                   min_trades=args.min_trades, max_mdd=args.max_mdd,
                   strategy_name=args.strategy)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    cols = [c for c in df.columns]
    show = [c for c in df.columns
            if c in ("score", "ann_net_min", "ann_net_mean", "pos_ratio",
                     "sharpe_min", "sortino_min", "calmar_max", "cagr_max",
                     "max_dd_days_max", "x_mdd_max", "S", "pareto", "filter_pass")
            or c in GRID_KEYS or c in spec_keys]
    print(df[show].head(args.top).to_string(index=False))
    print(f"\n全部结果已保存: {args.out} ({len(df)} 行; 列: {cols})")

    # ---- 自动选最优 + 落盘 + 单独 git commit (--save-defaults) ----
    if args.save_defaults:
        from .strategies._defaults_loader import (
            save_best_from_sweep, format_reason_log,
        )
        chosen, reason, saved_path, commit_ok = save_best_from_sweep(
            df, args.strategy, csv_path=args.out)
        if chosen is None:
            print(f"[警告] sweep 自动选最优失败: {reason.get('error', '?')}",
                  flush=True)
        else:
            print(format_reason_log(args.strategy, reason,
                                    saved_path, commit_ok),
                  flush=True)

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
    ap.add_argument("--all-in", action="store_true",
                    help="全仓模式 (等价 buy_pct=sell_pct=1)")
    ap.add_argument("--buy-pct", type=float, default=0.0)
    ap.add_argument("--sell-pct", type=float, default=0.0)
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
          f"scale={args.scale}  "
          f"资金模式: {'ALL-IN' if args.all_in else f'buy={args.buy_pct}/sell={args.sell_pct}'}\n",
          flush=True)

    k = replay_kernel(bars, args.period, warm, args.tf1, args.low1, args.low2,
                      args.high1, args.high2, scale=args.scale,
                      buy_pct=args.buy_pct, sell_pct=args.sell_pct, all_in=args.all_in)
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
                  args.high1, args.high2, scale=args.scale,
                  buy_pct=args.buy_pct, sell_pct=args.sell_pct, all_in=args.all_in)


# ============ params 子命令 (默认参数落盘) ============

def build_params_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="evtrade params",
        description="策略默认参数管理 (落盘 evtrade/strategies/_defaults/<name>.json)")
    sub = ap.add_subparsers(dest="params_cmd", required=True)

    # save
    p_save = sub.add_parser("save", help="保存最优参数到默认目录")
    p_save.add_argument("strategy", help="策略 key (来自 available_strategies())")
    src = p_save.add_mutually_exclusive_group(required=True)
    src.add_argument("--params", default="",
                    help="'k1:v1;k2:v2' (类型自动推导, 同 backtest)")
    src.add_argument("--from-csv", default=None,
                    help="sweep 结果 CSV 路径; 与 --rank 配合取第 N 行")
    p_save.add_argument("--rank", type=int, default=1,
                    help="--from-csv 时取第 N 行 (1=最高 score, 默认 1)")

    # show
    p_show = sub.add_parser("show", help="打印策略当前默认参数")
    p_show.add_argument("strategy", help="策略 key")

    # list
    p_list = sub.add_parser("list", help="列出所有已有默认参数的策略")
    return ap


def params_main(argv=None):
    args = build_params_parser().parse_args(argv)

    if args.params_cmd == "list":
        from .strategies._defaults_loader import list_defaulted
        names = list_defaulted()
        if not names:
            print("(无默认参数文件; evtrade/strategies/_defaults/ 为空)")
            return
        print("已落盘默认参数的策略:")
        for n in names:
            print(f"  - {n}")
        return

    if args.params_cmd == "show":
        from .strategies._defaults_loader import load
        try:
            data = load(args.strategy)
        except FileNotFoundError as e:
            print(f"[错误] {e}", flush=True)
            return 1
        print(f"策略: {data.get('strategy', args.strategy)}")
        print(f"落盘: {data.get('saved_at', '?')}")
        print("参数:")
        for k, v in (data.get("params") or {}).items():
            print(f"  {k} = {v!r}")
        src = data.get("source") or {}
        if src:
            print(f"来源: {src}")
        return

    if args.params_cmd == "save":
        from .strategies._defaults_loader import save, params_from_csv_row, path_for
        if args.from_csv:
            import csv as _csv
            with open(args.from_csv, "r", encoding="utf-8-sig", newline="") as f:
                rows = list(_csv.DictReader(f))
            if not rows:
                print(f"[错误] {args.from_csv} 为空", flush=True)
                return 1
            idx = args.rank - 1
            if not (0 <= idx < len(rows)):
                print(f"[错误] --rank {args.rank} 越界 (共 {len(rows)} 行)",
                      flush=True)
                return 1
            row = rows[idx]
            # param_keys 顺序来自策略的 params_spec; 让 _defaults_loader 按已知
            # 字段抽出; 缺则退回到 row 全部键 (含 tf1 等引擎参数, 但 _defaults
            # 的 params 仅策略参数, 故用 specs 约束)。
            from .strategies import get_strategy_param_spec
            spec = get_strategy_param_spec(args.strategy)
            param_keys = list(spec.keys())
            params = params_from_csv_row(row, param_keys)
            if not params:
                print(f"[警告] CSV 行 {args.rank} 没有命中策略 params_spec "
                      f"({param_keys}); 不落盘", flush=True)
                return 1
            source = {"kind": "from_csv", "csv": args.from_csv,
                      "rank": args.rank}
        else:
            params = _parse_params(args.params)
            if not params:
                print("[错误] --params 为空或解析失败", flush=True)
                return 1
            source = {"kind": "from_params"}

        p = save(args.strategy, params, source=source)
        print(f"[已落盘] {p}")
        print("参数:")
        for k, v in params.items():
            print(f"  {k} = {v!r}")
        return


def build_root_parser():
    """根 parser: 用 add_subparsers 让 backtest/sweep/replay 各自独立 --help

    子命令的 build_*_parser 提供各自参数; 子命令分发通过 cmd 字段决定
    调用哪个 main。
    """
    ap = argparse.ArgumentParser(
        prog="evtrade",
        description="evtrade CLI: 策略回测 / 参数扫描 / 行情回放",
    )
    sub = ap.add_subparsers(dest="cmd", help="子命令")

    # 复用各子命令的 build_*_parser: 它们返回 ArgumentParser, 我们拿它的
    # _actions 嫁接到子 parser 上, 避免重复定义参数。
    for name, builder in (("backtest", build_backtest_parser),
                          ("sweep", build_sweep_parser),
                          ("replay", build_replay_parser),
                          ("params", build_params_parser)):
        sub_p = sub.add_parser(name, help=f"{name} 子命令 (见 {name} -h)",
                               add_help=False)
        for action in builder()._actions:
            # 把所有 action 复制过来 (add_argument 已在各 builder 里)
            sub_p._add_action(action)
    return ap


def main(argv=None):
    import sys
    raw = sys.argv[1:] if argv is None else argv
    # 默认行为: 无子命令 = backtest (向后兼容, 不破坏现有脚本调用)
    if not raw or raw[0] not in ("backtest", "sweep", "replay", "params",
                                 "-h", "--help"):
        if raw and raw[0].startswith("-"):
            # 形如 -h / --help 等根选项, 走 root parser 展示帮助
            return build_root_parser().parse_args(raw)
        if raw:
            # 把第一个位置参数当 backtest 的策略名等看待, 走 backtest 子命令
            return backtest_main(raw)
        # 完全无参数: 显示 root help
        build_root_parser().print_help()
        return None
    args = build_root_parser().parse_args(raw)
    handlers = {"backtest": backtest_main, "sweep": sweep_main,
                "replay": replay_main, "params": params_main}
    return handlers[args.cmd](raw[1:])


if __name__ == "__main__":
    main()
