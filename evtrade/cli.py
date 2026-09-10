from __future__ import annotations
"""命令行入口 (DSL/numba 已下线; 唯一路径 = vectorized)

================================================================
✅  可改层模块  ✅  (用户面的主要修改点)
================================================================
本文件定义四个子命令: backtest / sweep / replay / params。

唯一执行路径 (CPU/GPU 统一):
  - backtest: vectorized_engine.run_vectorized
  - sweep:    core.sweep.sweep (内部走 run_one_vectorized)
  - replay:   replay.replay_vectorized (--against-ref 加 replay.reconcile)

设备选择: --device {cpu, gpu, auto} (默认 auto)
  - auto: 优先 gpu (cupy 可用), 否则 cpu
  - cpu:  xp = numpy
  - gpu:  xp = cupy (需 cupy + CUDA)

常见修改:
  1. 加新参数: 在 build_*_parser 加 add_argument; 在对应的 _run_* 中读取并透传。
  2. 改输出格式: _run_backtest() 末尾的 print 段。
  3. 加新子命令: 在 main() 的 handlers 字典加一项。
"""

import argparse
import time

from .core.config import INIT_CASH, INIT_POSITION, INTERVAL, TF1, TRADE_QTY
from .core.timeutils import resolve_period_seconds


def _parse_params(spec: str) -> dict:
    """'k1:v1;k2:v2' -> dict (类型自动推导: int / float / bool / str)"""
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
    resolve_period_seconds(s)
    return s


def _resolve_strategy_params(strategy_name: str, params_arg: str) -> dict:
    """解析策略参数, 优先级 (高 -> 低):
      1. CLI --params 显式传入
      2. evtrade/strategies/_defaults/<strategy_name>.json
      3. 空 dict (后续 _resolve_params 用 params_spec 默认值)
    """
    if params_arg:
        return _parse_params(params_arg)
    from .strategies._defaults_loader import exists, load
    if exists(strategy_name):
        data = load(strategy_name)
        return dict(data.get("params") or {})
    return {}


def _emit_deprecation_warning(old_key: str, new_key: str, mapping: dict) -> None:
    """打印 --engine / 旧 key 的 deprecation 警告 + 自动转换"""
    import warnings
    warnings.warn(
        f"--{old_key} 已下线 (DSL/numba 已下线), "
        f"请改用 --{new_key} {{{', '.join(sorted(mapping))}}}; "
        f"当前按映射自动转换。",
        DeprecationWarning,
        stacklevel=3,
    )


def _coalesce_legacy_engine(argv: list[str]) -> tuple[list[str], str]:
    """检测旧 --engine 值并转换为 --device; 返回 (new_argv, device)

    映射: kernel -> auto, ref -> cpu, vectorized -> cpu
    """
    new_argv = []
    device = "auto"
    skip_next = False
    for i, a in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if a == "--engine":
            # 下一个 argv[i+1] 是 value
            if i + 1 < len(argv):
                v = argv[i + 1].lower()
                mapping = {"kernel": "auto", "ref": "cpu", "vectorized": "cpu"}
                if v in mapping:
                    device = mapping[v]
                    _emit_deprecation_warning("engine", "device", mapping)
                    skip_next = True
                    continue
        new_argv.append(a)
    return new_argv, device


# ============ backtest 子命令 ============

def build_backtest_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="策略回测 (vectorized; CPU=xp=numpy / GPU=xp=cupy)")
    ap.add_argument("--period", default="5m", type=_period_type,
                    help="K线周期, 任意 数字+m/h/d: 5m/7m/15m/30m/90m/2h/4h/6h/1d/3d ...")
    ap.add_argument("--strategy", default="channel_deviation",
                    help="策略 key (来自 evtrade.strategies.available_strategies())")
    ap.add_argument("--params", default="",
                    help="策略参数 (通用 dict 形式): 'k1:v1;k2:v2'")
    ap.add_argument("--start", default="20250101", help="策略起始日期 YYYYMMDD")
    ap.add_argument("--end", default="20260903", help="策略结束日期 YYYYMMDD")
    ap.add_argument("--tf1", type=int, default=TF1, help="EMA 周期 (策略/指标层)")
    ap.add_argument("--trade-qty", type=float, default=TRADE_QTY, help="每次信号交易股数")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="倍投系数: 连续同向信号数量=上次×scale (反向重置); 1.0=关闭")
    ap.add_argument("--all-in", action="store_true",
                    help="全仓模式 (等价 --buy-pct 1.0 --sell-pct 1.0)")
    ap.add_argument("--buy-pct", type=float, default=0.0,
                    help="BUY 时按当前现金的该比例下注 (0=关闭走 --trade-qty)")
    ap.add_argument("--sell-pct", type=float, default=0.0,
                    help="SELL 时按当前持仓的该比例卖 (0=关闭走 --trade-qty)")
    ap.add_argument("--code", default="159992.SZ", help="证券代码")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"],
                    help="xp 后端: cpu=numpy / gpu=cupy / auto=优先 gpu (默认)")
    ap.add_argument("--warmup-days", type=int, default=365,
                    help="预热天数 (拉取 start 之前的行情供指标就绪)")
    ap.add_argument("--data-cache", default=None,
                    help="行情 npz 缓存目录")
    ap.add_argument("--show-bars", action="store_true",
                    help="打印每根周期K线的 OHLCV 与策略额外列")
    ap.add_argument("--bars-out", default=None,
                    help="同 --show-bars 内容输出 CSV")
    ap.add_argument("--signals-out", default=None,
                    help="信号轨迹 CSV (ts,sig,策略额外列)")
    ap.add_argument("--init-cash", type=float, default=INIT_CASH,
                    help="期初资金 (默认 20万); 传 0 忽略, 走零起点")
    ap.add_argument("--init-position", type=float, default=INIT_POSITION,
                    help="期初持仓股数 (默认 20万); 传 0 忽略, 走零起点")
    # --- 兼容层: --engine 已下线 (DSL/numba 已删除); 仅打 DeprecationWarning + 自动映射 device ---
    ap.add_argument("--engine", default=None, choices=["kernel", "ref", "vectorized"],
                    help=argparse.SUPPRESS)
    # --- 兼容层: --no-sleep / --step-days 是 ref 引擎参数, 已下线 ---
    ap.add_argument("--no-sleep", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--step-days", type=int, default=7, help=argparse.SUPPRESS)
    return ap


def _f4(v) -> str:
    """数值格式化: NaN -> ----"""
    try:
        if v != v:
            return "----"
    except TypeError:
        return "----"
    return f"{v:.4f}"


def _run_backtest(args):
    """统一 backtest 入口 (vectorized 引擎)"""
    # 兼容旧 --engine: 自动映射到 --device (kernel->auto, ref->cpu, vectorized->cpu)
    if getattr(args, "engine", None):
        mapping = {"kernel": "auto", "ref": "cpu", "vectorized": "cpu"}
        mapped = mapping[args.engine]
        if args.device == "auto" or args.device != mapped:
            _emit_deprecation_warning("engine", "device", mapping)
            args.device = mapped

    strategy_params = _resolve_strategy_params(args.strategy, args.params)

    from .core.data import load_bars
    from .core.vectorized_engine import run_vectorized
    from .strategies import get_strategy

    strategy = get_strategy(args.strategy, params=strategy_params)
    bars = load_bars(args.code, args.start, args.end,
                     warmup_days=args.warmup_days, cache_dir=args.data_cache)
    n = len(bars["stime"])
    print(f"证券: {args.code}  周期: {args.period}  策略日期: {args.start}~{args.end}  "
          f"预热: {args.warmup_days}天  TF1={args.tf1}  "
          f"策略: {args.strategy}  "
          f"scale={args.scale}  "
          f"资金模式: {'ALL-IN' if args.all_in else f'buy={args.buy_pct}/sell={args.sell_pct}'}  "
          f"device={args.device}\n", flush=True)

    t0 = time.perf_counter()
    buy_pct = max(args.buy_pct, 1.0) if args.all_in else args.buy_pct
    sell_pct = max(args.sell_pct, 1.0) if args.all_in else args.sell_pct
    result = run_vectorized(
        bars, period=args.period, warmup_until=int(args.start) * 1_000_000,
        strategy=strategy, params=strategy.params,
        init_cash=args.init_cash, init_position=args.init_position,
        trade_qty=args.trade_qty, scale=args.scale,
        buy_pct=buy_pct, sell_pct=sell_pct,
        device=args.device)
    dt = time.perf_counter() - t0

    s = result["summary"]
    for t in s["trades"]:
        side = "BUY " if t["side"] == "BUY" else "SELL"
        print(f"        >> {side} {t['qty']:.0f}股 @ {t['price']:.4f}  "
              f"[{t['ts']}]  剩余资金 {t['cash_after']:.2f}", flush=True)

    print("\n" + "=" * 60)
    print("回测盈亏汇总 (vectorized) [25 字段]")
    print("=" * 60)
    print(f"期末价 (最后一根close) : {s['final_price']:.4f}")
    print(f"交易次数              : {s['n_trades']} (BUY {s['n_buy']} / SELL {s['n_sell']})")
    print(f"期初资金 / 期初持仓    : {INIT_CASH:.0f} / {INIT_POSITION:.0f}股")
    print(f"期末资金 / 期末持仓    : {s['final_cash']:.2f} / {s['final_position']:.0f}股")
    print(f"期末持仓市值           : {s['final_position'] * s['final_price']:.2f}")
    print(f"策略总资产 (资金+市值) : {s['final_equity']:.2f}")
    print(f"不操作基线 (资金+市值) : {s['baseline']:.2f}")
    print(f"盈亏比例              : {s['excess_pct']:+.2f}%")
    print("-" * 60)
    print(f"跨度 years            : {s['years']:.3f}")
    print(f"年化复合 CAGR         : {s['cagr']:+.2f}%/年")
    print(f"年化超额 CAGR (复合)   : {s['cagr_excess']:+.2f}%/年")
    print(f"超额 Sharpe / IR      : {s['sharpe_excess']:+.3f} / {s['ir']:+.3f}")
    print(f"超额 Sortino          : {s['sortino_excess']:+.3f}")
    print(f"Calmar (年化/回撤)    : {s['calmar']:+.3f}")
    print("-" * 60)
    print(f"最大回撤 (策略)        : {s['max_drawdown']:+.2%}")
    print(f"最大回撤 (基准)        : {s['baseline_max_dd']:+.2%}")
    print(f"回撤差 (策略-基准)     : {s['dd_excess']:+.2%}  (正值=策略比基准更深)")
    print(f"最大回撤持续天数       : {s['max_dd_days']:.1f} 天")
    # max_dd_recovered: -1 sentinel 表示从未恢复
    dd_recovered = "未恢复" if s['max_dd_recovered'] == -1 else f"{s['max_dd_recovered']} 桶"
    print(f"最大回撤恢复 (trough→) : {dd_recovered}")
    print("-" * 60)
    # 持仓行为: 仅在有成交时打印关键值
    pf_str = "inf" if s['profit_factor'] == float('inf') else f"{s['profit_factor']:.2f}"
    print(f"胜率 / 盈亏比         : {s['win_rate']:.1%} / {pf_str}")
    print(f"平均每笔 PnL          : {s['avg_pnl']:+.2f}")
    print(f"最大连盈 / 连亏笔数   : {s['max_consecutive_wins']} / {s['max_consecutive_losses']}")
    print(f"平均 / 最大持仓周期    : {s['avg_hold_bars']:.1f} / {s['max_hold_bars']} 桶")
    print("-" * 60)
    print(f"成交额合计            : {s['turnover']:.0f}")
    print(f"引擎耗时              : {dt * 1000:.1f} ms ({n} 根 1m bar, {args.device})")
    print("=" * 60)

    if args.signals_out:
        sig = result["sig"]
        stime = bars["stime"]
        with open(args.signals_out, "w", encoding="utf-8-sig") as f:
            f.write("stime,signal\n")
            for i in range(len(stime)):
                f.write(f"{int(stime[i])},{int(sig[i])}\n")
        print(f"信号轨迹已保存: {args.signals_out} ({len(stime)} 行)")


def backtest_main(argv=None):
    """backtest 主入口 (检测旧 --engine 自动转换)"""
    raw = list(argv) if argv is not None else None
    if raw is not None:
        raw, device = _coalesce_legacy_engine(raw)
        if device != "auto":
            # 把默认 device 改成探测出来的 device; 但仍允许 --device 显式覆盖
            # 这里仅在 argv 没有 --device 时应用
            if "--device" not in raw:
                raw = ["--device", device] + raw
    args = build_backtest_parser().parse_args(raw)
    _run_backtest(args)


# ============ sweep 子命令 ============

def build_sweep_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="参数并发扫描 (vectorized; CPU/GPU 统一)")
    ap.add_argument("--strategy", default="channel_deviation")
    ap.add_argument("--params", default="",
                    help="基础策略参数: 'k1:v1;k2:v2' (与 --grid 笛卡尔积叠加)")
    ap.add_argument("--code", default="159992.SZ")
    ap.add_argument("--start", default="20250101")
    ap.add_argument("--end", default="20260903")
    ap.add_argument("--period", default="5m", type=_period_type)
    ap.add_argument("--tf1", type=int, default=TF1)
    ap.add_argument("--trade-qty", type=float, default=TRADE_QTY)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--all-in", action="store_true")
    ap.add_argument("--buy-pct", type=float, default=0.0)
    ap.add_argument("--sell-pct", type=float, default=0.0)
    ap.add_argument("--grid", action="append", default=[],
                    help="参数网格, 可多次: --grid key=v1,v2,v3")
    ap.add_argument("--split", default=None,
                    help="单分割日 YYYYMMDD")
    ap.add_argument("--splits", default=None,
                    help="滚动 WFO 分割日, 逗号分隔")
    ap.add_argument("--fee-bp", type=float, default=5.0)
    ap.add_argument("--score-lambda", type=float, default=1.0)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--max-mdd", type=float, default=1.0)
    ap.add_argument("--mc", type=int, default=0)
    ap.add_argument("--mc-top", type=int, default=5)
    ap.add_argument("--warmup-days", type=int, default=365)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "gpu"],
                    help="xp 后端: cpu / gpu / auto (默认)")
    ap.add_argument("--data-cache", default=None)
    ap.add_argument("--synthetic-days", type=int, default=0,
                    help=">0 时用合成数据 (不连库)")
    ap.add_argument("--out", default="sweep_results.csv")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--save-defaults", action="store_true", default=False)
    ap.add_argument("--no-save-defaults", dest="save_defaults", action="store_false")
    return ap


def sweep_main(argv=None):
    args = build_sweep_parser().parse_args(argv)
    from .core.data import load_bars, synthetic_bars
    from .core.metrics import bars_to_arrays
    from .core.sweep import GRID_KEYS, parse_grid, sweep as run_sweep
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

    base_params = _resolve_strategy_params(args.strategy, args.params)
    base = {"start": args.start, "period": args.period, "tf1": args.tf1,
            "trade_qty": args.trade_qty, "scale": args.scale,
            "buy_pct": args.buy_pct, "sell_pct": args.sell_pct,
            "all_in": args.all_in,
            "init_cash": INIT_CASH, "init_position": INIT_POSITION,
            "params": base_params}
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
        from .core.permutation import permutation_test
        print(f"\n蒙特卡洛置换检验 (前 {args.mc_top} 名, 各打乱 {args.mc} 次):")
        warm = int(base["start"]) * 1_000_000
        for _, row in df.head(args.mc_top).iterrows():
            combo = {k: row[k] for k in GRID_KEYS if k in row}
            params = {**base, **combo}
            r = permutation_test(bars, params, warm, n=args.mc,
                                 fee_bp=args.fee_bp)
            print(f"  {' '.join(f'{k}={row[k]}' for k in combo)}  "
                  f"真实年化超额 {r['real_ann_net']:+.2f}%/年  "
                  f"p={r['p_value']:.3f}", flush=True)


# ============ replay 子命令 ============

def build_replay_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="录制回放对账 (vectorized 引擎; --against-ref 走 Engine.on_bars 对账)")
    ap.add_argument("--log", required=True,
                    help="bar 日志 (CSV: stime,code,open,high,low,close,volume)")
    ap.add_argument("--strategy", default="channel_deviation")
    ap.add_argument("--period", default="5m", type=_period_type)
    ap.add_argument("--tf1", type=int, default=TF1)
    ap.add_argument("--params", default="",
                    help="策略参数 (通用 dict 形式)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--all-in", action="store_true")
    ap.add_argument("--buy-pct", type=float, default=0.0)
    ap.add_argument("--sell-pct", type=float, default=0.0)
    ap.add_argument("--warmup-until", default=None)
    ap.add_argument("--against-ref", action="store_true",
                    help="同时用 Engine.on_bars 回放并逐 bar 对账")
    ap.add_argument("--signals-out", default=None, help="信号轨迹 CSV")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    return ap


def replay_main(argv=None):
    args = build_replay_parser().parse_args(argv)
    from .core.replay import (
        read_bars_log, reconcile, replay_vectorized,
    )
    from .strategies import get_strategy

    strategy_name = args.strategy
    sp = _resolve_strategy_params(strategy_name, args.params)

    bars = read_bars_log(args.log)
    if not bars:
        raise SystemExit("日志为空")
    warm = int(args.warmup_until) * 1_000_000 if args.warmup_until else 0
    print(f"回放: {len(bars)} 根 bar [{bars[0].stime} ~ {bars[-1].stime}]  "
          f"period={args.period} tf1={args.tf1} 策略={strategy_name} "
          f"params={sp or '(默认)'}  scale={args.scale}  device={args.device}\n",
          flush=True)

    k = replay_vectorized(bars, args.period, warm,
                          strategy_name=strategy_name, strategy_params=sp,
                          scale=args.scale,
                          buy_pct=args.buy_pct, sell_pct=args.sell_pct,
                          device=args.device)
    s = k["summary"]
    n_sig = int((k["sig"] != 0).sum()) if hasattr(k["sig"], "__len__") else 0
    print(f"信号 {n_sig} 个 (BUY {s['n_buy']} / SELL {s['n_sell']}), "
          f"成交 {s['n_trades']} 笔, 期末总资产 {s['final_equity']:,.2f} "
          f"(基线 {s['baseline']:,.2f}, 超额 {s['excess_pct']:+.2f}%)")

    if args.signals_out:
        strategy = get_strategy(strategy_name, params=sp or {})
        extra_cols = strategy.get_extra_signal_columns(sig=k["sig"])
        with open(args.signals_out, "w", encoding="utf-8-sig") as f:
            cols = ["stime", "signal"] + list(extra_cols.keys())
            f.write(",".join(cols) + "\n")
            extras = [v.tolist() if hasattr(v, "tolist") else list(v)
                      for v in extra_cols.values()]
            for i, b in enumerate(bars):
                row = f"{b.stime},{int(k['sig'][i])}"
                for arr in extras:
                    row += f",{arr[i]}"
                f.write(row + "\n")
        print(f"信号轨迹已保存: {args.signals_out}")

    if args.against_ref:
        print()
        reconcile(bars, args.period, warm, args.tf1,
                  strategy_name=strategy_name, strategy_params=sp,
                  scale=args.scale,
                  buy_pct=args.buy_pct, sell_pct=args.sell_pct,
                  all_in=args.all_in,
                  device=args.device)


# ============ params 子命令 (默认参数落盘) ============

def build_params_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="evtrade params",
        description="策略默认参数管理 (落盘 evtrade/strategies/_defaults/<name>.json)")
    sub = ap.add_subparsers(dest="params_cmd", required=True)

    p_save = sub.add_parser("save", help="保存最优参数到默认目录")
    p_save.add_argument("strategy", help="策略 key")
    src = p_save.add_mutually_exclusive_group(required=True)
    src.add_argument("--params", default="")
    src.add_argument("--from-csv", default=None)
    p_save.add_argument("--rank", type=int, default=1)

    p_show = sub.add_parser("show", help="打印策略当前默认参数")
    p_show.add_argument("strategy")

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
        from .strategies._defaults_loader import save, params_from_csv_row
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
    ap = argparse.ArgumentParser(
        prog="evtrade",
        description="evtrade CLI: 策略回测 / 参数扫描 / 行情回放 / 默认参数管理",
    )
    sub = ap.add_subparsers(dest="cmd", help="子命令")
    for name, builder in (("backtest", build_backtest_parser),
                          ("sweep", build_sweep_parser),
                          ("replay", build_replay_parser),
                          ("params", build_params_parser)):
        sub_p = sub.add_parser(name, help=f"{name} 子命令 (见 {name} -h)",
                               add_help=False)
        for action in builder()._actions:
            sub_p._add_action(action)
    return ap


def main(argv=None):
    import sys
    raw = sys.argv[1:] if argv is None else argv
    if not raw or raw[0] not in ("backtest", "sweep", "replay", "params",
                                 "-h", "--help"):
        if raw and raw[0].startswith("-"):
            return build_root_parser().parse_args(raw)
        if raw:
            return backtest_main(raw)
        build_root_parser().print_help()
        return None
    args = build_root_parser().parse_args(raw)
    handlers = {"backtest": backtest_main, "sweep": sweep_main,
                "replay": replay_main, "params": params_main}
    return handlers[args.cmd](raw[1:])


if __name__ == "__main__":
    main()
