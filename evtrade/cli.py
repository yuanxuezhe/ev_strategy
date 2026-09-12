from __future__ import annotations
"""命令行入口: backtest / sweep / params 三个子命令 (2026-09-13 重构)

framework 不持有资金/撮合/PnL/收益概念; CLI 不再有 --init-cash / --buy-pct /
--trade-qty 等参数; 策略资金/持仓/撮合/PnL 由策略 step 自管理。

唯一执行路径 (CPU/GPU 统一走 vectorized 引擎):
  - backtest: vectorized_engine.run_vectorized (仅驱动 step, 返回 final_state 透传)
  - sweep:    core.sweep.sweep (内部走 run_vectorized)

设备选择: --device {cpu, gpu, auto} (默认 auto)
"""
import argparse
import time

import numpy as np

from .core.data import DB_URL, TABLE
from .core.timeutils import resolve_period_seconds


def _parse_params(spec: str) -> dict:
    """'k1:v1;k2:v2' -> dict (类型自动推导: int / float / bool / str)"""
    from .strategies._defaults_loader import auto_cast
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
        out[k.strip()] = auto_cast(v.strip())
    return out


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


# ============ 共享选项父 parser (argparse parents= 单点声明) ============

def _common_parent() -> argparse.ArgumentParser:
    """backtest / sweep 共享选项 (声明一次, 两处 parents= 引用)

    2026-09-13 重构: 不再有 --init-cash / --buy-pct / --sell-pct / --all-in /
    --trade-qty / --warmup-days / --data-cache (framework 不再管这些)
    """
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--strategy", default="channel_deviation",
                    help="策略 key (来自 evtrade.strategies.available_strategies())")
    ap.add_argument("--params", default="",
                    help="策略参数 (通用 dict 形式): 'k1:v1;k2:v2'")
    ap.add_argument("--period", default="5m", type=_period_type,
                    help="K线周期, 任意数字+m/h/d: 5m/7m/15m/30m/90m/2h/4h/6h/1d/3d ...")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"],
                    help="计算后端: cpu / gpu / auto=优先 gpu (默认)")
    ap.add_argument("--signals-out", default=None,
                    help="信号轨迹 CSV (ts,sig); framework 仅透传策略 sig")
    return ap


def _data_parent() -> argparse.ArgumentParser:
    """backtest / sweep 共享的行情拉取选项 (声明一次, 两处 parents= 引用)

    2026-09-13 重构: --warmup-days / --trade-qty / --data-cache 删除
    """
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--code", default="159992.SZ", help="证券代码")
    ap.add_argument("--start", default="20250101", help="开始日期 YYYYMMDD")
    ap.add_argument("--end", default="20260903", help="结束日期 YYYYMMDD")
    ap.add_argument("--synthetic-days", type=int, default=0,
                    help=">0 时用合成数据 (不连库, A 股交易时段 seed=42)")
    return ap


# ============ backtest 子命令 ============

def build_backtest_parser(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """backtest 选项 (注册到给定 parser; prog/description 在 backtest_main)"""
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="逐根打印 strategy.format_signal_line 输出 (sig!=0 时)")
    return ap


def _write_signals_csv(path: str, rows: list[tuple]) -> None:
    """写信号轨迹 CSV (stime,signal; utf-8-sig 供 Excel 直开)"""
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("stime,signal\n")
        for t, s in rows:
            f.write(f"{t},{s}\n")
    print(f"信号轨迹已保存: {path} ({len(rows)} 行)")


def _run_backtest(args):
    """backtest 入口 (vectorized 引擎; 仅驱动 step)"""
    # 设备解析: --device gpu 无 CUDA 抛错; auto 降级
    from .backends import gpu_available, resolve_device
    args.device = resolve_device(args.device, gpu_available())

    strategy_params = _resolve_strategy_params(args.strategy, args.params)

    from .core.data import load_bars
    from .core.vectorized_engine import run_vectorized
    from .strategies import get_strategy

    strategy = get_strategy(args.strategy, params=strategy_params)
    if args.synthetic_days > 0:
        from datetime import datetime, timedelta
        from .core.data import synthetic_bars
        d_end = datetime.strptime(args.end, "%Y%m%d")
        d_start = d_end - timedelta(days=args.synthetic_days)
        print(f"生成合成数据: {d_start:%Y%m%d} ~ {args.end} "
              f"({args.synthetic_days} 天, seed=42)", flush=True)
        from .primitives import Bar
        synth = synthetic_bars(days=args.synthetic_days,
                               start_ymd=f"{d_start:%Y%m%d}")
        # 转 numpy dict (与 load_bars 同构)
        n = len(synth)
        arrays = {
            "stime": np.array([int(b.stime) for b in synth], dtype=np.int64),
            "open":  np.array([b.open for b in synth], dtype=np.float64),
            "high":  np.array([b.high for b in synth], dtype=np.float64),
            "low":   np.array([b.low for b in synth], dtype=np.float64),
            "close": np.array([b.close for b in synth], dtype=np.float64),
            "volume": np.array([b.volume for b in synth], dtype=np.float64),
        }
        bars = arrays
    else:
        try:
            bars = load_bars(args.code, args.start, args.end)
        except Exception as e:
            # 友好提示: 默认 DB 不可达时告诉用户默认主机 + 逃生口 (设计 D3)
            from sqlalchemy.exc import OperationalError as _SAOperationalError
            if isinstance(e, _SAOperationalError):
                # 用 DEFAULT_DB_URL 而非 DB_URL: 前者不受 EVTRADE_DB_URL 覆写,
                # 始终是项目默认值 (spec Scenario "DB 不可达时 CLI 输出默认主机提示")
                from .core.data import DEFAULT_DB_URL, DEFAULT_TABLE
                print(
                    f"\n[数据源不可达] 当前默认 DB_URL = {DEFAULT_DB_URL}\n"
                    f"               默认表名 = {DEFAULT_TABLE}\n"
                    f"  → 网络/VPN/库未启; 可 export EVTRADE_DB_URL / EVTRADE_TABLE 指向其他库,\n"
                    f"  → 或加 --synthetic-days N 用合成数据跳过 DB。",
                    flush=True,
                )
                raise SystemExit(2)
            raise
    n = len(bars["stime"])
    print(f"证券: {args.code}  周期: {args.period}  策略日期: {args.start}~{args.end}  "
          f"策略: {args.strategy}  device={args.device}\n", flush=True)

    t0 = time.perf_counter()
    result = run_vectorized(
        bars, period=args.period, warmup_until=int(args.start) * 1_000_000,
        strategy=strategy, params=strategy.params,
        verbose=args.verbose)
    dt = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(f"回测完成 (vectorized; 仅驱动 step; framework 不汇总 PnL/收益)")
    print("=" * 60)
    print(f"信号轨迹:    {len(result['sig'])} 桶 (mark=1 段)")
    print(f"桶数:        {len(result['buckets']['ts'])} (含预热)")
    print(f"引擎耗时:    {dt * 1000:.1f} ms ({n} 根 1m bar, {args.device})")
    print("-" * 60)
    print("策略 final_state (framework 仅透传):")
    final = result["final_state"]
    if final is None:
        print("  (None)")
    elif hasattr(final, "__dataclass_fields__"):
        for k, v in final.__dataclass_fields__.items():
            val = getattr(final, k)
            if isinstance(val, (int, float, str, bool)):
                print(f"  {k} = {val!r}")
            else:
                print(f"  {k} = <{type(val).__name__}>")
    elif isinstance(final, dict):
        for k, v in final.items():
            print(f"  {k} = {v!r}")
    else:
        print(f"  {final!r}")
    print("=" * 60)

    if args.signals_out:
        # sig 是桶级 (mark=1 段); ts 与 sig 同源同 mask, 逐对写出
        b = result["buckets"]
        live = b["mark"] == 1
        _write_signals_csv(
            args.signals_out,
            [(int(t), int(s)) for t, s in zip(b["ts"][live], result["sig"])])


def backtest_main(argv=None):
    """backtest 主入口"""
    ap = argparse.ArgumentParser(
        prog="evtrade backtest",
        description="策略回测 (vectorized; CPU/GPU 统一走 PyTorch, --device 路由)",
        parents=[_common_parent(), _data_parent()])
    args = build_backtest_parser(ap).parse_args(argv)
    _run_backtest(args)


# ============ sweep 子命令 ============

def build_sweep_parser(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """sweep 选项 (注册到给定 parser; prog/description 在 sweep_main)"""
    ap.add_argument("--grid", action="append", default=[],
                    help="参数网格, 可多次: --grid key=v1,v2,v3")
    ap.add_argument("--split", default=None, help="单分割日 YYYYMMDD")
    ap.add_argument("--splits", default=None,
                    help="滚动 WFO 分割日, 逗号分隔")
    ap.add_argument("--score-lambda", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="sweep_results.csv")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--save-defaults", action="store_true", default=False)
    ap.add_argument("--no-save-defaults", dest="save_defaults", action="store_false")
    return ap


def sweep_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="evtrade sweep",
        description="参数并发扫描 (vectorized; CPU/GPU 统一)",
        parents=[_common_parent(), _data_parent()])
    args = build_sweep_parser(ap).parse_args(argv)
    from .core.data import load_bars, synthetic_bars
    from .core.sweep import GRID_KEYS, parse_grid, sweep as run_sweep
    from .strategies import get_strategy_param_spec

    if args.synthetic_days > 0:
        from datetime import datetime, timedelta
        d_end = datetime.strptime(args.end, "%Y%m%d")
        d_start = d_end - timedelta(days=args.synthetic_days)
        print(f"生成合成数据: {d_start:%Y%m%d} ~ {args.end} "
              f"({args.synthetic_days} 天, seed=42)", flush=True)
        from .primitives import Bar
        synth = synthetic_bars(days=args.synthetic_days,
                               start_ymd=f"{d_start:%Y%m%d}")
        bars = {
            "stime": np.array([int(b.stime) for b in synth], dtype=np.int64),
            "open":  np.array([b.open for b in synth], dtype=np.float64),
            "high":  np.array([b.high for b in synth], dtype=np.float64),
            "low":   np.array([b.low for b in synth], dtype=np.float64),
            "close": np.array([b.close for b in synth], dtype=np.float64),
            "volume": np.array([b.volume for b in synth], dtype=np.float64),
        }
    else:
        bars = load_bars(args.code, args.start, args.end)

    base_params = _resolve_strategy_params(args.strategy, args.params)
    base = {"start": args.start, "period": args.period,
            "params": base_params}
    spec_keys = set(get_strategy_param_spec(args.strategy))
    spec_map = get_strategy_param_spec(args.strategy)
    combos = parse_grid(args.grid, extra_keys=spec_keys,
                       type_hints={k: v.get("type")
                                   for k, v in (spec_map or {}).items()}) or [{}]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()] if args.splits else None
    if splits:
        print(f"扫描 {len(combos)} 组参数 (滚动 WFO {len(splits)} 窗: "
              f"train + test1..test{len(splits)}) ...\n", flush=True)
    else:
        print(f"扫描 {len(combos)} 组参数 (单窗) ...\n", flush=True)
    df = run_sweep(bars, base, combos, split_ymd=args.split,
                   n_workers=args.workers, device=args.device,
                   splits=splits, lam=args.score_lambda,
                   strategy_name=args.strategy)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    cols = [c for c in df.columns]
    show = [c for c in df.columns
            if c in ("score", "primary_score_min", "primary_score_mean",
                     "S", "pareto")
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


# ============ params 子命令 (默认参数落盘) ============

def build_params_parser(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """params 选项 (含 save / show / list 二级子命令;
    注册到给定 parser; prog/description 在 params_main)"""
    sub = ap.add_subparsers(dest="params_cmd", required=True)

    p_save = sub.add_parser("save", help="保存最优参数到默认目录")
    p_save.add_argument("strategy", help="策略 key")
    src = p_save.add_mutually_exclusive_group(required=True)
    src.add_argument("--params", default="")
    src.add_argument("--from-csv", default=None)
    p_save.add_argument("--rank", type=int, default=1)

    p_show = sub.add_parser("show", help="打印策略当前默认参数")
    p_show.add_argument("strategy")

    sub.add_parser("list", help="列出所有已有默认参数的策略")
    return ap


def params_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="evtrade params",
        description="策略默认参数管理 (落盘 evtrade/strategies/_defaults/<name>.json)")
    args = build_params_parser(ap).parse_args(argv)

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
        description="evtrade CLI: 策略回测 / 参数扫描 / 默认参数管理",
    )
    sub = ap.add_subparsers(dest="cmd", help="子命令")
    builders = (
        ("backtest", build_backtest_parser, (_common_parent(), _data_parent()),
         "策略回测"),
        ("sweep", build_sweep_parser, (_common_parent(), _data_parent()),
         "参数并发扫描"),
        ("params", build_params_parser, (),
         "默认参数管理"),
    )
    for name, builder, parents, help_text in builders:
        subp = sub.add_parser(name, help=f"{help_text} (见 {name} -h)",
                              parents=list(parents))
        builder(subp)
    return ap


def main(argv=None):
    import sys
    raw = sys.argv[1:] if argv is None else argv
    if not raw or raw[0] not in ("backtest", "sweep", "params",
                                 "-h", "--help"):
        if raw and raw[0].startswith("-"):
            return build_root_parser().parse_args(raw)
        if raw:
            return backtest_main(raw)
        build_root_parser().print_help()
        return None
    args = build_root_parser().parse_args(raw)
    handlers = {"backtest": backtest_main, "sweep": sweep_main,
                "params": params_main}
    return handlers[args.cmd](raw[1:])


if __name__ == "__main__":
    main()