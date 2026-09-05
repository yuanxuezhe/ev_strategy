from __future__ import annotations
"""基准测试: 参考引擎 vs numba 内核 单次回测 + 并发参数扫描

用法: python scripts/benchmark.py [天数]   (默认 700 天 ≈ 16.8万根 1m bar)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evtrade import (Account, BarAggregator, ChannelDeviationStrategy, Engine,
                     Feed, SimulatedExecutor)
from evtrade.data import synthetic_bars
from evtrade.kernel import bars_to_arrays, make_state, run_backtest, summarize
from evtrade.sweep import parse_grid, sweep


class ListFeed(Feed):
    def __init__(self, bars):
        self.bars = bars

    def stream(self):
        yield from self.bars


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 700
    bars = synthetic_bars(days=days, start_ymd="20250101", seed=42)
    arr = bars_to_arrays(bars)
    n = len(bars)
    params = dict(tf1=21, low1=0.4, low2=0.25, high1=0.4, high2=0.2)
    warmup = 20250301000000
    print(f"数据: {n} 根 1m bar ({days} 天)  参数: {params}  "
          f"CPU 核数: {os.cpu_count()}\n")

    # ---- 参考引擎 (纯 Python, 与原单文件相同路径) ----
    t0 = time.perf_counter()

    class RecExec(SimulatedExecutor):
        records = None

        def trade(self, signal, price, ts):
            ok = super().trade(signal, price, ts)
            if ok:
                self.records = self.account.trades
            return ok

    account = Account(cash=200000.0, position=200000.0)
    executor = RecExec(account, qty=10000.0, verbose=False)
    aggregator = BarAggregator(PERIODS_CFG["5m"], on_bars=None, warmup_until="20250301000000")
    strategy = ChannelDeviationStrategy(low1=params["low1"], low2=params["low2"],
                                        high1=params["high1"], high2=params["high2"])
    engine = Engine(ListFeed(bars), aggregator, strategy, executor, tf1=21, verbose=False)
    engine.run()
    t_ref = time.perf_counter() - t0
    ref = dict(n=len(account.trades), cash=account.cash, pos=account.position,
               eq=account.equity())
    print(f"参考引擎   : {t_ref*1000:10.1f} ms  ({n/t_ref/1e6:.2f} M bar/s)  "
          f"trades={ref['n']} equity={ref['eq']:.2f}")

    # ---- 内核: 首次 (含 JIT 编译) ----
    st = make_state(period="5m", warmup_until=warmup, record_trades=True, trade_cap=n, **params)
    t0 = time.perf_counter()
    run_backtest(st, arr["stime"], arr["open"], arr["high"], arr["low"],
                 arr["close"], arr["volume"],
                 np.empty(0, np.int8), np.empty(0), np.empty(0))
    t_cold = time.perf_counter() - t0
    print(f"内核(冷)   : {t_cold*1000:10.1f} ms  (含 numba JIT 编译)")

    # ---- 内核: 热运行 (取 10 次平均) ----
    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        st = make_state(period="5m", warmup_until=warmup, record_trades=False, **params)
        run_backtest(st, arr["stime"], arr["open"], arr["high"], arr["low"],
                     arr["close"], arr["volume"],
                     np.empty(0, np.int8), np.empty(0), np.empty(0))
    t_warm = (time.perf_counter() - t0) / reps
    s = summarize(st)
    print(f"内核(热)   : {t_warm*1000:10.1f} ms  ({n/t_warm/1e6:.2f} M bar/s)  "
          f"提速 {t_ref/t_warm:,.0f}x  trades={s['n_trades']} equity={s['final_equity']:.2f}")

    # ---- 与参考引擎结果核对 ----
    assert s["n_trades"] == ref["n"], (s["n_trades"], ref["n"])
    assert s["final_cash"] == ref["cash"] and s["final_position"] == ref["pos"]
    print("核对       : 参考引擎与内核成交数/资金/持仓一致 ✓\n")

    # ---- 并发扫描: 1000 组参数 (CPU) ----
    base = {"start": "20250301", "period": "5m", "tf1": 21,
            "low1": 0.4, "low2": 0.25, "high1": 0.4, "high2": 0.2,
            "trade_qty": 10000.0, "init_cash": 200000.0, "init_position": 200000.0}
    grid = parse_grid([
        "low1=0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75",
        "low2=0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60",
        "high2=0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55",
    ])
    workers = min(8, os.cpu_count() or 4)
    t0 = time.perf_counter()
    df = sweep(arr, base, grid, n_workers=workers)
    t_cpu_sweep = time.perf_counter() - t0
    print(f"\nCPU 扫描耗时: {t_cpu_sweep:.2f}s ({workers} 线程)")
    print("最优 5 组 (按超额收益, CPU):")
    cols = ["low1", "low2", "high2", "n_trades", "excess_pct", "max_drawdown"]
    print(df[cols].head(5).to_string(index=False))

    # ---- GPU 对比 (需 cupy; 与 CPU 结果按网格顺序逐组核对) ----
    try:
        from evtrade.gpu import cuda_sweep_window
        from evtrade.sweep import _run_window
    except Exception as e:
        print(f"\n[GPU] cupy 不可用, 跳过 ({e})")
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        cpu_res = list(ex.map(lambda c: _run_window(arr, dict(base, **c), 20250301000000),
                              grid))
    t0 = time.perf_counter()
    gpu_res = cuda_sweep_window(arr, [dict(base, **c) for c in grid], 20250301000000)
    t_gpu = time.perf_counter() - t0
    mism = sum(1 for g, c in zip(gpu_res, cpu_res)
               if g["n_trades"] != c["n_trades"] or g["excess"] != c["excess"])
    print(f"\nGPU 1000 组 x {n} 根: {t_gpu:.2f}s  "
          f"(与 CPU 逐位核对不一致组数: {mism}, GPU/CPU 耗时比 {t_gpu / t_cpu_sweep:.2f}x)")

    # ---- GPU 大网格扩展性: 1 万组 ----
    big = parse_grid([
        "low1=0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75",
        "low2=0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60",
        "high1=0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00,1.20",
        "high2=0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55",
    ])
    t0 = time.perf_counter()
    cuda_sweep_window(arr, [dict(base, **c) for c in big], 20250301000000)
    t_big = time.perf_counter() - t0
    total = len(big) * n
    print(f"GPU {len(big)} 组 x {n} 根 = {total / 1e9:.2f} G bar-steps: {t_big:.1f}s "
          f"({total / t_big / 1e9:.2f} G bar-steps/s)")


from evtrade.config import PERIODS as PERIODS_CFG

if __name__ == "__main__":
    main()
