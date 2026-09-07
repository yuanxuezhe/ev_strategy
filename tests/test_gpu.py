from __future__ import annotations
"""GPU vs CPU 内核差分测试 + GPU 吞吐冒烟

CUDA 内核是 kernel.step 的逐行移植 (float64 + --fmad=false),
期望与 CPU 内核逐位一致: 成交笔数 / cash / position / 期末价 / 最大回撤。
"""

import time

import numpy as np
import pytest

cupy_ready = True
_reason = ""
try:
    from evtrade.gpu import _ensure_cupy  # noqa: F401
    _ensure_cupy()
except Exception as e:  # noqa: BLE001
    cupy_ready = False
    _reason = str(e)[:80]

pytestmark = pytest.mark.skipif(not cupy_ready, reason=f"cupy/GPU 不可用: {_reason}")

from evtrade.data import synthetic_bars  # noqa: E402
from evtrade.gpu import cuda_sweep_window_generic  # noqa: E402
from evtrade.kernel import bars_to_arrays  # noqa: E402
from evtrade.sweep import run_one_from_dict  # noqa: E402

WARMUP = 20250110000000


def _params_list():
    """确定性取 ~64 组参数: 覆盖阈值/周期/tf1/倍投组合 (含自定义周期 90m/7m)"""
    out = []
    i = 0
    scales = (1.0, 2.0, 1.5)
    for low1 in (0.3, 0.5, 0.8, 1.2, 1.8, 2.5):
        for high2 in (0.15, 0.3, 0.6, 1.0):
            for tf1, period in ((10, "5m"), (21, "5m"), (60, "15m"),
                                (21, "90m"), (13, "7m")):
                low2 = low1 * 0.6
                high1 = low1
                out.append({"period": period, "tf1": tf1,
                            "scale": scales[i % 3],
                            "init_cash": 200000.0, "init_position": 200000.0,
                            "trade_qty": 10000.0,
                            "params": {"low1": low1, "low2": round(low2, 4),
                                       "high1": high1, "high2": high2}})
                i += 1
                if i >= 64:
                    return out
    return out


def test_gpu_matches_cpu_bitwise():
    bars = bars_to_arrays(synthetic_bars(days=25, start_ymd="20250101", seed=11))
    params_list = _params_list()
    assert len(params_list) == 64

    gpu_res = cuda_sweep_window_generic(bars, params_list, WARMUP,
                                        strategy_name="channel_deviation")
    cpu_res = [run_one_from_dict(bars, p, WARMUP, strategy_name="channel_deviation")
               for p in params_list]

    n_sig = sum(1 for g in gpu_res if g["n_trades"] > 0)
    for i, (g, c) in enumerate(zip(gpu_res, cpu_res)):
        assert g["n_trades"] == c["n_trades"], f"#{i} 笔数: gpu={g['n_trades']} cpu={c['n_trades']}"
        assert g["final_price"] == c["final_price"], f"#{i} 期末价"
        assert g["final_cash"] == c["final_cash"], f"#{i} cash: gpu={g['final_cash']!r} cpu={c['final_cash']!r}"
        assert g["final_position"] == c["final_position"], f"#{i} position"
        assert g["max_drawdown"] == c["max_drawdown"], f"#{i} mdd"
        assert g["excess"] == c["excess"], f"#{i} excess"
        # 超额曲线新增指标
        assert g["years"] == c["years"], f"#{i} years"
        assert g["ann_excess_pct"] == c["ann_excess_pct"], f"#{i} 年化超额"
        assert g["sharpe_excess"] == c["sharpe_excess"], f"#{i} 超额Sharpe"
        assert g["x_mdd"] == c["x_mdd"], f"#{i} 超额回撤"
    assert n_sig > 10, "有效信号样本过少, 测试无说服力"


def test_gpu_walkforward_window_isolation():
    """warmup=split: GPU 测试窗成交全部落在 split 之后"""
    bars = bars_to_arrays(synthetic_bars(days=30, start_ymd="20250101", seed=23))
    split = 20250120000000
    p = [{"period": "5m", "tf1": 21,
          "params": {"low1": 0.4, "low2": 0.25, "high1": 0.4, "high2": 0.2},
          "init_cash": 200000.0, "init_position": 200000.0, "trade_qty": 10000.0}]
    res = cuda_sweep_window_generic(bars, p, split,
                                    strategy_name="channel_deviation")[0]
    assert res["final_price"] > 0


def test_gpu_throughput_smoke():
    """1 万组参数吞吐冒烟 (断言宽松, 只验证可扩展性, 不作性能断言)"""
    bars = bars_to_arrays(synthetic_bars(days=30, start_ymd="20250101", seed=42))
    n = len(bars["stime"])
    combos = []
    k = 0
    for low1 in (0.3 + 0.01 * j for j in range(100)):
        for high2 in (0.1 + 0.01 * j for j in range(100)):
            combos.append({"period": "5m", "tf1": 21,
                           "params": {"low1": low1, "low2": low1 * 0.6,
                                      "high1": low1, "high2": high2},
                           "init_cash": 200000.0, "init_position": 200000.0,
                           "trade_qty": 10000.0})
            k += 1
            if k >= 10000:
                break
        if k >= 10000:
            break
    t0 = time.perf_counter()
    res = cuda_sweep_window_generic(bars, combos, 20250110000000,
                                    strategy_name="channel_deviation")
    dt = time.perf_counter() - t0
    assert len(res) == 10000
    total_bar_steps = 10000 * n
    print(f"\nGPU 吞吐: 10000 组 x {n} 根 = {total_bar_steps / 1e6:.0f}M bar-steps "
          f"in {dt:.2f}s ({total_bar_steps / dt / 1e9:.2f} G bar-steps/s)")
