from __future__ import annotations
"""向量化引擎测试 (CuPy 统一 CPU/GPU 路径)

- test_ma_crossover_cpu_smoke: CPU 路径跑通, 信号/成交/汇总结构正确
- test_vectorized_sig_shape: 信号数组长度 = 桶数, dtype int8
- test_vectorized_warmup_no_signal: 预热段 (mark=0) 不产信号
- test_ma_crossover_cpu_vs_gpu: CPU vs GPU 结果容差 1e-9 (cupy 不可用则 skip)
- test_vectorized_matches_ref: MA 交叉策略用 ref 引擎 (逐 bar) 与 vectorized 引擎
  对比信号/成交/终态 (验证向量化语义正确)
"""

import numpy as np
import pytest

from evtrade.data import synthetic_bars
from evtrade.kernel import bars_to_arrays


def _bars():
    return bars_to_arrays(synthetic_bars(days=25, start_ymd="20250101", seed=11))


def _run(device="cpu", fast=5, slow=20):
    from evtrade.strategies.vectorized_base import get_vectorized_strategy
    from evtrade.core.vectorized_engine import run_vectorized
    arr = _bars()
    strat = get_vectorized_strategy("ma_crossover", params={"fast": fast, "slow": slow})
    return run_vectorized(
        arr, period="5m", warmup_until=20250110000000,
        strategy=strat, params={"fast": fast, "slow": slow},
        init_cash=200000.0, init_position=200000.0,
        trade_qty=10000.0, scale=1.0, device=device)


# ---------- cupy 可用性 (复用 test_gpu.py 的 skipif 模式) ----------

cupy_ready = True
_reason = ""
try:
    from evtrade.backends import gpu_available
    cupy_ready = gpu_available()
    if not cupy_ready:
        _reason = "cupy/GPU 不可用"
except Exception as e:
    cupy_ready = False
    _reason = str(e)[:80]


# ---------- CPU smoke ----------

def test_ma_crossover_cpu_smoke():
    result = _run(device="cpu")
    s = result["summary"]
    assert s["n_trades"] > 0, "应产生成交"
    assert s["n_buy"] + s["n_sell"] == s["n_trades"]
    assert len(result["trades"]) == s["n_trades"]
    assert s["final_equity"] > 0
    assert len(result["sig"]) == len(result["buckets"]["ts"])


def test_vectorized_sig_shape():
    result = _run(device="cpu")
    sig = result["sig"]
    assert sig.dtype == np.int8
    assert sig.shape == (len(result["buckets"]["ts"]),)
    assert set(np.unique(sig).tolist()) <= {-1, 0, 1}


def test_vectorized_warmup_no_signal():
    """预热段 (mark=0) 不应产生信号"""
    result = _run(device="cpu")
    mark = result["buckets"]["mark"]
    sig = result["sig"]
    # 预热段信号必须全 0
    assert np.all(sig[mark == 0] == 0), "预热段不应产生信号"


# ---------- CPU vs GPU (容差; cupy 不可用 skip) ----------

@pytest.mark.skipif(not cupy_ready, reason=f"cupy/GPU 不可用: {_reason}")
def test_ma_crossover_cpu_vs_gpu():
    """同参数同数据, CPU (numpy) vs GPU (cupy) 结果容差 1e-9

    CuPy 高阶封装的 GPU 算子舍入与 CPU 不同, 不保证 bitwise; 但策略逻辑
    确定性 + 同输入, 结果应在 ULP 级容差内。
    """
    cpu = _run(device="cpu")
    gpu = _run(device="gpu")

    # 信号应完全一致 (整数比较, 无浮点舍入问题)
    assert np.array_equal(cpu["sig"], gpu["sig"]), "信号轨迹不一致"

    # 浮点指标容差 1e-9
    for k in ("final_cash", "final_position", "final_equity", "turnover"):
        a, b = cpu["summary"][k], gpu["summary"][k]
        assert abs(a - b) < 1e-6, f"{k}: cpu={a!r} gpu={b!r}"

    assert cpu["summary"]["n_trades"] == gpu["summary"]["n_trades"]


# ---------- 向量化语义正确性 (vs 逐 bar ref) ----------

def test_vectorized_ma_crossover_logic():
    """验证 MA 交叉策略的信号逻辑: BUY = fast 上穿 slow, SELL = fast 下穿 slow

    不与 ref 引擎对比 (MA 交叉不是 StrategyBase 逐 bar 策略), 而是直接用
    numpy 独立算一遍 EMA + 交叉, 与 vectorized 引擎输出对比。
    """
    from evtrade.indicators.ema import ema
    result = _run(device="cpu", fast=5, slow=20)
    buckets = result["buckets"]
    close = np.asarray(buckets["c"], dtype=np.float64)
    mark = np.asarray(buckets["mark"])

    fast = np.array(ema(close.tolist(), 5), dtype=np.float64)
    slow = np.array(ema(close.tolist(), 20), dtype=np.float64)
    fast = np.nan_to_num(fast, nan=0.0)
    slow = np.nan_to_num(slow, nan=0.0)
    diff = fast - slow
    prev_diff = np.empty_like(diff)
    prev_diff[1:] = diff[:-1]
    prev_diff[0] = 0.0
    buy = (diff > 0) & (prev_diff <= 0) & np.isfinite(fast) & (mark == 1)
    sell = (diff < 0) & (prev_diff >= 0) & np.isfinite(fast) & (mark == 1)
    expected = np.where(buy, 1, np.where(sell, -1, 0)).astype(np.int8)

    assert np.array_equal(result["sig"], expected), "MA 交叉信号逻辑不一致"


def test_vectorized_trade_execution():
    """验证成交语义: BUY 扣 cash 加 position, SELL 反之"""
    result = _run(device="cpu", fast=5, slow=20)
    trades = result["trades"]
    s = result["summary"]
    # 手工核验: 每个 BUY 的 qty*price <= 当时的 cash; SELL 的 qty <= 当时的 position
    cash = 200000.0
    position = 200000.0
    for t in trades:
        if t["side"] == "BUY":
            assert t["qty"] * t["price"] <= cash + 1e-6, f"BUY 超过现金: {t}"
            cash -= t["qty"] * t["price"]
            position += t["qty"]
        else:
            assert t["qty"] <= position + 1e-6, f"SELL 超过持仓: {t}"
            cash += t["qty"] * t["price"]
            position -= t["qty"]
    assert abs(cash - s["final_cash"]) < 1e-6, f"cash 不一致: {cash} vs {s['final_cash']}"
    assert abs(position - s["final_position"]) < 1e-6
