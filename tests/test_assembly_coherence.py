"""simplify-assembly-and-coherence 行为锁定

锁定本次 change 的三处可观测行为变化 (见 spec delta):
  1. sweep 网格轴白名单: --grid all_in=... 被拒 (静默空转轴已删)
  2. reconcile 两腿同有效资金参数: all_in 时 vectorized 腿与 Engine 腿成交对称
  3. replay_vectorized 返回与 sig 对齐的 "ts" (修复 --signals-out 以 1m bar 数
     索引桶级 sig 的越界/错位 latent bug)
"""
from __future__ import annotations

from evtrade.core.data import synthetic_bars
from evtrade.core.replay import reconcile, replay_vectorized
from evtrade.core.sweep import parse_grid


def _bars(days=5):
    return synthetic_bars(days=days, start_ymd="20260105", seed=42)


def test_grid_all_in_axis_rejected():
    """all_in 曾是下游从不读取的静默空转轴; 现应报 ValueError"""
    try:
        parse_grid(["all_in=true,false"])
    except ValueError as e:
        assert "all_in" in str(e)
        return
    raise AssertionError("--grid all_in=... 应报 ValueError, 未抛")


def test_grid_funding_axes_expand():
    """资金模式网格化应经 buy_pct/sell_pct 表达 (all_in 语义 = 两者皆 1.0)"""
    combos = parse_grid(["buy_pct=0.5,1.0", "sell_pct=0.5,1.0"])
    assert len(combos) == 4
    assert {"buy_pct": 1.0, "sell_pct": 1.0} in combos  # all-in 组合可达


def test_reconcile_all_in_legs_symmetric():
    """all_in 时两腿须收到相同有效成交参数 (否则成交流不可比, 对账必 FAIL)"""
    bars = _bars()
    warm = int("20260107") * 1_000_000
    rep = reconcile(bars, "5m", warm, strategy_name="channel_deviation",
                    strategy_params={}, buy_pct=1.0, sell_pct=1.0,
                    all_in=True, verbose=False)
    assert rep["trades_ok"] is True, "all-in 下两腿成交应逐笔一致"


def test_reconcile_regression_non_all_in():
    """非 all-in 路径回归: 对账仍 PASS (逐笔一致)"""
    bars = _bars()
    warm = int("20260107") * 1_000_000
    rep = reconcile(bars, "5m", warm, strategy_name="channel_deviation",
                    strategy_params={}, verbose=False)
    assert rep["trades_ok"] is True
    assert rep["pass"] is True


def test_replay_vectorized_ts_aligned_with_sig():
    """replay_vectorized 须返回与 sig 等长且对齐的桶级 ts (修复 --signals-out)"""
    bars = _bars()
    warm = int("20260107") * 1_000_000
    out = replay_vectorized(bars, "5m", warm,
                            strategy_name="channel_deviation", strategy_params={})
    assert "ts" in out
    assert len(out["ts"]) == len(out["sig"]), (
        f"ts/sig 长度不齐: {len(out['ts'])} vs {len(out['sig'])}")
    # 桶数 < 1m bar 数 (warmup 存在时)
    assert len(out["sig"]) < len(bars)
