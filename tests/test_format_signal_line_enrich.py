"""策略 format_signal_line hook 含方向/价格/K线 测试

锁定 spec R "Strategy signal line shows trigger bar context" 的 4 个 Scenario:
  - filtered_mr sig!=0 行含 BUY/SELL + price + 4 个 OHLC 字段名
  - channel_deviation sig!=0 行含 BUY/SELL + price + 4 个 OHLC + dev 向后兼容
  - 空 info 时 hook 不抛 KeyError
  - _last_info["c"] 等于触发该 sig 的 bar 的 close
"""
from __future__ import annotations

import numpy as np
import pytest

from evtrade.strategies import get_strategy_class


# ---------- filtered_mr ----------

def _make_filtered_mr():
    cls = get_strategy_class("filtered_mr")
    # 用默认 params 即可; 不需要真撮合, 仅验 hook 输出
    s = cls(params={})
    s._last_info = None
    return s


def test_filtered_mr_format_signal_line_buy_contains_all():
    s = _make_filtered_mr()
    info = {"side": "BUY", "price": 10.5,
            "o": 10.0, "h": 10.7, "l": 9.9, "c": 10.5, "v": 12345.0}
    out = s.format_signal_line(20260903150000, 1, info)
    assert "BUY" in out
    assert "price=" in out
    assert "o=" in out and "h=" in out and "l=" in out and "c=" in out
    assert "v=" in out
    # 价格值必须出现在行中 (fmt 默认输出)
    assert "10.5" in out
    assert "10.0" in out  # o
    assert "10.7" in out  # h
    assert "9.9" in out   # l


def test_filtered_mr_format_signal_line_sell_contains_all():
    s = _make_filtered_mr()
    info = {"side": "SELL", "price": 9.8,
            "o": 10.2, "h": 10.3, "l": 9.7, "c": 9.8, "v": 5000.0}
    out = s.format_signal_line(20260903150000, -1, info)
    assert "SELL" in out
    assert "price=" in out
    assert "o=" in out and "h=" in out and "l=" in out and "c=" in out


def test_filtered_mr_format_signal_line_empty_info_no_keyerror():
    s = _make_filtered_mr()
    # info={} MUST NOT 抛 KeyError; 返回字符串 MUST 仍含 BUY (从 sig=+1 推)
    out = s.format_signal_line(20260903150000, 1, {})
    assert "BUY" in out
    # 缺字段显示为空字符串, 字段名 `o=` `h=` 仍在 (空值渲染)
    assert "o=" in out and "h=" in out and "l=" in out and "c=" in out


def test_filtered_mr_format_signal_line_no_info_arg_no_keyerror():
    s = _make_filtered_mr()
    # info=None 也不抛
    out = s.format_signal_line(20260903150000, -1, None)
    assert "SELL" in out


# ---------- channel_deviation ----------

def _make_channel_deviation():
    cls = get_strategy_class("channel_deviation")
    s = cls(params={"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5,
                    "tf1": 21})
    s._last_info = None
    return s


def test_channel_deviation_format_signal_line_contains_price_and_ohlcv():
    s = _make_channel_deviation()
    info = {"side": "BUY", "price": 10.5,
            "o": 10.0, "h": 10.7, "l": 9.9, "c": 10.5, "v": 12345.0,
            "up": 10.8, "dw": 9.7, "low_dev": 5.0, "high_dev": 4.5,
            "cash": 100000.0, "position": 0.0}
    out = s.format_signal_line(20260903150000, 1, info)
    assert "BUY" in out
    assert "price=" in out
    assert "o=" in out and "h=" in out and "l=" in out and "c=" in out
    assert "v=" in out
    # 向后兼容: 原通道字段必须仍出现 (spec Scenario `channel_deviation 信号行扩展后含触发 K 线`)
    assert "UP=" in out or "up=" in out
    assert "DW=" in out or "dw=" in out
    assert "low_dev" in out or "high_dev" in out


def test_channel_deviation_format_signal_line_empty_info_no_keyerror():
    s = _make_channel_deviation()
    out = s.format_signal_line(20260903150000, -1, {})
    assert "SELL" in out
    # 不抛 + 字段名都在
    assert "price=" in out
    assert "o=" in out


# ---------- _last_info 字段与触发 bar 一致 (spec Scenario) ----------

def test_filtered_mr_last_info_c_equals_trigger_bar_close():
    """跑真实 step, 断言 _last_info 字段 = step 接收的最后一个 bar 字段
    (spec Scenario `_last_info 字段集与 step 触发 bar 一致`)
    """
    from evtrade.strategies.filtered_mr import FilteredMRState
    cls = get_strategy_class("filtered_mr")
    s = cls(params={})
    state = s.init_state({})

    bar_steady = {"ts": 20260903140000, "o": 10.0, "h": 10.1, "l": 9.9,
                  "c": 10.0, "v": 1000.0, "mark": 1}
    # 跑 30 根稳态让所有 EMA/ATR/ADX 预热
    for i in range(30):
        b = {**bar_steady, "ts": 20260903140000 + i * 60_000}
        state, _ = s.step(state, b, s.params)

    # 最后喂一根明确 bar, 断言 _last_info 字段 = 该 bar 字段
    # (不要求触发 sig != 0: framework 只在 sig != 0 时调 hook,
    #  但 step 始终写 _last_info, 字段集应始终等于该 bar 的 OHLCV)
    last_bar = {"ts": 20260903150000, "o": 10.2, "h": 10.4,
                "l": 10.1, "c": 10.3, "v": 2500.0, "mark": 1}
    state, sig = s.step(state, last_bar, s.params)
    assert s._last_info is not None
    assert s._last_info["c"] == last_bar["c"]
    assert s._last_info["o"] == last_bar["o"]
    assert s._last_info["h"] == last_bar["h"]
    assert s._last_info["l"] == last_bar["l"]
    assert s._last_info["v"] == last_bar["v"]
    # side 字段存在, 取值合法
    assert s._last_info["side"] in ("BUY", "SELL", "")
    # price 与 c 一致 (filtered_mr 撮合用 close)
    assert s._last_info["price"] == last_bar["c"]


# ---------- hook 内不允许 info[k] 下标访问 (spec Scenario `hook 用 info.get 不抛 KeyError`) ----------

@pytest.mark.parametrize("strategy_name", ["filtered_mr", "channel_deviation"])
def test_hook_no_info_subscript_access(strategy_name):
    """源扫描: hook 函数体内不能出现 `info[` 下标访问, 必须走 info.get
    (spec Scenario `hook 用 info.get 不抛 KeyError` 防御性约束)"""
    import re
    from pathlib import Path
    src = Path(f"evtrade/strategies/{strategy_name}.py").read_text(encoding="utf-8")

    # 定位 format_signal_line 函数体 (从 def 行到下一个 def / 顶层语句)
    m = re.search(r"def format_signal_line\(self.*?\n(.*?)(?=\n    def |\nclass |\Z)",
                  src, re.DOTALL)
    assert m, f"{strategy_name} 找不到 format_signal_line"
    body = m.group(1)

    # body 内不能出现 info[ (下标访问)
    assert "info[" not in body, (
        f"{strategy_name}.format_signal_line 内有 `info[...]` 下标访问, "
        f"违反 spec 'hook 用 info.get 不抛 KeyError':\n{body[:400]}")
    # 反向断言: 至少有一个 info.get
    assert "info.get" in body, (
        f"{strategy_name}.format_signal_line 应至少用一次 info.get(...)")
