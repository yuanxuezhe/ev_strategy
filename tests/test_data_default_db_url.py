"""`evtrade/core/data.py` 默认 DB_URL / TABLE 形态 + env 优先级

锁定 spec R "Market data DB connection has a sane default" 的 4 个 Scenario:
  - 默认值形态 (host / port / user / pass / db / URL-encode)
  - EVTRADE_DB_URL env 优先
  - EVTRADE_TABLE env 优先
  - `@` 已编码为 `%40`

静态断言: 不连库 (CI 无 MySQL)。
"""
from __future__ import annotations

import importlib
import os

import pytest


# ---------- 1. 默认值形态 ----------

def test_default_db_url_scheme_and_user():
    """默认 URL scheme + user = mysql+pymysql + EvTrade"""
    from evtrade.core import data
    assert data.DB_URL.startswith("mysql+pymysql://EvTrade:"), (
        f"DB_URL 起始不对: {data.DB_URL!r}")


def test_default_db_url_host_and_port():
    """默认主机 = 192.168.10.2:33066"""
    from evtrade.core import data
    assert "192.168.10.2:33066" in data.DB_URL, (
        f"DB_URL 缺默认主机 192.168.10.2:33066: {data.DB_URL!r}")


def test_default_db_url_password_url_encoded():
    """密码中 `@` MUST URL-encode 为 `%40`"""
    from evtrade.core import data
    assert "p%40ssw0rd" in data.DB_URL, (
        f"DB_URL 缺 URL-encoded 密码 p%40ssw0rd: {data.DB_URL!r}")
    # 反向断言: 绝不能含裸 `@` (避免 pymysql 把它当 host 分割符)
    assert "p@ssw0rd" not in data.DB_URL, (
        f"DB_URL 含未编码 @, 会破坏 host 解析: {data.DB_URL!r}")


def test_default_db_url_database():
    """默认库 = evtrade, 且 charset=utf8mb4"""
    from evtrade.core import data
    assert "/evtrade?" in data.DB_URL, (
        f"DB_URL 缺默认库 evtrade: {data.DB_URL!r}")
    assert "charset=utf8mb4" in data.DB_URL


def test_default_table_name():
    """TABLE 默认值 = minute_bars"""
    from evtrade.core import data
    assert data.TABLE == "minute_bars"


# ---------- 2. env 优先级 ----------

def test_evtrade_db_url_env_overrides_default(monkeypatch):
    """EVTRADE_DB_URL 设置时 DB_URL 走 env"""
    monkeypatch.setenv(
        "EVTRADE_DB_URL",
        "mysql+pymysql://other:other@10.0.0.1:3306/other?charset=utf8mb4")
    # data 模块在 import 时已把 os.environ.get 求值到 DB_URL; 这里必须 reload
    import evtrade.core.data as data_mod
    importlib.reload(data_mod)
    assert data_mod.DB_URL == (
        "mysql+pymysql://other:other@10.0.0.1:3306/other?charset=utf8mb4")
    # 关键: env 值生效时, 默认主机的标记 MUST 不在
    assert "192.168.10.2" not in data_mod.DB_URL


def test_evtrade_table_env_overrides_default(monkeypatch):
    """EVTRADE_TABLE 设置时 TABLE 走 env"""
    monkeypatch.setenv("EVTRADE_TABLE", "minute_bars_v2")
    import evtrade.core.data as data_mod
    importlib.reload(data_mod)
    assert data_mod.TABLE == "minute_bars_v2"


def test_env_unset_keeps_default(monkeypatch):
    """清掉 env 后 reload, 回到默认值"""
    monkeypatch.delenv("EVTRADE_DB_URL", raising=False)
    monkeypatch.delenv("EVTRADE_TABLE", raising=False)
    import evtrade.core.data as data_mod
    importlib.reload(data_mod)
    assert "192.168.10.2:33066" in data_mod.DB_URL
    assert data_mod.TABLE == "minute_bars"


# ---------- 3. DEFAULT_* 常量不受 env 覆写 (CLI 文案用) ----------

def test_default_db_url_constant_env_independent(monkeypatch):
    """DEFAULT_DB_URL 是 env-independent 常量 (CLI 中文提示读这个)"""
    monkeypatch.setenv(
        "EVTRADE_DB_URL",
        "mysql+pymysql://other:other@10.0.0.1:3306/other?charset=utf8mb4")
    # reload 后 DB_URL 走 env, 但 DEFAULT_DB_URL 仍是项目默认值
    import evtrade.core.data as data_mod
    importlib.reload(data_mod)
    assert data_mod.DEFAULT_DB_URL == (
        "mysql+pymysql://EvTrade:p%40ssw0rd@192.168.10.2:33066/evtrade?charset=utf8mb4")
    # 反向: 此时 DB_URL 已被 env 改写, 与 DEFAULT_DB_URL 不等
    assert data_mod.DB_URL != data_mod.DEFAULT_DB_URL


def test_default_table_constant_env_independent(monkeypatch):
    """DEFAULT_TABLE 是 env-independent 常量"""
    monkeypatch.setenv("EVTRADE_TABLE", "minute_bars_v2")
    import evtrade.core.data as data_mod
    importlib.reload(data_mod)
    assert data_mod.DEFAULT_TABLE == "minute_bars"
    assert data_mod.TABLE != data_mod.DEFAULT_TABLE
