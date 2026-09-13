"""evtrade.strategies._defaults_loader 行为

覆盖:
  - 基础存在性 / save / load / list_defaulted / env 变量切换
  - numpy / pandas 标量兼容 (save 入口 + save_best_from_sweep)
  - _auto_cast / params_from_csv_row 类型转换
  - pick_best_row: filter_pass 优先, score 排序, runner_up_gap
  - save_best_from_sweep: 落盘文件 + source 字段 + numpy 标量兼容
  - format_reason_log 输出含 score / ann_net_min / 选中参数
  - _commit_defaults_file 在临时 git 仓库里能 commit
"""
from __future__ import annotations

import json
import subprocess

import pandas as pd
import pytest

import evtrade.strategies._defaults_loader as dl
from evtrade.cli import sweep_main
from evtrade.strategies._defaults_loader import (
    format_reason_log,
    pick_best_row,
    save_best_from_sweep,
)


# ============ shared fixtures ============

@pytest.fixture
def fake_sweep_df():
    """构造一个模拟 sweep 结果 DataFrame"""
    return pd.DataFrame([
        # rank 1: filter_pass=True, score 最高
        {"low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5,
         "score": 0.10, "ann_net_min": 0.08, "S": 0.5, "filter_pass": True,
         "train_ann_net_min": 0.09, "test1_ann_net_min": 0.08},
        # rank 2: filter_pass=False, score 更高但被过滤
        {"low1": 2.5, "low2": 0.5, "high1": 2.0, "high2": 0.4,
         "score": 0.15, "ann_net_min": 0.12, "S": 0.3, "filter_pass": False,
         "train_ann_net_min": 0.13, "test1_ann_net_min": 0.12},
        # rank 3: filter_pass=True 但 score 低
        {"low1": 1.0, "low2": 1.5, "high1": 1.0, "high2": 0.8,
         "score": 0.06, "ann_net_min": 0.05, "S": 0.4, "filter_pass": True,
         "train_ann_net_min": 0.07, "test1_ann_net_min": 0.05},
    ])


# ============ 基础存在性 / 路径 ============

def test_path_for_returns_under_defaults_dir():
    p = dl.path_for("channel_deviation")
    assert p.name == "channel_deviation.json"
    assert p.parent.name == dl.DEFAULT_DIR_NAME


def test_exists_false_when_no_file():
    assert dl.exists("definitely_not_a_strategy_xyz") is False


def test_list_defaulted_empty_when_dir_missing(tmp_path, monkeypatch):
    """目录不存在时 list_defaulted 返回 []"""
    nonexistent = tmp_path / "no_such_dir"
    monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", str(nonexistent))
    assert dl.list_defaulted() == []


# ============ save / load ============

def test_save_creates_file_and_load_returns_it(isolated_defaults):
    p = dl.save("channel_deviation", {"low1": 1.5, "low2": 1.0,
                                       "high1": 1.5, "high2": 0.5},
                source={"kind": "from_params"})
    assert p.is_file()
    assert p.parent == isolated_defaults

    data = dl.load("channel_deviation")
    assert data["strategy"] == "channel_deviation"
    assert data["params"] == {"low1": 1.5, "low2": 1.0,
                              "high1": 1.5, "high2": 0.5}
    assert data["source"] == {"kind": "from_params"}
    assert "saved_at" in data


def test_save_then_exists(isolated_defaults):
    assert dl.exists("channel_deviation") is False
    dl.save("channel_deviation", {"low1": 1.5})
    assert dl.exists("channel_deviation") is True


def test_save_rejects_non_dict_params(isolated_defaults):
    with pytest.raises(TypeError, match="dict"):
        dl.save("channel_deviation", "not a dict")


def test_load_missing_raises_with_helpful_message(isolated_defaults):
    with pytest.raises(FileNotFoundError, match="params save"):
        dl.load("does_not_exist")


def test_save_overwrites_existing(isolated_defaults):
    dl.save("channel_deviation", {"low1": 1.5})
    dl.save("channel_deviation", {"low1": 2.5})
    assert dl.load("channel_deviation")["params"]["low1"] == 2.5


def test_save_is_atomic_no_tmp_left(isolated_defaults):
    """save 后不应残留 *.tmp"""
    dl.save("channel_deviation", {"low1": 1.5})
    leftovers = list(isolated_defaults.glob("*.tmp"))
    assert leftovers == [], f"残留临时文件: {leftovers}"


# ============ numpy / pandas 标量兼容 ============

def test_save_accepts_numpy_int_params(isolated_defaults):
    """filtered_mr 的 int 参数 (cur_ema_period / adx_period 等) 经 sweep 选出后
    chosen[k] 是 numpy.int64; save_best_from_sweep 必须能落盘否则 json.dump 抛
    TypeError: Object of type int64 is not JSON serializable。

    直接对 save() 喂 numpy.int64 验证落盘路径通畅; save_best_from_sweep 已在
    apply 阶段用 .item() 兼容 (见 _defaults_loader.save_best_from_sweep)。
    """
    import numpy as np
    params = {
        "band_mult": 1.8,
        "cur_ema_period": np.int64(25),
        "adx_threshold": np.float64(30.0),
        "higher_period": "1h",
    }
    dl.save("filtered_mr", params)
    loaded = dl.load("filtered_mr")
    assert loaded["params"]["cur_ema_period"] == 25
    assert loaded["params"]["adx_threshold"] == 30.0
    assert isinstance(loaded["params"]["cur_ema_period"], int)


# ============ list_defaulted ============

def test_list_defaulted_returns_saved_names(isolated_defaults):
    dl.save("channel_deviation", {"low1": 1.5})
    dl.save("dev_trigger", {"entry_dev": 0.5})
    names = dl.list_defaulted()
    assert names == ["channel_deviation", "dev_trigger"]


def test_list_defaulted_ignores_dotfiles_and_non_json(isolated_defaults):
    dl.save("channel_deviation", {"low1": 1.5})
    (isolated_defaults / "README.md").write_text("hi")
    (isolated_defaults / ".gitkeep").write_text("")
    assert dl.list_defaulted() == ["channel_deviation"]


# ============ params_from_csv_row + _auto_cast ============

def test_params_from_csv_row_casts_types():
    row = {"low1": "1.5", "low2": "1.0", "high1": "2", "high2": "true"}
    out = dl.params_from_csv_row(row, ["low1", "low2", "high1", "high2", "missing"])
    assert out == {"low1": 1.5, "low2": 1.0, "high1": 2, "high2": True}


def test_params_from_csv_row_skips_missing_keys():
    row = {"low1": "1.5"}
    out = dl.params_from_csv_row(row, ["low1", "low2"])
    assert out == {"low1": 1.5}


def test_params_from_csv_row_passes_through_non_string():
    """已是 Python 原生类型时 (来自 in-memory dict) 不强转"""
    row = {"low1": 1.5, "low2": 1}
    out = dl.params_from_csv_row(row, ["low1", "low2"])
    assert out == {"low1": 1.5, "low2": 1}


def test_auto_cast_string_variants():
    assert dl.auto_cast("true") is True
    assert dl.auto_cast("false") is False
    assert dl.auto_cast("42") == 42
    assert dl.auto_cast("1.5") == 1.5
    assert dl.auto_cast("hello") == "hello"


# ============ _defaults_dir 环境变量切换 ============

def test_defaults_dir_honors_env_var(tmp_path, monkeypatch):
    custom = tmp_path / "my_defaults"
    monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", str(custom))
    p = dl.save("x", {"a": 1})
    assert p.parent == custom
    assert (custom / "x.json").is_file()


# ============ pick_best_row ============

def test_pick_best_row_prefers_filter_pass(fake_sweep_df):
    chosen, reason = pick_best_row(fake_sweep_df, "channel_deviation")
    # rank 1 有 filter_pass=True 且 score 0.10, 应被选中
    assert float(chosen["score"]) == 0.10
    assert chosen["low1"] == 1.5
    assert reason["sort_key"].startswith("filter_pass")
    assert reason["filter_pass"] is True
    assert reason["rank"] == 1
    assert reason["candidate_total"] == 3


def test_pick_best_row_runner_up_gap(fake_sweep_df):
    chosen, reason = pick_best_row(fake_sweep_df, "channel_deviation")
    # 次优 rank 2 score=0.15 (虽然 filter_pass=False, 但 score 高)
    # 排名按 row 顺序 (rank 2)
    assert reason["runner_up_score"] == 0.15
    assert reason["runner_up_gap"] == pytest.approx(0.10 - 0.15, abs=1e-9)


def test_pick_best_row_runner_up_is_true_rank2():
    """次优 = 真 rank-2 (非 chosen 中 score 最高者), 而非迭代序里第一行

    chosen 在 idx1 (score 0.10 最高); 其余两行 score 0.05(idx0) / 0.08(idx2)。
    旧实现会取迭代序首行 idx0 (0.05); 正确应为 idx2 (0.08, 真 rank-2)。
    """
    df = pd.DataFrame([
        {"low1": 1.0, "score": 0.05, "ann_net_min": 0.04, "S": 0.3},
        {"low1": 2.0, "score": 0.10, "ann_net_min": 0.09, "S": 0.4},
        {"low1": 3.0, "score": 0.08, "ann_net_min": 0.07, "S": 0.35},
    ])
    chosen, reason = pick_best_row(df, "channel_deviation")
    assert float(chosen["score"]) == 0.10
    assert reason["runner_up_score"] == 0.08
    assert reason["runner_up_gap"] == pytest.approx(0.10 - 0.08, abs=1e-9)


def test_pick_best_row_no_filter_pass_column():
    """没有 filter_pass 列时, 退化为纯 score 排序"""
    df = pd.DataFrame([
        {"low1": 1.0, "score": 0.05, "ann_net_min": 0.04, "S": 0.3},
        {"low1": 2.0, "score": 0.10, "ann_net_min": 0.09, "S": 0.4},
    ])
    chosen, reason = pick_best_row(df, "channel_deviation")
    assert float(chosen["score"]) == 0.10
    assert "无 filter_pass 列" in reason["sort_key"]


def test_pick_best_row_empty():
    chosen, reason = pick_best_row(pd.DataFrame(), "x")
    assert chosen is None
    assert "error" in reason


def test_pick_best_row_extracts_strategy_params(fake_sweep_df):
    chosen, reason = pick_best_row(fake_sweep_df, "channel_deviation")
    assert reason["chosen_params"] == {
        "low1": 1.5, "low2": 1.0, "high1": 1.5, "high2": 0.5}


# ============ save_best_from_sweep ============

def test_save_best_from_sweep_writes_file(isolated_defaults, fake_sweep_df):
    chosen, reason, saved_path, commit_ok = save_best_from_sweep(
        fake_sweep_df, "channel_deviation", csv_path="/tmp/fake.csv")
    assert saved_path is not None
    assert saved_path.is_file()
    data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert data["strategy"] == "channel_deviation"
    assert data["params"] == {"low1": 1.5, "low2": 1.0,
                              "high1": 1.5, "high2": 0.5}
    src = data["source"]
    assert src["kind"] == "from_sweep"
    assert src["rank"] == 1
    assert src["filter_pass"] is True
    assert src["csv"] == "/tmp/fake.csv"


def test_save_best_from_sweep_handles_filtered_mr_int_params(isolated_defaults):
    """filtered_mr 的 int 参数 (cur_ema_period/adx_period 等) 经 sweep 选出后
    Series[k] 是 numpy.int64; save_best_from_sweep 必须能落盘否则 json.dump 抛
    TypeError: Object of type int64 is not JSON serializable (2026-09-12 apply 期
    间发现; 修法: .item() 兼容 numpy/pandas 标量)
    """
    df = pd.DataFrame([
        {"band_mult": 1.8, "atr_vol_mult": 1.5, "adx_threshold": 30.0,
         "cur_ema_period": 25, "higher_period": "1h",
         "score": 0.10, "ann_net_min": 0.08, "S": 0.5, "filter_pass": True,
         "train_ann_net_min": 0.09, "test1_ann_net_min": 0.08},
    ])
    chosen, reason, saved_path, commit_ok = save_best_from_sweep(
        df, "filtered_mr", csv_path="/tmp/fake_fmr.csv")
    assert saved_path is not None
    assert saved_path.is_file()
    data = json.loads(saved_path.read_text(encoding="utf-8"))
    # params 字段必须全是 Python 原生 int/float/str (非 numpy.int64/float64)
    for k, v in data["params"].items():
        assert type(v) in (int, float, str), (
            f"{k} 类型 {type(v).__name__} 应为 JSON 原生, 实际值 {v!r}")
    assert data["params"]["cur_ema_period"] == 25
    assert data["params"]["band_mult"] == 1.8


# ============ format_reason_log ============

def test_format_reason_log_contains_key_fields(isolated_defaults, fake_sweep_df):
    chosen, reason, saved_path, commit_ok = save_best_from_sweep(
        fake_sweep_df, "channel_deviation", csv_path="/tmp/fake.csv")
    log = format_reason_log("channel_deviation", reason, saved_path, commit_ok)
    assert "channel_deviation" in log
    assert "score" in log
    assert "ann_net_min" in log
    assert "filter_pass" in log
    assert "低值" in log or "ann_net" in log
    assert "low1 = 1.5" in log
    assert "落盘文件" in log


# ============ _commit_defaults_file 在临时 git 仓库里 ============

def test_commit_defaults_file_in_temp_git_repo(tmp_path, monkeypatch):
    """在临时 git 仓库里 commit 应能成功"""
    import subprocess as sp

    # 1. 初始化 git 仓库
    sp.run(["git", "init"], cwd=tmp_path, check=True,
           capture_output=True, encoding="utf-8")
    sp.run(["git", "config", "user.email", "test@test"],
           cwd=tmp_path, check=True, capture_output=True)
    sp.run(["git", "config", "user.name", "test"],
           cwd=tmp_path, check=True, capture_output=True)

    # 2. 写一个初始 commit, 让仓库非空
    (tmp_path / "README.md").write_text("init")
    sp.run(["git", "add", "README.md"], cwd=tmp_path, check=True,
           capture_output=True)
    sp.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True,
           capture_output=True)

    # 3. 用 _defaults_loader 在仓库根目录写一个默认 JSON
    monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", str(tmp_path))
    from evtrade.strategies import _defaults_loader as dl
    p = dl.save("channel_deviation", {"low1": 1.5, "low2": 1.0,
                                       "high1": 1.5, "high2": 0.5},
                source={"kind": "from_sweep", "score": 0.1})

    # 4. 调 _commit_defaults_file, 但让它在 tmp_path 仓库里跑
    from evtrade.strategies._defaults_loader import _commit_defaults_file
    reason = {"sort_key": "filter_pass=True 且 score 最高", "rank": 1,
              "score": 0.1, "ann_net_min": 0.08, "S": 0.5,
              "filter_pass": True, "candidate_total": 3,
              "chosen_params": {"low1": 1.5}, "csv": "fake.csv"}
    # patch subprocess.Popen 让它 cwd=tmp_path
    import subprocess as _sp
    real_run = _sp.Popen

    class _PatchedPopen(_sp.Popen):
        def __init__(self, args, **kw):
            kw["cwd"] = kw.get("cwd", tmp_path)
            super().__init__(args, **kw)

    monkeypatch.setattr(_sp, "Popen", _PatchedPopen)
    ok = _commit_defaults_file(p, "channel_deviation", reason)
    assert ok is True

    # 5. 验证最近一次 commit message 包含选择原因
    r = sp.run(["git", "log", "-1", "--pretty=%B"],
               cwd=tmp_path, capture_output=True, encoding="utf-8")
    msg = r.stdout
    assert "channel_deviation" in msg
    assert "filter_pass" in msg
    assert "score" in msg
