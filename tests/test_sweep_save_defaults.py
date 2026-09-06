"""sweep 自动选最优并落盘 + 自动 git commit

覆盖:
  - pick_best_row: filter_pass 优先, 否则 score 第一行
  - pick_best_row: 与次优差距 (runner_up_gap) 计算
  - save_best_from_sweep: 落盘文件包含正确 source 字段
  - format_reason_log: 输出含 score / ann_net_min / 选中参数
  - sweep --save-defaults 走完整落盘流程 (用 fake df, 不真跑 sweep)
  - --no-save-defaults 跳过
  - _commit_defaults_file: 在临时 git 仓库里能 commit
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from evtrade.cli import sweep_main
from evtrade.strategies._defaults_loader import (
    format_reason_log,
    pick_best_row,
    save_best_from_sweep,
)


# ---------- fixtures ----------


@pytest.fixture
def isolated_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("EVTRADE_DEFAULTS_DIR", str(tmp_path))
    return tmp_path


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


# ---------- pick_best_row ----------


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


# ---------- save_best_from_sweep ----------


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


# ---------- format_reason_log ----------


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


# ---------- _commit_defaults_file 在临时 git 仓库里 ----------


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
    # patch subprocess.run 让它 cwd=tmp_path
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
