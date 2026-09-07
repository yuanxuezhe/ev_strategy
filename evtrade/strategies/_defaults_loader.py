"""策略默认参数加载/保存 (evtrade/strategies/_defaults/<name>.json)

用途:
  - 'python -m evtrade params save <name> --params ...' 把最优参数落盘;
  - 回测/实盘不传 --params 时, 自动从这里加载;
  - 实盘接入点: from evtrade.strategies._defaults_loader import load

文件格式 (落盘 JSON):
  {
    "strategy": "<name>",
    "params": {"k1": v1, ...},
    "saved_at": "ISO 8601 timestamp",
    "source": {"kind": "from_params"|"from_csv", ...}
  }

落盘位置:
  默认 = <repo>/evtrade/strategies/_defaults/<name>.json
  可通过 EVTRADE_DEFAULTS_DIR 环境变量切换到机器特定目录。

原子写: 写到 <path>.tmp + os.replace, 避免半截 JSON 被读到。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 仓库内默认目录: evtrade/strategies/_defaults/
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR_NAME = "_defaults"


def _defaults_dir() -> Path:
    """落盘目录: EVTRADE_DEFAULTS_DIR 环境变量 > 仓库内 _defaults/

    用环境变量切换不污染仓库 (CI / 实盘机常用)。
    """
    env = os.environ.get("EVTRADE_DEFAULTS_DIR")
    if env:
        return Path(env)
    return _REPO_ROOT / "evtrade" / "strategies" / DEFAULT_DIR_NAME


def path_for(strategy_name: str) -> Path:
    """返回默认参数的落盘路径 (不论是否已存在)"""
    return _defaults_dir() / f"{strategy_name}.json"


def exists(strategy_name: str) -> bool:
    return path_for(strategy_name).is_file()


def load(strategy_name: str) -> dict:
    """加载策略的默认参数; 不存在抛 FileNotFoundError 带清晰诊断

    返回 dict: 至少含 'params' 键; 'strategy' / 'saved_at' / 'source' 可选。
    """
    p = path_for(strategy_name)
    if not p.is_file():
        raise FileNotFoundError(
            f"策略 {strategy_name!r} 没有默认参数 ({p}); "
            f"请先跑 'python -m evtrade params save {strategy_name} "
            f"--params \"k1:v1;k2:v2\"' 或显式 --params。")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save(strategy_name: str, params: dict, source: dict | None = None) -> Path:
    """保存策略默认参数; 原子写; 返回落盘路径

    source: 自由 dict, 通常含 {"kind": "from_csv", "csv": ..., "rank": N} 或
            {"kind": "from_params"}; 不强制 schema。
    """
    if not isinstance(params, dict):
        raise TypeError(f"params 必须是 dict, 收到 {type(params).__name__}")
    payload = {
        "strategy": strategy_name,
        "params": params,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source or {},
    }
    p = path_for(strategy_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)
    return p


def list_defaulted() -> list[str]:
    """扫描默认目录, 返回已有默认参数的策略名列表 (按文件名字序)"""
    d = _defaults_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def params_from_csv_row(row: dict, param_keys: list[str]) -> dict:
    """从 sweep_results.csv 的某一行抽出 params 子集 (按 param_keys 顺序)

    用于 'params save --from-csv <csv> --rank N': 取第 N 行 (1-based, 1=score 最高),
    抽出 param_keys 里的字段作为 params dict。

    类型: CSV 全字符串读入, 这里用 _auto_cast 把数字 / bool 转回 Python 原生类型,
    与 CLI --params 的 _auto_cast 保持一致。
    """
    out: dict = {}
    for k in param_keys:
        if k not in row:
            continue
        v = row[k]
        out[k] = _auto_cast(v) if isinstance(v, str) else v
    return out


def _auto_cast(s: str):
    """字符串 -> int / float / bool / str (与 cli._auto_cast 行为一致)"""
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


# ============================================================
# sweep 自动选最优并落盘 (含原因 + 自动 git commit)
# ============================================================

import subprocess  # noqa: E402  放在 save() 之后保持模块可读


def pick_best_row(df, strategy_name: str):
    """从 sweep 结果 DataFrame 选最优行 + 选参原因

    策略 (高 -> 低优先级):
      1. filter_pass=True 且 score 最高 (含可解释的 mdd / n_trades 约束)
      2. 否则按 score 降序第一行

    返回 (chosen_row, reason_dict):
      - chosen_row: pandas Series (整行)
      - reason_dict: 含 sort_key / rank / score / ann_net_min / S /
        filter_pass / 次优差距 (与 rank 2 的 score 差, 若有) /
        chosen_params / candidate_total
    """
    import pandas as _pd
    if df is None or len(df) == 0:
        return None, {"error": "empty DataFrame"}
    work = df.reset_index(drop=True)
    has_filter_col = "filter_pass" in work.columns
    passed = work[work["filter_pass"] == True] if has_filter_col else work
    if has_filter_col and len(passed) > 0:
        chosen_idx = int(passed["score"].astype(float).idxmax())
        sort_key = "filter_pass=True 且 score 最高"
    elif has_filter_col:
        chosen_idx = int(work["score"].astype(float).idxmax())
        sort_key = "filter_pass 全 False, 按 score 最高"
    else:
        chosen_idx = int(work["score"].astype(float).idxmax())
        sort_key = "无 filter_pass 列, 按 score 最高"

    chosen = work.iloc[chosen_idx]
    # 与次优对比
    runner_up = None
    runner_up_gap = None
    if len(work) > 1 and chosen_idx + 1 < len(work):
        # 找次优 (排名 != chosen_idx 的最高 score)
        for i, s in work.iterrows():
            if i == chosen_idx:
                continue
            runner_up = s
            runner_up_gap = float(chosen["score"]) - float(s["score"])
            break

    reason = {
        "sort_key": sort_key,
        "rank": int(chosen_idx) + 1,
        "score": float(chosen["score"]),
        "ann_net_min": float(chosen.get("ann_net_min", 0.0)),
        "S": float(chosen.get("S", 0.0)),
        "filter_pass": bool(chosen.get("filter_pass", False)),
        "candidate_total": int(len(work)),
        "chosen_params": {k: chosen[k] for k in chosen.index
                          if _is_strategy_param(k, strategy_name)},
    }
    if runner_up is not None:
        reason["runner_up_score"] = float(runner_up["score"])
        reason["runner_up_gap"] = runner_up_gap
    return chosen, reason


def _is_strategy_param(key: str, strategy_name: str) -> bool:
    """key 是否是 strategy_name 的 params_spec 字段"""
    try:
        from . import get_strategy_param_spec
        spec = get_strategy_param_spec(strategy_name)
        return key in spec
    except Exception:
        return False


def save_best_from_sweep(df, strategy_name: str, csv_path: str):
    """选最优 + 落盘 + 尝试 git commit + 返回 (chosen_row, reason, saved_path, commit_ok)

    与 pick_best_row + save + _commit_defaults_file 的串联入口。
    """
    chosen, reason = pick_best_row(df, strategy_name)
    if chosen is None:
        return None, reason, None, False
    params = dict(reason["chosen_params"])
    if not params:
        return None, {**reason, "error": "未抽到任何 params 字段"}, None, False

    source = {
        "kind": "from_sweep",
        "csv": csv_path,
        "rank": reason["rank"],
        "score": reason["score"],
        "ann_net_min": reason["ann_net_min"],
        "S": reason["S"],
        "filter_pass": reason["filter_pass"],
        "sort_key": reason["sort_key"],
        "candidate_total": reason["candidate_total"],
        "runner_up_score": reason.get("runner_up_score"),
        "runner_up_gap": reason.get("runner_up_gap"),
    }
    p = save(strategy_name, params, source=source)
    commit_ok = _commit_defaults_file(p, strategy_name, reason)
    return chosen, reason, p, commit_ok


def _commit_defaults_file(path: Path, strategy_name: str,
                           reason: dict) -> bool:
    """对默认参数 JSON 单独 git add + commit (中文 message 含选择原因)

    返回是否成功 (失败仅 log warning, 不抛异常)。
    """
    rel = path.as_posix()
    rel_unix = rel.replace("\\", "/")
    try:
        # 1. git add
        r = subprocess.run(["git", "add", "--", rel_unix],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"[警告] git add 失败: {r.stderr.strip()}", flush=True)
            return False
        # 2. 检查是否有差异 (空 commit 不必做)
        r = subprocess.run(["git", "diff", "--cached", "--name-only"],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0 or not r.stdout.strip():
            print(f"[跳过] {rel_unix} 无差异, 不 commit", flush=True)
            return False
        # 3. 构造中文 commit message
        msg = (
            f"参数(自动选): {strategy_name} -> score={reason.get('score', 0):.4f}, "
            f"rank={reason.get('rank', '?')}/{reason.get('candidate_total', '?')}\n\n"
            f"原因: {reason.get('sort_key', '?')}\n"
            f"  ann_net_min = {reason.get('ann_net_min', 0):+.4f}\n"
            f"  S (邻域衰减) = {reason.get('S', 0):.4f}\n"
            f"  filter_pass  = {reason.get('filter_pass', False)}\n"
            f"  score        = {reason.get('score', 0):+.4f}\n"
        )
        if reason.get("runner_up_score") is not None:
            gap = reason.get("runner_up_gap", 0.0)
            msg += (f"  runner_up score = {reason['runner_up_score']:+.4f} "
                    f"(与次优差距 {gap:+.4f})\n")
        msg += (f"\nchosen params: "
                + ", ".join(f"{k}={v!r}" for k, v in
                            (reason.get("chosen_params") or {}).items())
                + f"\n\n来源: 自动从 sweep 结果挑选 (CSV={reason.get('csv', '?')})")
        r = subprocess.run(["git", "commit", "-m", msg],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"[警告] git commit 失败: {r.stderr.strip()}",
                  flush=True)
            return False
        print(f"[已 commit] {rel_unix}", flush=True)
        return True
    except FileNotFoundError:
        print("[跳过] git 不在 PATH, 不 commit", flush=True)
        return False


def format_reason_log(strategy_name: str, reason: dict,
                       saved_path, commit_ok: bool) -> str:
    """打印"最优参数选择原因"块 (多行)"""
    lines = [
        "",
        "=" * 60,
        f"  最优参数选择原因 (策略: {strategy_name})",
        "=" * 60,
        f"  候选总数  : {reason.get('candidate_total', '?')}",
        f"  选择依据  : {reason.get('sort_key', '?')}",
        f"  选中排名  : {reason.get('rank', '?')}",
        f"  score     : {reason.get('score', 0):+.4f}",
        f"  ann_net_min: {reason.get('ann_net_min', 0):+.4f} "
        "(test 段年化超额最低值)",
        f"  S (邻域衰减): {reason.get('S', 0):.4f}",
        f"  filter_pass: {reason.get('filter_pass', False)}",
    ]
    if reason.get("runner_up_score") is not None:
        lines.append(
            f"  次优 score : {reason['runner_up_score']:+.4f} "
            f"(差距 {reason.get('runner_up_gap', 0):+.4f})")
    params = reason.get("chosen_params") or {}
    if params:
        lines.append("  选中参数  :")
        for k, v in params.items():
            # 用 str(v) 而非 {v!r}: np.float64(1.5) 的 repr 是 "np.float64(1.5)",
            # str 则为 "1.5" (test_format_reason_log_contains_key_fields 锁定)
            lines.append(f"    {k} = {v}")
    if saved_path is not None:
        lines.append(f"  落盘文件  : {saved_path}")
        lines.append(f"  git commit: {'已 commit' if commit_ok else '跳过/失败 (见日志)'}")
    lines.append("=" * 60)
    return "\n".join(lines)
