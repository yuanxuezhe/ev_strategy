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
    """
    return {k: row[k] for k in param_keys if k in row}
