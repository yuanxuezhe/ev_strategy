## 1. 主 spec 三处修订

- [ ] 1.1 `R: Single trade-execution implementation` 删除 `scale（加仓翻倍）与 last_side 状态逻辑属于调用方状态，MAY 留在各自调用处。` 一句。验证：`grep -n "scale" openspec/specs/evtrade-architecture/spec.md` 0 命中。
- [ ] 1.2 `R: CLI options declared once` 共享选项列表去掉 `--scale`。验证同上。
- [ ] 1.3 `R: Sweep grid accepts only effective axes` 引擎轴枚举去掉 `scale /`。验证同上。

## 2. KB 清理

- [ ] 2.1 `kbs/06-交易策略详解.md`：删除 §8 整段（L206–240，含 8.1 / 8.2 子段）；后续 `## 9. 策略展示 hook` 重编为 `## 8. 策略展示 hook`。验证：`grep -n "倍投\|scale\|加仓翻倍" kbs/06-交易策略详解.md` 0 命中。
- [ ] 2.2 `kbs/07-账户与执行器.md` L49：删除 `> **历史**：早期版本支持 \`scale\` 倍投...` 整段。验证：`grep -n "scale\|倍投" kbs/07-账户与执行器.md` 0 命中。
- [ ] 2.3 `kbs/12-重构与性能内核.md` L183：网格键列表去掉 `/scale`。验证：`grep -n "scale" kbs/12-重构与性能内核.md` 0 命中。
- [ ] 2.4 `kbs/使用说明.md` L272：删除 `--scale` 行。验证：`grep -n "scale\|--scale" kbs/使用说明.md` 0 命中。

## 3. 源码注释清理

- [ ] 3.1 `evtrade/execution/base.py` L11–13 file docstring：删除 `不再维护 scale 倍投状态机 (2026-09-12 下线...)` 三行（与原 file docstring 结构相符）。验证：`grep -n "scale\|倍投" evtrade/execution/base.py` 0 命中。
- [ ] 3.2 `evtrade/execution/base.py` L71 `SimulatedExecutor` docstring：删除 `注: 早期版本支持 scale 倍投..., 2026-09-12 下线.` 一行。验证同上。

## 4. archive/ 维持原样

- [ ] 4.1 不动 `openspec/changes/archive/2026-09-09-.../2026-09-10-...`（历史快照，scale 是当时确实存在过的实现）。验证：`grep -rn "scale" openspec/changes/archive/` 命中数保持 ≥ 当前。

## 5. 验证清单

- [ ] 5.1 `grep -rn "scale" evtrade/ kbs/ openspec/specs/` 在主源码/主 spec/主 KB 命中 ≤ 0（archive/ 命中数 ≥ 当前）。
- [ ] 5.2 `openspec validate --specs` 通过。
- [ ] 5.3 `uv run pytest -q` 全绿。
- [ ] 5.4 `python -m evtrade replay --log <log.csv> --strategy channel_deviation --device cpu --against-ref` 输出 PASS（清理不动行为，应无回归）。

## 6. archive

- [ ] 6.1 合入 main 后跑 `openspec archive 2026-09-12-drop-scale-residue` 归档；确认 `openspec/specs/evtrade-architecture/spec.md` 三处 delta 已合并 + change 进入 archive/。