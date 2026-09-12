## Why

`evtrade/core/data.py::DB_URL` 当前默认值指向 `127.0.0.1:3306`(开发占位),在用 `mysql+pymysql://EvTrade:p@ssw0rd@192.168.10.2:33066/evtrade` 的协作者机器上,首次跑 `backtest`/`sweep` 时 100% 触发 `ConnectionRefusedError (WinError 10061)`,用户必须手动 `export EVTRADE_DB_URL=...` 才能用。这条"必须手工覆盖"是常驻协作摩擦,不是真正的安全护栏(凭据已经在仓库 KB/CLAUDE.md 中以环境变量约定形式存在)。

把默认值改成项目共享的 `192.168.10.2:33066/evtrade` 一行可省,保留 `EVTRADE_DB_URL` 环境变量覆写作为"临时切库 / 离线 / 测"逃生口。

## What Changes

- **`evtrade/core/data.py`**:`DB_URL` 默认值从 `mysql+pymysql://root:@127.0.0.1:3306/market_data?charset=utf8mb4` 改为 `mysql+pymysql://EvTrade:p%40ssw0rd@192.168.10.2:33066/evtrade?charset=utf8mb4`;`@` URL-encode 为 `%40`;`TABLE` 默认值 `minute_bars` 不变。
- **`evtrade/core/data.py::load_bars`** 错误提示文案:把"检查 `EVTRADE_DB_URL`/网络"改为显式列出当前默认主机,方便不熟的人对照排查。
- **`specs/evtrade-architecture/spec.md`** 新增 `Requirement: Market data DB connection has a sane default`,覆盖三件事:(1) 默认 DB_URL 是 192.168.10.2:33066/evtrade;(2) `EVTRADE_DB_URL` env 优先于默认;(3) `EVTRADE_TABLE` env 覆写表名(默认 `minute_bars`)。
- **`kbs/08-行情源Feed.md` §1.3** 把"默认值"段对齐到新默认值(已经在 kbs 中描述了 192.168.10.2,但与 data.py 实现不一致 —— 同步修复)。

无 **BREAKING**:`EVTRADE_DB_URL` / `EVTRADE_TABLE` 行为完全保留;旧机器设了 `EVTRADE_DB_URL` 的不受影响;唯一变化是"不设 env 时会连哪台库"。

## Capabilities

### New Capabilities
（无 —— 行情 DB 连接是 framework 行为契约的一部分,并入 `evtrade-architecture` spec 的新 Requirement,不开新 capability）

### Modified Capabilities
- `evtrade-architecture`: 新增 `Requirement: Market data DB connection has a sane default`(含 3 个 Scenario);改动的可观测行为 = "不设 `EVTRADE_DB_URL` 时 `load_bars` 试图连的具体主机"。这是行为契约变化(spec 级),不是纯实现细节,故纳入 modified。

## Impact

- **代码**:`evtrade/core/data.py` 改 ~3 行(默认值 + 错误文案)。
- **凭据可见性**:默认 URL 现在直接进源码(已 URL-encode),等同 KB 已有的描述。仍然是开发库凭据,与生产凭据分离;不需要新增 secret 管理。
- **API / 行为**:`load_bars(code, start, end, ...)` 签名不变;`db_url=None` 走默认的语义不变;`use_cache` / `cache_dir` / `warmup_days` 不变。
- **测试**:新增 `tests/test_data_default_db_url.py`,断言 `data.DB_URL` 是 192.168.10.2 主机 + `EvTrade` 用户 + `%40` 编码的密码 + `evtrade` 库,以及 `os.environ` 设了 `EVTRADE_DB_URL` 时 `load_bars` 优先使用。
- **KB / CLAUDE.md**:KB §1.3 文案对齐;CLAUDE.md 不动(spec/KB 已是 single source of truth)。
