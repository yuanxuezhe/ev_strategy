## Context

`evtrade/core/data.py::DB_URL` 当前默认指向 `127.0.0.1:3306`(占位),但 `kbs/08-行情源Feed.md` §1.3 已经把项目真实库描述为 `192.168.10.2:33066/evtrade` + `EvTrade:p@ssw0rd`。文档与实现不一致,首次 `backtest` 100% 触发 `ConnectionRefusedError`。本次修改仅同步实现与 KB。

参考:`proposal.md` Why / What Changes;`specs/evtrade-architecture/spec.md` 新 Requirement `Market data DB connection has a sane default`。

## Goals / Non-Goals

**Goals:**
- `data.DB_URL` 默认值与 KB §1.3 + spec 一致,首次协作零配置可用
- `EVTRADE_DB_URL` / `EVTRADE_TABLE` env 优先级保留(逃生口不动)
- 默认值的错误文案显式提示默认主机,方便对照排查
- 新增 `tests/test_data_default_db_url.py` 锁住默认值形态(防回归)

**Non-Goals:**
- 不引入 `.env` / `dotenv` / secret 管理机制(凭据已显式入 KB/源码,不是新增敏感面)
- 不改 `load_bars` / `_fetch` SQL 模板或列名
- 不动 CLI 选项(本就无 `--db-url`)
- 不动 `TABLE` 默认值

## Decisions

### D1: `@` 在源码里硬编码为 `%40`,而非依赖 Python 自动 encode

**为什么**:SQLAlchemy 接受原始 URL,但 pymysql 在某些边界场景下对未编码 `@` 的容忍度不一致;显式 `%40` 在所有路径都安全。**备选**:`urllib.parse.quote` 运行时拼 → 引入运行期复杂度,且连接串不可静态扫;**否决**。

### D2: 默认值进源码,而不是要求每个协作者 `export EVTRADE_DB_URL`

**为什么**:KB §1.3 已暴露主机/端口/库名,等价"半公开"。强制每个 shell 都 export 是常驻摩擦,而开发库凭据(非生产)放到默认值,与 KB/CLAUDE.md 既有的"环境变量可覆写"约定不冲突。**备选**:改默认值用 `~/.evtrade/db_url` 文件 → 引入新文件 + 新解析逻辑,over-engineered。**否决**。

### D3: 错误文案显式含默认主机

**为什么**:`load_bars` 现有 `RuntimeError("数据库无数据 ... 请检查 EVTRADE_DB_URL/网络, 或改用 --synthetic-days ...")` 在连不上时**不会触发**(因为 SQLAlchemy 自己先抛 `OperationalError`);`OperationalError` 文案又是英文 pymysql 堆栈,对不熟的人不友好。**做法**:不动 SQLAlchemy 抛的异常,改在 `cli.py::_run_backtest` 用 `try/except OperationalError` 包一层,补一行中文提示"默认 DB_URL = 192.168.10.2:33066, 可设 EVTRADE_DB_URL 覆写"。**备选**:改 `data._fetch` 包 try/except → 让 library 层感知业务文案,污染分层。**否决**,在 CLI 层做。

### D4: 测试用静态常量断言,不做真连测

**为什么**:`test_data_default_db_url.py` 跑在 CI,CI 无 MySQL,真连会 100% 失败。**做法**:断言 `data.DB_URL` 字符串形态 + `os.environ.get` 覆写路径(monkeypatch),不连库。**备选**:加 `pytest.mark.requires_db` + `mysql_container` fixture → 当前测试基建无 container;out of scope。

## Risks / Trade-offs

- **[凭据进源码]** → 与 KB §1.3 已暴露的事实一致,非新增敏感面;开发库与生产库分离,事故半径可控。
- **`192.168.10.2` 是内网地址,出差 / VPN 断开时无法直连** → 走 `EVTRADE_DB_URL` 切到本地 / 跳板 / 合成数据(`--synthetic-days`)逃生;已存在,本次不重复造。
- **KB §1.3 默认值段落已写 192.168.10.2(本次让实现追齐文档)** → 文档/实现双向同步,无半更新风险。
- **改的是 framework 默认值,跨所有协作者** → 是预期行为,KB/spec 同步是 release gate。

## Migration Plan

无迁移 —— 默认值变化对所有现存调用透明:
- 已设 `EVTRADE_DB_URL` 的协作者:行为不变
- 已设 `EVTRADE_TABLE` 的协作者:行为不变
- 第一次跑的新协作者:从"必败"变为"开箱即用"
- 回退:任何时候 `git revert` 即可(spec delta 也会被 archive 流程保留)

## Open Questions

（无）
