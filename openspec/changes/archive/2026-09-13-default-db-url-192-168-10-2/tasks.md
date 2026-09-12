## 1. 改 data.py 默认值

- [ ] 1.1 把 `evtrade/core/data.py` 第 24-25 行 `DB_URL` 默认值改为 `mysql+pymysql://EvTrade:p%40ssw0rd@192.168.10.2:33066/evtrade?charset=utf8mb4`(注意 `@` → `%40`)。验证:`grep -n "DB_URL" evtrade/core/data.py` 命中 `192.168.10.2:33066` + `%40ssw0rd`。
- [ ] 1.2 保留 `TABLE = os.environ.get("EVTRADE_TABLE", "minute_bars")` 不变。验证:`grep -n "EVTRADE_TABLE" evtrade/core/data.py` 仍命中两处(env 读 + 默认值)。
- [ ] 1.3 `load_bars` 错误文案(`data.py:53-55` `RuntimeError`)保留"请检查 EVTRADE_DB_URL/网络, 或改用 --synthetic-days"原文,不破坏 spec 的 Scenario `数据库无数据` 行为。

## 2. CLI 层补默认主机提示

- [ ] 2.1 在 `evtrade/cli.py::_run_backtest`(148 行 `bars = load_bars(...)`)外加 `try/except sqlalchemy.exc.OperationalError as e`,捕到时打印一行中文提示"默认 DB_URL = 192.168.10.2:33066 (库 evtrade); 可设 EVTRADE_DB_URL 覆写, 或 --synthetic-days 跳过",再 `raise SystemExit(2)`。验证:不设 EVTRADE_DB_URL 且库不可达时,终端打印包含 `192.168.10.2:33066` + `EVTRADE_DB_URL`。

## 3. spec 同步(主 spec 落地)

- [ ] 3.1 把 `openspec/changes/default-db-url-192-168-10-2/specs/evtrade-architecture/spec.md` 的 `ADDED Requirements`(整个 `### Requirement: Market data DB connection has a sane default` 段含 4 个 Scenario)合入 `openspec/specs/evtrade-architecture/spec.md`,插在 `## Requirements` 末尾(在新 Requirement 之前/之后均可,保持编号连续)。验证:`grep -n "Market data DB connection has a sane default" openspec/specs/evtrade-architecture/spec.md` 命中。
- [ ] 3.2 同步 `kbs/08-行情源Feed.md` §1.3(数据库连接串段):把"默认值 `mysql+pymysql://EvTrade:p%40ssw0rd@192.168.10.2:33066/evtrade?charset=utf8mb4`"明示为 `data.DB_URL` 当前实现值,而非"文档约定"。验证:`grep -n "192.168.10.2" kbs/08-行情源Feed.md` 命中,前后文含"data.DBURL 默认值"。

## 4. 测试

- [ ] 4.1 新建 `tests/test_data_default_db_url.py`,内容含 4 个断言:(a) `data.DB_URL` 字符串以 `mysql+pymysql://EvTrade:` 开头;(b) 含 `192.168.10.2:33066`;(c) 含 `p%40ssw0rd`(`@` 已编码);(d) 含 `/evtrade?` 库名。验证:`uv run pytest tests/test_data_default_db_url.py -v` 4 通过。
- [ ] 4.2 同一文件加 monkeypatch 测试:monkeypatch `os.environ` 设 `EVTRADE_DB_URL=...` 与 `EVTRADE_TABLE=foo`,import 后 `data.DB_URL` / `data.TABLE` 必须读取新值。验证:测试通过。
- [ ] 4.3 跑全量 `uv run pytest -q`,确认无回归(应保持原有 passed 数)。验证:无 failed / errored。

## 5. 验证清单

- [ ] 5.1 `openspec validate --specs` 通过(主 spec 已合入新 Requirement)。验证:命令退出码 0,无 error。
- [ ] 5.2 不设 `EVTRADE_DB_URL` 时 `python -m evtrade backtest --strategy filtered_mr --code 159992.SZ --start 20260101 --end 20260903 --device gpu` 不再报 `ConnectionRefused 127.0.0.1:3306`;要么正常跑出信号轨迹,要么报默认主机的 OperationalError(被 2.1 包的中文提示接住)。验证:终端不再出现 `127.0.0.1`。
- [ ] 5.3 设 `EVTRADE_DB_URL` 指向不存在的主机时,报"默认 DB_URL = 192.168.10.2:33066"提示。验证:提示文案出现。

## 6. archive

- [ ] 6.1 任务 1-5 全绿且 commit 后跑 `openspec archive default-db-url-192-168-10-2` 归档;`openspec/changes/default-db-url-192-168-10-2/` 进入 `openspec/changes/archive/`,delta 已合并进主 spec。
