## ADDED Requirements

### Requirement: Market data DB connection has a sane default

`evtrade.core.data.load_bars` 在未指定 `db_url` 参数时 MUST 解析到一个对当前协作组开箱即用的 MySQL 连接串默认值;默认主机 = `192.168.10.2:33066`、库 = `evtrade`、用户 = `EvTrade`。密码中 `@` MUST URL-encode 为 `%40`。`TABLE` 默认名 MUST 仍为 `minute_bars`。这两个常量是 framework 行为契约的一部分,被 `load_bars` / `_fetch` / CLI `backtest` / `sweep` 间接依赖;`EVTRADE_DB_URL` 与 `EVTRADE_TABLE` 环境变量 MUST 优先于默认值(分别覆写连接串与表名),作为"临时切库 / 离线 / 测"的逃生口。

#### Scenario: 默认 DB_URL 指向 192.168.10.2
- **WHEN** 协作者在不设 `EVTRADE_DB_URL` 的情况下跑 `python -m evtrade backtest --strategy filtered_mr --code 159992.SZ --start 20260101 --end 20260903 --device gpu`
- **THEN** `load_bars` MUST 尝试连 `mysql+pymysql://EvTrade:p%40ssw0rd@192.168.10.2:33066/evtrade?charset=utf8mb4`(不是 `127.0.0.1:3306`),即默认主机 = `192.168.10.2`、端口 = `33066`、库 = `evtrade`

#### Scenario: EVTRADE_DB_URL 优先于默认
- **WHEN** 协作者 `export EVTRADE_DB_URL=mysql+pymysql://other:other@10.0.0.1:3306/other?charset=utf8mb4` 后再跑 `backtest`
- **THEN** `load_bars` MUST 尝试连 `10.0.0.1:3306/other`,不连 `192.168.10.2`;`EVTRADE_DB_URL` 必须解析到最终 SQLAlchemy `create_engine` 调用

#### Scenario: EVTRADE_TABLE 覆写表名
- **WHEN** 协作者 `export EVTRADE_TABLE=other_table` 后跑 `backtest`
- **THEN** `_fetch` 生成的 SQL MUST FROM `other_table`,不是 `minute_bars`;`EVTRADE_TABLE` 默认值 MUST 为 `minute_bars`(与 spec 一致)

#### Scenario: 默认 DB_URL 中 `@` 已 URL-encode
- **WHEN** 静态扫描 `evtrade/core/data.py` 中的 `DB_URL` 默认值
- **THEN** MUST 含 `p%40ssw0rd`(而非裸 `p@ssw0rd`);确保 SQLAlchemy / pymysql 不会把 `@` 当 host 分割符
