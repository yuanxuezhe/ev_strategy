from __future__ import annotations
"""numba 流式决策内核 —— DSL splice 模板 + 数据流基础设施

================================================================
⚠️  冻结层核心模块 (DSL splice 模板)  ⚠️
================================================================
本文件是**DSL splice 的模板源**:
  - 通用部分: KernelState / step / _execute / summarize / 时间桶 / EMA 增量
    任何 DSL 策略都通过 core.kernel_dsl.build_dsl_kernel(name)
    把本文件源码整段替换 DSL-STRATEGY-BEGIN/END 区间, exec 出该策略专用的模块。
  - 公共 API (make_state / run_backtest / step / _execute / summarize / bars_to_arrays
    等): 留作兼容性引用, 但**直接调用 step() 时信号为 0** (因 _strategy_check 函数
    体已清空, 渲染产物从 build_dsl_kernel 注入); 真实执行必须经 dsl_kernel(name)。
  - 72 项差分测试 (test_differential.py + test_kernel_unit.py) 现已改为走 dsl_kernel。

DSL 渲染契约 (唯一真相源在 strategies/dsl.py::_CTX_TO_KERNEL):
  ctx.p0..p15        -> st.p0..st.p15    (按策略 params_spec 声明顺序)
  ctx.cur_ts/high/low/close/open/volume -> st.cur_*
  ctx.up / ctx.dw    -> _strategy_check 的函数参数 (裸名)
  ctx.low_hit / high_hit        -> st.low_hit / st.high_hit
  ctx._low_acted / _high_acted  -> st.low_acted / st.high_acted
  ctx._bucket_ts     -> st.lock_ts
  其余 ctx.x / 裸名 x -> 函数局部变量 (首次赋值前不可读)

修改本文件**前**请读 kbs/12-重构与性能内核.md 第 2 节"流式内核"。
通用部分的改动 (step / _ema_push / _execute / summarize / KernelState 字段)
需同步改 evtrade/gpu.py CUDA source (且重新编译 PTX) 与 kbs/05/06/07/13 文档。
策略逻辑**不要**写在本文件 —— 改 strategies/<name>.py::_DSL docstring。
================================================================

设计要点:
  * "流式"是语义属性, 不是实现属性: 内核逐根处理 bar, 决策只依赖截至当前 bar 的
    数据与状态递推, 绝不使用未来数据。回测只是把整段数组灌进同一个 step() 循环。
  * 与参考实现 (aggregator/indicators/strategy/execution) 的对应关系逐行对齐:
      bucket_ts_encoded   <-> timeutils.compute_bucket      (字符串算法 -> 纯整数算法)
      _ema_push/_ema_current <-> indicators.IncrementalEMA.push/current (同一表达式)
      _strategy_check     <-> strategy.ChannelDeviationStrategy.check
      _execute            <-> execution.SimulatedExecutor.trade + Account.apply
    浮点表达式保持与参考实现完全相同的运算顺序, 保证逐位一致 (bitwise)。
  * 一个内核, 四种用法:
      回测:   run_backtest(state, bar数组...)  (内部仍是逐根递推)
      实盘:   step(state, stime, o, h, l, c, v) 单根调用 (微秒级)
      扫描:   每组参数一个 state, run_backtest N 次 (nogil, 线程池并行)
      GPU:    同一步进语义移植为 CUDA kernel (见 gpu.py)
  * 状态用 jitclass 承载 (纯标量 + 可选成交记录数组), 无堆分配, 每次扫描零成本新建。

参考实现中 warmup_until 为字符串阈值, 内核统一为 14 位整数 (字典序 == 数值序)。
参考实现的 flush() 收尾回调已证明对信号/账户无影响 (单桶锁保证), 内核无需复现。
参考实现在空数据时 flush 会 IndexError; 内核对空数组安全返回零交易 (已知差异, 仅空数据)。
"""

import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba.types import boolean, float64, int64, int8

from .timeutils import resolve_period_seconds  # 同包导入; 保留 ``from evtrade.kernel import resolve_period_seconds`` 路径


# ============ 时间戳: 14位整数 <-> epoch 秒 (纯整数历法) ============

@njit(cache=True)
def _days_from_civil(y: int64, m: int64, d: int64) -> int64:
    """公历日期 -> 自 1970-01-01 的天数 (Howard Hinnant 算法)"""
    y = y - (1 if m <= 2 else 0)
    era = y // 400
    yoe = y - era * 400                                  # [0, 399]
    mp = (m + 9) % 12                                    # 3月=0
    doy = (153 * mp + 2) // 5 + d - 1                    # [0, 365]
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy        # [0, 146096]
    return era * 146097 + doe - 719468


@njit(cache=True)
def _civil_from_days(z: int64):
    """自 1970-01-01 的天数 -> 公历日期 (y, m, d)"""
    z = z + 719468
    era = z // 146097
    doe = z - era * 146097                               # [0, 146096]
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + (3 if mp < 10 else -9)
    return y + (1 if m <= 2 else 0), m, d


@njit(cache=True)
def encoded_to_epoch(t: int64) -> int64:
    """YYYYMMDDHHmmss 整数 -> epoch 秒"""
    y = t // 10000000000
    mo = (t // 100000000) % 100
    d = (t // 1000000) % 100
    h = (t // 10000) % 100
    mi = (t // 100) % 100
    s = t % 100
    return _days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + s


@njit(cache=True)
def epoch_to_encoded(e: int64) -> int64:
    """epoch 秒 -> YYYYMMDDHHmmss 整数"""
    days = e // 86400
    sod = e % 86400
    h = sod // 3600
    mi = (sod % 3600) // 60
    s = sod % 60
    y, mo, d = _civil_from_days(days)
    return ((((y * 100 + mo) * 100 + d) * 100 + h) * 100 + mi) * 100 + s


@njit(cache=True)
def bucket_ts_encoded(t: int64, period_seconds: int64) -> int64:
    """合并桶时间戳 (前开后闭, 标注右端点) —— 任意 m/h/d 周期通用。

    stime 即北京墙钟时间, naive-epoch 的午夜天然是 86400 的整数倍, 故直接以
    naive-epoch 为锚: e0 = (e // P) × P; 恰在边界 (e % P == 0) → ts = e0,
    否则 ts = e0 + P。
    对老 7 周期 (1m..1d) 与原"字段取整"算法逐例等价 (锚定后能整除天长的周期,
    边界恰好落在同一钟面位置); 对 90m/7m/3d 等给出连续无重叠的分区。
    注意不能对分钟/小时"字段"直接取整: 90m 的分钟字段 0-59 无法对 90 取整,
    3d 会在月初拼出 day=00 的非法日期 (原字段算法的固有缺陷)。
    """
    e = encoded_to_epoch(t)
    e0 = (e // period_seconds) * period_seconds
    if e == e0:
        return epoch_to_encoded(e0)
    return epoch_to_encoded(e0 + period_seconds)


# ============ EMA 标量助手 (与 IncrementalEMA 同一表达式) ============

@njit(cache=True)
def _ema_push(value: float64, s_sum: float64, s_count: int64, s_ema: float64,
              p: int64):
    """闭合一个新桶, O(1) 更新 EMA 状态; 返回 (sum, count, ema)"""
    if s_count < p:
        s_sum += value
        s_count += 1
        if s_count == p:
            s_ema = s_sum / p             # SMA seed
    else:
        k = 2.0 / (p + 1.0)
        s_ema = value * k + s_ema * (1.0 - k)
        s_count += 1
    return s_sum, s_count, s_ema


@njit(cache=True)
def _ema_current(s_sum: float64, s_count: int64, s_ema: float64,
                 p: int64, pending: float64) -> float64:
    """带当前未闭合桶的临时 EMA (不改状态); 数据不足返回 NaN (对应参考的 None)"""
    if s_count < p - 1:
        return np.nan
    if s_count == p - 1:
        return (s_sum + pending) / p      # pending 凑满 p 个 -> SMA seed
    k = 2.0 / (p + 1.0)
    return pending * k + s_ema * (1.0 - k)


# ============ 内核状态 (jitclass, 纯标量 + 可选成交记录) ============

_STATE_SPEC = [
    # -- 周期/策略/资金 配置 --
    ("period_seconds", int64),
    ("warmup_until", int64), ("tf1", int64),
    # 通用策略参数 (按 params_spec 顺序; p0..p15, 任意策略用)
    ("p0", float64), ("p1", float64), ("p2", float64), ("p3", float64),
    ("p4", float64), ("p5", float64), ("p6", float64), ("p7", float64),
    ("p8", float64), ("p9", float64), ("p10", float64), ("p11", float64),
    ("p12", float64), ("p13", float64), ("p14", float64), ("p15", float64),
    ("init_cash", float64), ("init_position", float64), ("trade_qty", float64),
    ("scale", float64), ("last_side", int64), ("cur_qty", float64),
    # -- 资金模式 (阶段 2: --all-in / --buy-pct / --sell-pct) --
    #   buy_pct, sell_pct ∈ [0, 1]; 0 表示关闭 (保持旧 fixed-qty 行为)
    #   all_in=True 时: BUY 吃满 cash, SELL 清光 position (兼容旧简写)
    ("buy_pct", float64), ("sell_pct", float64), ("all_in", boolean),
    # -- 聚合器: 当前桶 (闭合桶无需保存, 只在其切换时喂 EMA) --
    ("has_cur", boolean), ("cur_ts", int64),
    ("cur_open", float64), ("cur_high", float64), ("cur_low", float64),
    ("cur_close", float64), ("cur_volume", float64), ("cur_count", int64),
    ("cur_mark", int64),
    # -- EMA 通道 (上/下轨各一组增量状态) --
    ("up_sum", float64), ("up_count", int64), ("up_ema", float64),
    ("dw_sum", float64), ("dw_count", int64), ("dw_ema", float64),
    # -- 策略锁存状态 --
    ("low_hit", boolean), ("high_hit", boolean),
    ("lock_ts", int64),
    ("low_acted", boolean), ("high_acted", boolean),
    # -- 账户 --
    ("cash", float64), ("position", float64), ("last_price", float64),
    # -- 绩效统计 (内核新增: 扫描选参用) --
    ("n_trades", int64), ("n_buy", int64), ("n_sell", int64),
    ("turnover", float64),
    ("peak_equity", float64), ("max_drawdown", float64),
    ("peak_eq_ts", int64), ("valley_eq_ts", int64), ("recovered", boolean),
    # -- 绩效统计: 超额曲线 (相对基线, 见 summarize/13号文档) --
    ("cur_day", int64), ("day_init", boolean),
    ("x_day_end", float64), ("x_day_end_prev", float64), ("day_end_init", boolean),
    ("d_sum", float64), ("d_sum2", float64), ("d_n", int64),
    ("d_neg_sum2", float64), ("d_neg_n", int64),       # Sortino: 仅下行日二阶距 / 计数
    ("x_peak", float64), ("x_mdd", float64),
    ("first_ts", int64), ("last_ts", int64), ("first_ts_set", boolean),
    ("init_equity", float64),                           # 期初权益 (供 CAGR)
    # -- 成交记录 (可选) --
    ("record_trades", boolean),
    ("trade_ts", int64[:]), ("trade_side", int8[:]), ("trade_qty_a", float64[:]),
    ("trade_price", float64[:]), ("trade_cash_after", float64[:]),
]


@jitclass(_STATE_SPEC)
class KernelState:
    """单次回测/实盘会话的全部状态 (每线程独立, 天然并发安全)

    p0..p15: 通用策略参数 (按策略的 params_spec 声明顺序填入; 任意 DSL 策略同路径)
    """

    def __init__(self, period_seconds, warmup_until, tf1,
                 init_cash, init_position, trade_qty, scale,
                 buy_pct, sell_pct, all_in,
                 record_trades, trade_cap,
                 p0=0.0, p1=0.0, p2=0.0, p3=0.0,
                 p4=0.0, p5=0.0, p6=0.0, p7=0.0,
                 p8=0.0, p9=0.0, p10=0.0, p11=0.0,
                 p12=0.0, p13=0.0, p14=0.0, p15=0.0):
        self.period_seconds = period_seconds
        self.warmup_until = warmup_until
        self.tf1 = tf1
        self.p0, self.p1, self.p2, self.p3 = p0, p1, p2, p3
        self.p4, self.p5, self.p6, self.p7 = p4, p5, p6, p7
        self.p8, self.p9, self.p10, self.p11 = p8, p9, p10, p11
        self.p12, self.p13, self.p14, self.p15 = p12, p13, p14, p15
        self.init_cash = init_cash
        self.init_position = init_position
        self.trade_qty = trade_qty
        self.scale = scale
        self.last_side = 0            # 上一信号方向: 0=无 1=BUY -1=SELL
        self.cur_qty = trade_qty      # 下一次同向信号的基础数量 (倍投累乘)
        self.buy_pct = buy_pct
        self.sell_pct = sell_pct
        self.all_in = all_in
        self.has_cur = False
        self.cur_ts = 0
        self.cur_open = 0.0
        self.cur_high = 0.0
        self.cur_low = 0.0
        self.cur_close = 0.0
        self.cur_volume = 0.0
        self.cur_count = 0
        self.cur_mark = 1
        self.up_sum = 0.0
        self.up_count = 0
        self.up_ema = np.nan
        self.dw_sum = 0.0
        self.dw_count = 0
        self.dw_ema = np.nan
        self.low_hit = False
        self.high_hit = False
        self.lock_ts = 0
        self.low_acted = False
        self.high_acted = False
        self.cash = init_cash
        self.position = init_position
        self.last_price = 0.0
        self.n_trades = 0
        self.n_buy = 0
        self.n_sell = 0
        self.turnover = 0.0
        self.peak_equity = 0.0
        self.max_drawdown = 0.0
        self.peak_eq_ts = 0
        self.valley_eq_ts = 0
        self.recovered = True
        self.cur_day = 0
        self.day_init = False
        self.x_day_end = 0.0
        self.x_day_end_prev = np.nan
        self.day_end_init = False
        self.d_sum = 0.0
        self.d_sum2 = 0.0
        self.d_n = 0
        self.d_neg_sum2 = 0.0
        self.d_neg_n = 0
        self.x_peak = -1.0e18
        self.x_mdd = 0.0
        self.first_ts = 0
        self.last_ts = 0
        self.first_ts_set = False
        self.init_equity = 0.0                              # 占位 0, 首根策略期 bar 锚定
        self.record_trades = record_trades
        self.trade_ts = np.empty(trade_cap, np.int64)
        self.trade_side = np.empty(trade_cap, np.int8)
        self.trade_qty_a = np.empty(trade_cap, np.float64)
        self.trade_price = np.empty(trade_cap, np.float64)
        self.trade_cash_after = np.empty(trade_cap, np.float64)


# ============ 策略状态机 (DSL 渲染目标; 本尊函数体为空) ============

@njit(nogil=True)
def _strategy_check(st, up: float64, dw: float64) -> int64:
    """策略检查: 返回 0=无信号 / 1=BUY / -1=SELL

    本函数体由 kernel_dsl.build_dsl_kernel 通过 DSL-STRATEGY-BEGIN/END 标记
    整段注入渲染产物。**所有 DSL 策略**都走 build_dsl_kernel
    路径, 此函数体为空是正常的 (本尊上的 step() 会返回 0); 真实执行请用
    `dsl_kernel(name)` 返回的特化模块。

    历史: 2026-09 重构前, 此函数体内嵌特定策略的手写 st 形式;
    现已删除, 所有策略走 dsl_kernel(name) 共用渲染管线
    —— 一份 DSL docstring 派生 Python/numba/CUDA 三端。
    """
    # ==== DSL-STRATEGY-BEGIN (kernel_dsl.py 按此标记整段替换) ====
    pass    # body 由 build_dsl_kernel 注入
    # ==== DSL-STRATEGY-END ====
    return 0


# ============ 模拟成交 (与 SimulatedExecutor.trade + Account.apply 逐行等价) ============

@njit(nogil=True)
def _execute(st, signal: int64, price: float64, ts: int64):
    # 倍投: 连续同方向信号数量累乘 (乘法逐次进行, 保证 CPU/GPU 逐位一致);
    # 方向翻转重置为基础数量。信号即计数 (即使因资金/持仓不足未成交)。
    if signal == st.last_side:
        st.cur_qty = st.cur_qty * st.scale
    else:
        st.cur_qty = st.trade_qty
        st.last_side = signal

    if signal == 1:                                        # BUY
        if price > 0.0:
            max_by_cash = st.cash / price
            if st.buy_pct > 0.0:
                # 比例模式: 按当前现金的 buy_pct 算目标, 与资金上限取小;
                # 比例本身就是按当下资金算的, 不再被 trade_qty 上限截断
                target = st.buy_pct * max_by_cash
                q = target if target < max_by_cash else max_by_cash
            else:
                # 固定数量模式: min(trade_qty 倍投, 资金上限)
                q = st.cur_qty if st.cur_qty < max_by_cash else max_by_cash
        else:
            q = 0.0
        if q <= 0:
            return
        st.cash -= q * price
        st.position += q
        st.n_trades += 1
        st.n_buy += 1
        st.turnover += q * price
    else:                                                  # SELL
        if st.sell_pct > 0.0:
            # 比例模式: 按当前持仓的 sell_pct 算目标, 与持仓上限取小
            target = st.sell_pct * st.position
            q = target if target < st.position else st.position
        else:
            # 固定数量模式: min(trade_qty 倍投, 持仓上限)
            q = st.cur_qty if st.cur_qty < st.position else st.position
        if q <= 0:
            return
        st.cash += q * price
        st.position -= q
        st.n_trades += 1
        st.n_sell += 1
        st.turnover += q * price
    if st.record_trades and st.n_trades <= st.trade_ts.shape[0]:
        i = st.n_trades - 1
        st.trade_ts[i] = ts
        st.trade_side[i] = signal
        st.trade_qty_a[i] = q
        st.trade_price[i] = price
        st.trade_cash_after[i] = st.cash


# ============ 单根步进 (流式核心: 回测/实盘/扫描/GPU 共用) ============

@njit(nogil=True)
def step(st, stime: int64, o: float64, h: float64, l: float64,
         c: float64, v: float64):
    """处理一根 1m bar, 返回 (signal, up, dw); signal: 0/1/-1, up/dw 为 NaN 时表示未就绪

    处理顺序与参考 Engine.on_bars 严格一致:
      桶切换(闭合旧桶->喂EMA) -> mark 预热检查 -> 更新 last_price ->
      通道值 -> 策略状态机 -> 模拟成交 -> 权益轨迹(最大回撤, 内核新增)。
    """
    # 1) 桶时间戳 + mark
    ts = bucket_ts_encoded(stime, st.period_seconds)
    if st.warmup_until != 0 and stime < st.warmup_until:
        mark = 0
    else:
        mark = 1

    # 2) 桶切换: 闭合旧桶 -> push 进 EMA (等价 _sync_ema), 新桶初始化
    if st.has_cur and ts != st.cur_ts:
        s, n, e = _ema_push(st.cur_high, st.up_sum, st.up_count, st.up_ema, st.tf1)
        st.up_sum, st.up_count, st.up_ema = s, n, e
        s, n, e = _ema_push(st.cur_low, st.dw_sum, st.dw_count, st.dw_ema, st.tf1)
        st.dw_sum, st.dw_count, st.dw_ema = s, n, e
        st.has_cur = False
    if not st.has_cur:
        st.has_cur = True
        st.cur_ts = ts
        st.cur_open = o
        st.cur_high = h
        st.cur_low = l
        st.cur_close = c
        st.cur_volume = v
        st.cur_count = 1
        st.cur_mark = mark
    else:
        if h > st.cur_high:
            st.cur_high = h
        if l < st.cur_low:
            st.cur_low = l
        st.cur_close = c
        st.cur_volume += v
        st.cur_count += 1
        st.cur_mark = mark

    # 3) 通道值 (预热期也计算, 供逐 bar 轨迹输出; O(1))
    up = _ema_current(st.up_sum, st.up_count, st.up_ema, st.tf1, st.cur_high)
    dw = _ema_current(st.dw_sum, st.dw_count, st.dw_ema, st.tf1, st.cur_low)

    # 4) 预热桶: 指标已累积, 不驱动策略
    if st.cur_mark == 0:
        return 0, up, dw

    # 4.5) 策略期首末时间 (年化用)
    if not st.first_ts_set:
        st.first_ts = stime
        st.first_ts_set = True
    st.last_ts = stime

    # 5) 最新价 (在策略判断前更新, 与参考一致)
    price = st.cur_close
    st.last_price = price

    # 6) 策略状态机
    signal = _strategy_check(st, up, dw)

    # 7) 模拟成交
    if signal != 0:
        _execute(st, signal, price, ts)

    # 8) 权益轨迹 (逐 bar 市值盯市: 权益回撤) + 超额曲线 (相对基线: 选参主口径)
    eq = st.cash + st.position * st.last_price
    if eq > st.peak_equity:
        st.peak_equity = eq
        st.peak_eq_ts = stime
        st.recovered = True
    if st.peak_equity > 0.0:
        dd = (st.peak_equity - eq) / st.peak_equity
        if dd > st.max_drawdown:
            st.max_drawdown = dd
            st.valley_eq_ts = stime
            st.recovered = False

    # 期初权益锚定 (策略期首根 bar 才设;供 CAGR)
    if st.init_equity == 0.0:
        st.init_equity = st.init_cash + st.init_position * st.last_price

    # 超额率 x_t = (策略权益 - 基线权益) / 基线权益; 基线 = 期初资金 + 期初持仓×现价
    base = st.init_cash + st.init_position * st.last_price
    x = (eq - base) / base
    if x > st.x_peak:
        st.x_peak = x
    xdd = st.x_peak - x
    if xdd > st.x_mdd:
        st.x_mdd = xdd
    # 日频差分 (日切时用上一日末与上上日末的 x 结算一次 Δx, 供超额 Sharpe)
    day = stime // 1000000
    if not st.day_init:
        st.day_init = True
        st.cur_day = day
    elif day != st.cur_day:
        if st.day_end_init:
            d = st.x_day_end - st.x_day_end_prev
            st.d_sum += d
            st.d_sum2 += d * d
            st.d_n += 1
            if d < 0.0:
                st.d_neg_sum2 += d * d
                st.d_neg_n += 1
        st.x_day_end_prev = st.x_day_end
        st.day_end_init = True
        st.cur_day = day
    st.x_day_end = x

    return signal, up, dw


# ============ 批量运行 (回测/扫描: 灌入 bar 数组, 内部仍是逐根递推) ============

@njit(nogil=True)
def run_backtest(st, stime, o, h, l, c, v, sig_out, up_out, dw_out) -> int64:
    """从旧到新灌入全部 1m bar; 可选输出轨迹: sig_out(int8) / up_out,dw_out(float64)

    轨迹数组传长度 0 表示不记录。nogil=True: 多线程可并行各自 state。
    """
    n = stime.shape[0]
    t_sig = sig_out.shape[0] > 0
    t_ind = up_out.shape[0] > 0
    for i in range(n):
        sig, up, dw = step(st, stime[i], o[i], h[i], l[i], c[i], v[i])
        if t_sig:
            sig_out[i] = sig
        if t_ind:
            up_out[i] = up
            dw_out[i] = dw
    return st.n_trades


@njit(nogil=True)
def run_backtest_trace(st, stime, o, h, l, c, v,
                       sig_out, up_out, dw_out,
                       ts_out, o_out, h_out, l_out, c_out, v_out) -> int64:
    """全轨迹版: 额外记录每根 1m bar 所在桶的状态 (桶 ts 与运行中 OHLCV + 信号)

    供 bucket_table 生成"每根周期K线"明细 (OHLCV + sig 轨迹)。
    指标 (EMA 通道 up/dw 等) 也写入对应数组, 但不归 framework.bucket_table
    处理; 调用方可经 StrategyBase.get_extra_bucket_columns(tab, up=..., dw=...)
    钩子追加策略专属列。各轨迹数组传长度 0 表示不记录。
    """
    n = stime.shape[0]
    t_sig = sig_out.shape[0] > 0
    t_ind = up_out.shape[0] > 0
    t_bar = ts_out.shape[0] > 0
    for i in range(n):
        sig, up, dw = step(st, stime[i], o[i], h[i], l[i], c[i], v[i])
        if t_sig:
            sig_out[i] = sig
        if t_ind:
            up_out[i] = up
            dw_out[i] = dw
        if t_bar:
            ts_out[i] = st.cur_ts
            o_out[i] = st.cur_open
            h_out[i] = st.cur_high
            l_out[i] = st.cur_low
            c_out[i] = st.cur_close
            v_out[i] = st.cur_volume
    return st.n_trades


def bucket_table(stime: np.ndarray, sig: np.ndarray,
                 ts_out, o_out, h_out, l_out, c_out, v_out) -> dict:
    """从全轨迹构建"每根周期K线"表格 (纯 numpy 向量化, framework 层)。

    每行 = 一个周期桶在**闭合时点**的状态:
      ts/open/high/low/close/volume/count  桶的最终 OHLCV 与含 1m 根数
      sig                                  桶最后一根 bar 的信号 (0/1/-1)
      n_sig                                桶内信号总数

    **框架只输出行情 + 信号轨迹的桶聚合; 任何指标列** (如 EMA 通道 up/dw,
    或策略专属的偏离百分比 low_dev/high_dev/...) **均不在此处计算**。
    调用方按需把指标数组 (来自 `run_backtest_trace` 或策略模块)
    传给 `strategy.get_extra_bucket_columns(tab, ...)` 等钩子拼接。
    """
    n = len(ts_out)
    new_bucket = np.r_[True, ts_out[1:] != ts_out[:-1]]
    first_idx = np.flatnonzero(new_bucket)
    last_idx = np.flatnonzero(np.r_[new_bucket[1:], True])
    count = np.diff(np.r_[first_idx, n])
    n_sig = np.add.reduceat(np.abs(sig).astype(np.int64), first_idx)

    return {
        "ts": ts_out[last_idx],
        "open": o_out[last_idx],
        "high": h_out[last_idx],
        "low": l_out[last_idx],
        "close": c_out[last_idx],
        "volume": v_out[last_idx],
        "count": count,
        "sig": sig[last_idx],
        "n_sig": n_sig,
    }


def bundle_per_bar(sig, up, dw, ts_arr, o_arr, h_arr, l_arr, c_arr, v_arr) -> dict:
    """把内核填充的 per-bar 数组打包为 framework 通用契约 dict

    返回:
      {"sig": per-bar 信号轨迹 (int8),
       "per_bar": {"up": ..., "dw": ..., "ts": ..., "o": ..., "h": ..., "l": ..., "c": ..., "v": ...}}

    框架约定 per-bar 子字典的键名 (kernel 输出通道), 用于让调用方 (CLI /
    钩子) 不必逐项命名; 策略在自己 hook 内按需取值。框架 CLI 不出现
    up/dw 等指标键作为函数关键字参数。
    """
    return {
        "sig": sig,
        "per_bar": {
            "up": up, "dw": dw,
            "ts": ts_arr, "o": o_arr, "h": h_arr,
            "l": l_arr, "c": c_arr, "v": v_arr,
        },
    }


# ============ Python 侧工具 (状态 -> 结果) ============

def summarize(st: KernelState) -> dict:
    """终态 -> 绩效字典 (口径与 Engine.print_summary 一致; 其余为选参新增, 见 kbs/13)

    超额口径定义: x_t = (策略权益 - 基线权益)/基线权益, 基线 = 期初资金 + 期初持仓×现价。
      years           策略期年数 (首末策略期 bar 的自然日差 / 365.25)
      ann_excess_pct  年化超额% (未扣费; 扣费在 sweep 层: turnover×费率)
      sharpe_excess   超额 Sharpe = mean(日Δx)/std(日Δx) × √252 (只衡量择时贡献的平稳性)
      x_mdd           超额率曲线最大回撤 (择时懊悔深度; 与权益 max_drawdown 分工不同)

    业界通用绩效口径 (新增, 2026-09-05):
      cagr            年化复合收益率% = (终值/初值)^(1/年) - 1 (绝对, 含底仓 β)
      sortino_excess  超额 Sortino = mean(日Δx) / std(下行日Δx) × √252 (只罚下行, 更稳健)
      calmar          ann_excess_pct / max_drawdown (年化收益承受多少回撤; 0 表示无回撤或负)
      max_dd_days     最大回撤持续天数 (peak_ts 到 valley_ts; 未恢复则记到期末; 实盘心态指标)
      max_dd_recovered 回撤是否恢复 (False = 当前仍处历史最深回撤谷底)
    """
    baseline = st.init_cash + st.init_position * st.last_price
    equity = st.cash + st.position * st.last_price
    init_eq = st.init_equity if st.init_equity > 0.0 else (st.init_cash + st.init_position * st.last_price)
    diff = equity - baseline
    pct = (diff / baseline * 100) if baseline else 0

    years = 0.0
    if st.first_ts_set and st.last_ts > st.first_ts:
        years = ((encoded_to_epoch(st.last_ts) - encoded_to_epoch(st.first_ts))
                 / (365.25 * 86400.0))
    ann_excess_pct = (pct / years) if years > 0 else 0.0

    sharpe_excess = 0.0
    if st.d_n >= 2:
        mean_d = st.d_sum / st.d_n
        var_d = st.d_sum2 / st.d_n - mean_d * mean_d
        if var_d > 0.0:
            sharpe_excess = mean_d / var_d ** 0.5 * 252.0 ** 0.5

    sortino_excess = 0.0
    if st.d_neg_n >= 1 and st.d_n >= 1:
        mean_d = st.d_sum / st.d_n
        # 下行二阶距 (sum d^2, d<0) -> 下行波动
        dn_var = st.d_neg_sum2 / st.d_neg_n
        if dn_var > 0.0:
            sortino_excess = mean_d / dn_var ** 0.5 * 252.0 ** 0.5

    # CAGR: 终值/期初权益, 仅在 init_eq>0 且 years>0 时有意义
    cagr = 0.0
    if init_eq > 0.0 and years > 0.0:
        ratio = equity / init_eq
        if ratio > 0.0:
            cagr = (ratio ** (1.0 / years) - 1.0) * 100.0

    # Calmar: ann_excess_pct / max_drawdown (无回撤 -> 0 避免除零; 回撤>100% 不常见, 截 0)
    calmar = 0.0
    if st.max_drawdown > 1e-9:
        calmar = ann_excess_pct / (st.max_drawdown * 100.0)

    # 最大回撤持续天数 (peak_ts 到 valley_ts; 未恢复用最后 ts)
    max_dd_days = 0.0
    if st.peak_eq_ts > 0 and st.valley_eq_ts > 0 and st.valley_eq_ts >= st.peak_eq_ts:
        end_ts = st.last_ts if not st.recovered else st.valley_eq_ts
        if end_ts >= st.peak_eq_ts:
            max_dd_days = (encoded_to_epoch(end_ts) - encoded_to_epoch(st.peak_eq_ts)) / 86400.0

    return {
        "final_price": st.last_price,
        "n_trades": int(st.n_trades),
        "n_buy": int(st.n_buy),
        "n_sell": int(st.n_sell),
        "final_cash": st.cash,
        "final_position": st.position,
        "final_equity": equity,
        "baseline": baseline,
        "excess": diff,
        "excess_pct": pct,
        "years": years,
        "ann_excess_pct": ann_excess_pct,
        "sharpe_excess": sharpe_excess,
        "sortino_excess": sortino_excess,
        "cagr": cagr,
        "calmar": calmar,
        "max_dd_days": max_dd_days,
        "max_dd_recovered": bool(st.recovered),
        "x_mdd": st.x_mdd,
        "max_drawdown": st.max_drawdown,
        "turnover": st.turnover,
    }


def trades_to_list(st: KernelState) -> list[dict]:
    """成交记录数组 -> list[dict] (与 Account.trades 字段对应)"""
    out = []
    for i in range(min(st.n_trades, st.trade_ts.shape[0])):
        out.append({
            "ts": int(st.trade_ts[i]),
            "side": "BUY" if st.trade_side[i] == 1 else "SELL",
            "qty": float(st.trade_qty_a[i]),
            "price": float(st.trade_price[i]),
            "cash_after": float(st.trade_cash_after[i]),
        })
    return out


def bars_to_arrays(bars) -> dict:
    """list[Bar] -> 内核输入数组 (stime 转 14 位整数)"""
    n = len(bars)
    stime = np.empty(n, np.int64)
    o = np.empty(n, np.float64)
    h = np.empty(n, np.float64)
    l = np.empty(n, np.float64)
    c = np.empty(n, np.float64)
    v = np.empty(n, np.float64)
    for i, b in enumerate(bars):
        stime[i] = int(b.stime)
        o[i] = b.open
        h[i] = b.high
        l[i] = b.low
        c[i] = b.close
        v[i] = b.volume
    return {"stime": stime, "open": o, "high": h, "low": l, "close": c, "volume": v}
