from __future__ import annotations
"""GPU 参数扫描 (CUDA 通用内核) 与环境探测

================================================================
本模块的 CUDA sweep_kernel_generic 与 evtrade.kernel.step 是**等价**的两个实现
(tests/test_dsl_cuda.py::test_generic_cuda_matches_cpu_channel_deviation 锁定,
--fmad=false 强制禁止 FMA 合并)。所有 DSL 策略统一走通用模板;
策略段由 render_cuda_device_function 编译期注入 {STRATEGY_BODY}。

新增 GPU 输出数组的步骤:
  1. _CUDA_SOURCE_GENERIC_TEMPLATE 函数签名末尾加指针参数
  2. kernel body 末尾写 out_xxx[tid]
  3. cuda_sweep_window_generic() 中加 cp.empty + 传参 + .get() 拉回 host
  4. Python 侧按 kernel.summarize() 同式汇总新指标
================================================================

与 kernel.py 的分工:
  * kernel.step (numba)  = 流式决策内核: 回测/实盘/扫描共用, 是语义的"唯一权威"。
  * 本模块的 CUDA kernel = 同一步进语义的 GPU 移植 (一行对一行), 只服务参数扫描。
    两者由 tests/test_dsl_cuda.py 差分锁定 (float64 + --fmad=false, 逐位一致)。

关键优化 —— "相同的东西只算一次":
  桶时间戳 ts[i] 与预热标记 mark[i] 只依赖 (周期, 预热阈值), 与策略参数无关,
  故按周期分组**预计算一次**全组共享 (numpy 向量化整数历法), 内核循环里不再有
  int64 除法 (GPU 上 64 位除法是软件模拟, 慢 1~2 个数量级)。

移植约定 (与 evtrade/kernel.py 逐行对应):
  * 一线程 = 一组参数; 线程内顺序扫 bar (流式语义), bar 数组全部线程广播读。
  * 状态全部放寄存器; 不落逐笔成交 (需要明细时用 CPU 内核复跑该组参数)。
  * 浮点表达式与 kernel.py 完全同式; NVRTC 关闭 FMA 合并 (--fmad=false),
    避免 a*b+c 融合改变舍入导致阈值比较漂移。
  * 输出 = 每组参数的终态标量; equity/baseline/excess 由 Python 侧按
    kernel.summarize 同式计算。

历史: 旧 _CUDA_SOURCE (channel_deviation 专用冻结模板, 参数写死 low1s/high1s
数组) 已移除, channel_deviation 一并并入通用模板 (回撤时间戳口径随之从"桶 ts"
修正为"1m bar stime", 与 CPU kernel.step 对齐; max_dd_days 不参与对账)。

本机验证 (2026-09-05): RTX 5090 (sm_120, Blackwell) + 驱动 CUDA 13.1, 无本地
CUDA toolkit; 经 pip 的 nvidia-cuda-*-cu12 轮子提供 DLL, NVRTC 编译 PTX 由驱动 JIT。
"""

import os
import shutil
import subprocess
from collections import OrderedDict

import numpy as np

from .config import INIT_CASH, INIT_POSITION
from .kernel import encoded_to_epoch, resolve_period_seconds
from .kernel_dsl import _invalidate_cache, _source_hash as _source_hash_gpu

# 旧的 _CUDA_SOURCE (channel_deviation 专用冻结模板) 已移除:
# 所有 DSL 策略 (含 channel_deviation) 统一走下方 _CUDA_SOURCE_GENERIC_TEMPLATE
# (cuda_sweep_window_generic)。

# DSL → CUDA 投影 (state_spec / params_spec → CUDA 字符串) 在 strategies/dsl.py:
#   build_cuda_state_decls       - state_spec 字段的寄存器声明
#   build_cuda_strategy_check_call - strategy_check(...) 调用处的 state arg 列表
#   build_cuda_device_header     - __device__ 函数签名头
# core/ 不再持有策略投影逻辑, 只剩 CUDA kernel 源码 + 编译 + 调度基础设施。

# ============ 通用 CUDA kernel: 策略段由 DSL 注入 (步骤 1/2) ============
# 接受 N 个 double 参数 (params 数组); 策略段是 render_cuda_device_function
# 生成的 __device__ int strategy_check(...) 整函数, 编译期注入 {STRATEGY_BODY}。
# device 函数形态的意义: DSL 里的 return X 直接成为函数返回,
# "提前返回跳过后续语句" 的语义与 Python/numba 端完全一致 (内联块做不到)。
# 设计要点:
#   - params 用 params_arr 索引 (p0..p7), 不写死策略参数名 → 任意 DSL 策略通用
#   - state_spec 字段由策略类声明; 本模板在编译期按 strategy_name 注入对应
#     寄存器声明 ({STATE_DECLS}) 与 strategy_check 调用参数 ({STRATEGY_STATE_ARGS})
#   - 桶切换重置锁用 state_spec 字段的"时间戳类"字段 != cur_ts 判断 (与 DSL 语义同式)
#   - 回撤时间戳用 1m bar stime (与 CPU kernel.step 同口径)
_CUDA_SOURCE_GENERIC_TEMPLATE = r"""
// ==== 策略段: DSL 渲染的 __device__ 函数 (编译期注入, 见 strategies/dsl.py) ====
{STRATEGY_BODY}
// ============================================================================

extern "C" __global__ void sweep_kernel_generic(
    const long long* __restrict__ ts_arr,
    const signed char* __restrict__ mark_arr,
    const int* __restrict__ day_id,
    const long long* __restrict__ stime_arr,
    const double* __restrict__ o, const double* __restrict__ h,
    const double* __restrict__ l, const double* __restrict__ c,
    const double* __restrict__ v,
    long long n, int num_combos,
    double init_cash, double init_position, double trade_qty,
    double buy_pct, double sell_pct,
    const double* __restrict__ scales,
    const long long* __restrict__ tf1s,
    const double* __restrict__ params_arr,    // [num_combos * num_params]
    int num_params,
    double* out_cash, double* out_pos, double* out_last,
    long long* out_ntrades, long long* out_nbuy, long long* out_nsell,
    double* out_mdd, double* out_turnover,
    double* out_dsum, double* out_dsum2, long long* out_dn, double* out_xmdd,
    double* out_dneg_sum2, long long* out_dneg_n,
    double* out_init_equity,
    long long* out_peak_ts, long long* out_valley_ts, int* out_recovered)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_combos) return;

    long long tf1 = tf1s[tid];
    double scale = scales[tid];
    double k_ema = 2.0 / ((double)tf1 + 1.0);
    double nan_bits = __longlong_as_double((long long)0x7ff8000000000000ULL);

    // ---- 状态 (寄存器), 与 KernelState 字段一一对应 ----
    int has_cur = 0; long long cur_ts = 0;
    double cur_open = 0.0, cur_high = 0.0, cur_low = 0.0, cur_close = 0.0, cur_vol = 0.0;
    long long cur_count = 0, cur_mark = 1;
    double up_sum = 0.0, dw_sum = 0.0, up_ema = nan_bits, dw_ema = nan_bits;
    long long up_count = 0, dw_count = 0;
    // ---- 策略 state_spec 字段 (按 strategy_name 的 state_spec 注入) ----
{STATE_DECLS}
    double cash = init_cash, position = init_position, last_price = 0.0;
    long long n_trades = 0, n_buy = 0, n_sell = 0;
    double turnover = 0.0, peak = 0.0, mdd = 0.0;
    long long peak_ts = 0, valley_ts = 0; int recovered = 1;
    long long last_side = 0; double cur_qty = trade_qty;   // 倍投状态
    // 超额曲线 (与 kernel.step 同式)
    int day_init = 0, day_end_init = 0, cur_day = -1;
    double x_day_end = 0.0, x_day_end_prev = nan_bits;
    double d_sum = 0.0, d_sum2 = 0.0, x_peak = -1.0e18, x_mdd = 0.0;
    long long d_n = 0;
    double d_neg_sum2 = 0.0; long long d_neg_n = 0;
    double init_equity = 0.0;

    // ---- 从 params_arr 索引 (num_params 个 double, 按策略 params_spec 顺序) ----
    double p0 = num_params > 0 ? params_arr[tid * num_params + 0] : 0.0;
    double p1 = num_params > 1 ? params_arr[tid * num_params + 1] : 0.0;
    double p2 = num_params > 2 ? params_arr[tid * num_params + 2] : 0.0;
    double p3 = num_params > 3 ? params_arr[tid * num_params + 3] : 0.0;
    double p4 = num_params > 4 ? params_arr[tid * num_params + 4] : 0.0;
    double p5 = num_params > 5 ? params_arr[tid * num_params + 5] : 0.0;
    double p6 = num_params > 6 ? params_arr[tid * num_params + 6] : 0.0;
    double p7 = num_params > 7 ? params_arr[tid * num_params + 7] : 0.0;

    for (long long i = 0; i < n; ++i) {
        long long ts = ts_arr[i];
        long long mark = (long long)mark_arr[i];

        // ---- 桶切换: 闭合旧桶 -> push EMA ----
        if (has_cur && ts != cur_ts) {
            if (up_count < tf1) {
                up_sum += cur_high; up_count++;
                if (up_count == tf1) up_ema = up_sum / (double)tf1;
            } else {
                up_ema = cur_high * k_ema + up_ema * (1.0 - k_ema); up_count++;
            }
            if (dw_count < tf1) {
                dw_sum += cur_low; dw_count++;
                if (dw_count == tf1) dw_ema = dw_sum / (double)tf1;
            } else {
                dw_ema = cur_low * k_ema + dw_ema * (1.0 - k_ema); dw_count++;
            }
            has_cur = 0;
        }
        if (!has_cur) {
            has_cur = 1; cur_ts = ts; cur_open = o[i]; cur_high = h[i]; cur_low = l[i];
            cur_close = c[i]; cur_vol = v[i]; cur_count = 1; cur_mark = mark;
        } else {
            if (h[i] > cur_high) cur_high = h[i];
            if (l[i] < cur_low) cur_low = l[i];
            cur_close = c[i]; cur_vol += v[i]; cur_count++; cur_mark = mark;
        }

        // ---- 预热桶: 只累积指标 ----
        if (cur_mark == 0) continue;

        double price = cur_close;
        last_price = price;

        // ---- 通道值 (与 kernel.py 同式) ----
        double up, dw;
        if (up_count < tf1 - 1) up = nan_bits;
        else if (up_count == tf1 - 1) up = (up_sum + cur_high) / (double)tf1;
        else up = cur_high * k_ema + up_ema * (1.0 - k_ema);
        if (dw_count < tf1 - 1) dw = nan_bits;
        else if (dw_count == tf1 - 1) dw = (dw_sum + cur_low) / (double)tf1;
        else dw = cur_low * k_ema + dw_ema * (1.0 - k_ema);

        // ============ 策略段 (DSL device 函数; return 语义与 DSL 一致) ============
        int signal = strategy_check({STRATEGY_STATE_ARGS},
                                    cur_ts, cur_open, cur_high, cur_low,
                                    cur_close, cur_vol, up, dw,
                                    p0, p1, p2, p3, p4, p5, p6, p7);
        // =========================================================================

        // ---- 模拟成交 (与 kernel.py 同式, 含倍投与比例模式) ----
        if (signal != 0) {
            if (signal == last_side) {
                cur_qty = cur_qty * scale;
            } else {
                cur_qty = trade_qty; last_side = signal;
            }
            if (signal == 1) {
                double q;
                if (price > 0.0) {
                    double max_by_cash = cash / price;
                    if (buy_pct > 0.0) {
                        double target = buy_pct * max_by_cash;
                        q = (target < max_by_cash) ? target : max_by_cash;
                    } else {
                        q = (cur_qty < max_by_cash) ? cur_qty : max_by_cash;
                    }
                } else { q = 0.0; }
                if (q > 0.0) {
                    cash -= q * price; position += q;
                    n_trades++; n_buy++; turnover += q * price;
                }
            } else if (signal == -1) {
                double q;
                if (sell_pct > 0.0) {
                    double target = sell_pct * position;
                    q = (target < position) ? target : position;
                } else {
                    q = (cur_qty < position) ? cur_qty : position;
                }
                if (q > 0.0) {
                    cash += q * price; position -= q;
                    n_trades++; n_sell++; turnover += q * price;
                }
            }
        }

        // ---- 权益回撤 + 超额曲线 (与 kernel.py 同式; 回撤时间戳用 1m bar
        //      stime, 与 CPU kernel.step 的 peak_eq_ts/valley_eq_ts 同口径) ----
        double eq = cash + position * last_price;
        long long stime_i = stime_arr[i];
        if (eq > peak) { peak = eq; peak_ts = stime_i; recovered = 1; }
        if (peak > 0.0) {
            double dd = (peak - eq) / peak;
            if (dd > mdd) { mdd = dd; valley_ts = stime_i; recovered = 0; }
        }
        if (init_equity == 0.0) init_equity = init_cash + init_position * last_price;
        double base_eq = init_cash + init_position * last_price;
        double x = (eq - base_eq) / base_eq;
        if (x > x_peak) x_peak = x;
        double xdd = x_peak - x;
        if (xdd > x_mdd) x_mdd = xdd;
        int day = day_id[i];
        if (!day_init) { day_init = 1; cur_day = day; }
        else if (day != cur_day) {
            if (day_end_init) {
                double d = x_day_end - x_day_end_prev;
                d_sum += d; d_sum2 += d * d; d_n++;
                if (d < 0.0) { d_neg_sum2 += d * d; d_neg_n++; }
            }
            x_day_end_prev = x_day_end;
            day_end_init = 1; cur_day = day;
        }
        x_day_end = x;
    }

    out_cash[tid] = cash; out_pos[tid] = position; out_last[tid] = last_price;
    out_ntrades[tid] = n_trades; out_nbuy[tid] = n_buy; out_nsell[tid] = n_sell;
    out_mdd[tid] = mdd; out_turnover[tid] = turnover;
    out_dsum[tid] = d_sum; out_dsum2[tid] = d_sum2; out_dn[tid] = d_n;
    out_xmdd[tid] = x_mdd;
    out_dneg_sum2[tid] = d_neg_sum2; out_dneg_n[tid] = d_neg_n;
    out_init_equity[tid] = init_equity;
    out_peak_ts[tid] = peak_ts; out_valley_ts[tid] = valley_ts; out_recovered[tid] = recovered;
}
"""

_cp = None

# GPU 缓存键: (strategy_name, source_hash, compute_capability, RENDERER_VERSION)
# 三元 key 让 DSL docstring 改动 / 渲染器版本升级 / 换 GPU 算力时自动失效旧缓存。
# 与 core/kernel_dsl.py 同源思路 (避免旧 GPU kernel 在运行时被静默复用)。
_GENERIC_KERNEL_CACHE: dict = {}
GPU_RENDERER_VERSION = "v1"


def invalidate_gpu_cache(strategy_name: str | None = None) -> int:
    """清除 GPU kernel 缓存; strategy_name=None 时清空全部

    返回清除的条目数, 供测试断言与 dev reload 工具用。
    复用 kernel_dsl._invalidate_cache (与 DSL 缓存失效同逻辑)。
    """
    return _invalidate_cache(_GENERIC_KERNEL_CACHE, strategy_name)


def _compile_generic_kernel(strategy_name: str):
    """编译通用 CUDA kernel (步骤 1/2)

    策略段 = strategies.dsl.render_cuda_device_function 生成的
    __device__ int strategy_check(...) 整函数 (ctx 字段按内核状态契约映射,
    局部变量自动声明), 编译结果缓存到 _GENERIC_KERNEL_CACHE。
    """
    from ..strategies import get_strategy_class
    from ..strategies.dsl import (
        CompileError,
        build_cuda_state_decls,
        build_cuda_strategy_check_call,
        render_cuda_device_function,
    )
    cls = get_strategy_class(strategy_name)
    if not (hasattr(cls, "compute_signal") and cls.compute_signal.__doc__):
        # 纯 Python 策略: 没 DSL docstring; CUDA 不可用
        raise ValueError(
            f"策略 {strategy_name!r} 没有 compute_signal DSL docstring; "
            f"GPU 扫描只支持 DSL 路径 (在 compute_signal 写 docstring)")
    cp = _ensure_cupy()
    cc_raw = cp.cuda.device.get_compute_capability()
    cc = f"{cc_raw[0]}{cc_raw[1]}" if isinstance(cc_raw, (tuple, list)) else "default"
    src_hash = _source_hash_gpu(cls)
    cache_key = (strategy_name, src_hash, cc, GPU_RENDERER_VERSION)
    cached = _GENERIC_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        strategy_func = render_cuda_device_function(cls)
    except CompileError as e:
        raise ValueError(f"策略 {strategy_name!r} 无法渲染到 CUDA: {e}") from e
    src = (_CUDA_SOURCE_GENERIC_TEMPLATE
           .replace("{STRATEGY_BODY}", strategy_func)
           .replace("{STATE_DECLS}", build_cuda_state_decls(cls))
           .replace("{STRATEGY_STATE_ARGS}", build_cuda_strategy_check_call(cls)))
    if isinstance(cc_raw, (tuple, list)):
        archs = [f"compute_{cc_raw[0]}{cc_raw[1]}", "compute_90"]
    else:
        archs = ["compute_90"]
    last_err = None
    for arch in archs:
        try:
            kern = cp.RawKernel(src, "sweep_kernel_generic",
                                options=(f"--gpu-architecture={arch}", "--fmad=false"))
            _GENERIC_KERNEL_CACHE[cache_key] = kern
            return kern
        except Exception as e:
            last_err = e
    raise last_err


def _ensure_cupy():
    """导入 cupy (pip 的 nvidia-*-cu12 轮子提供 DLL, 先补进进程 PATH)"""
    global _cp
    if _cp is not None:
        return _cp
    import site
    dirs = []
    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    for sp in roots:
        nv = os.path.join(sp, "nvidia")
        if os.path.isdir(nv):
            for name in os.listdir(nv):
                p = os.path.join(nv, name, "bin")
                if os.path.isdir(p):
                    dirs.append(p)
    if dirs:
        os.environ["PATH"] = ";".join(dirs + [os.environ.get("PATH", "")])
        for d in dirs:
            os.add_dll_directory(d)
    import cupy as cp
    _cp = cp
    return cp



# ============ ts / mark 预计算 (numpy 向量化整数历法, 与 kernel.py 同式) ============

def _encoded_to_epoch_np(t: np.ndarray) -> np.ndarray:
    y = t // 10_000_000_000
    mo = (t // 100_000_000) % 100
    d = (t // 1_000_000) % 100
    h = (t // 10_000) % 100
    mi = (t // 100) % 100
    s = t % 100
    yy = y - (mo <= 2)
    era = yy // 400
    yoe = yy - era * 400
    mp = (mo + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return (era * 146097 + doe - 719468) * 86400 + h * 3600 + mi * 60 + s


def _epoch_to_encoded_np(e: np.ndarray) -> np.ndarray:
    days = e // 86400
    sod = e % 86400
    h = sod // 3600
    mi = (sod % 3600) // 60
    s = sod % 60
    z = days + 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + np.where(mp < 10, 3, -9)
    y2 = y + (m <= 2)
    return ((((y2 * 100 + m) * 100 + d) * 100 + h) * 100 + mi) * 100 + s


# precompute_ts_mark 的 LRU 缓存: key=(id(bars), len(stime), period, warmup)
# 同 (bars, period, warmup_until) 重复调用直接返回缓存结果, 省掉 epoch
# 转换与桶对齐的开销 (WFO 多窗口扫描时尤其显著)。
# 注意: id(bars) 在 dict 复用场景下不严格; 用 (id, len) 作为组合键,
# 且缓存容量有界防止内存泄漏。
_PRECOMPUTE_TS_MARK_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_PRECOMPUTE_TS_MARK_MAXSIZE = 32


def _precompute_cache_key(bars: dict, period: str, warmup_until: int):
    """缓存键: 用 id+bars 长度避免 dict 复用错命中"""
    return (id(bars), len(bars.get("stime", ())), period, int(warmup_until))


def precompute_ts_mark(bars: dict, period: str, warmup_until: int):
    """(周期, 预热阈值) -> (ts int64[n], mark int8[n]); 与策略参数无关, 每组共享

    桶算法与 kernel.bucket_ts_encoded 同式 (本地锚定 epoch 取整, 任意 m/h/d 周期)。

    缓存: 同一 (id(bars), len(stime), period, warmup_until) 重复调用直接
    返回缓存结果; WFO 多窗口扫描时省掉 epoch 转换与桶对齐的开销。
    """
    key = _precompute_cache_key(bars, period, warmup_until)
    cached = _PRECOMPUTE_TS_MARK_CACHE.get(key)
    if cached is not None:
        # LRU 触尾
        _PRECOMPUTE_TS_MARK_CACHE.move_to_end(key)
        return cached
    stime = bars["stime"]
    P = resolve_period_seconds(period)
    e = _encoded_to_epoch_np(stime)
    e0 = (e // P) * P
    r = e - e0
    ts = np.where(r == 0,
                  _epoch_to_encoded_np(e0),
                  _epoch_to_encoded_np(e0 + P))
    mark = np.where(stime < warmup_until, 0, 1).astype(np.int8)
    out = (ts.astype(np.int64), mark)
    _PRECOMPUTE_TS_MARK_CACHE[key] = out
    # 容量控制
    while len(_PRECOMPUTE_TS_MARK_CACHE) > _PRECOMPUTE_TS_MARK_MAXSIZE:
        _PRECOMPUTE_TS_MARK_CACHE.popitem(last=False)
    return out


def invalidate_precompute_cache() -> int:
    """清除 precompute_ts_mark 缓存; 返回清除的条目数"""
    n = len(_PRECOMPUTE_TS_MARK_CACHE)
    _PRECOMPUTE_TS_MARK_CACHE.clear()
    return n


def gpu_info() -> dict:
    """探测 GPU/CUDA 环境 (任何缺失只记 None, 不抛异常)"""
    info = {"nvidia_gpu": None, "driver": None, "cuda_toolkit": None,
            "cupy": None, "numba_cuda": None}
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                                  "--format=csv,noheader"], capture_output=True,
                                 text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                name, _, drv = out.stdout.strip().partition(", ")
                info["nvidia_gpu"] = name or None
                info["driver"] = drv or None
        except Exception:
            pass
    info["cuda_toolkit"] = shutil.which("nvcc")
    try:
        cp = _ensure_cupy()
        info["cupy"] = f"{cp.__version__} @ {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
    except Exception:
        pass
    try:
        from numba import cuda
        info["numba_cuda"] = bool(cuda.is_available())
    except Exception:
        pass
    return info


def _alloc_gpu_outputs(m: int):
    """分配一组 GPU 输出数组 (cuda_sweep_window_generic 共用, 17 个 cp.empty)。

    返回 dict, key 与 _collect_gpu_results 形参同名。两处原先各写一份完全相同的
    cp.empty 序列; 提取后避免新增输出列时漏改其一。
    """
    return {
        "out_cash": cp.empty(m, cp.float64),
        "out_pos": cp.empty(m, cp.float64),
        "out_last": cp.empty(m, cp.float64),
        "out_ntrades": cp.empty(m, cp.int64),
        "out_nbuy": cp.empty(m, cp.int64),
        "out_nsell": cp.empty(m, cp.int64),
        "out_mdd": cp.empty(m, cp.float64),
        "out_turnover": cp.empty(m, cp.float64),
        "out_dsum": cp.empty(m, cp.float64),
        "out_dsum2": cp.empty(m, cp.float64),
        "out_dn": cp.empty(m, cp.int64),
        "out_xmdd": cp.empty(m, cp.float64),
        "out_dneg_sum2": cp.empty(m, cp.float64),
        "out_dneg_n": cp.empty(m, cp.int64),
        "out_init_equity": cp.empty(m, cp.float64),
        "out_peak_ts": cp.empty(m, cp.int64),
        "out_valley_ts": cp.empty(m, cp.int64),
        "out_recovered": cp.empty(m, cp.int32),
    }


def _collect_gpu_results(bars, params_list, idxs, warmup_until,
                         cash, pos, last, ntr, nbuy, nsell, mdd, turnover,
                         dsum, dsum2, dn, xmdd, dneg_sum2, dneg_n, init_eq,
                         peak_ts, valley_ts, recovered) -> dict:
    """GPU kernel 输出 host 数组 -> {param_idx: 绩效字典}

    cuda_sweep_window_generic 共用此汇总逻辑,
    避免 sharpe/sortino/cagr/calmar/max_dd_days 公式在两处复制
    (公式与 kernel.summarize 同口径; 任一改动需同步 kernel.summarize)。
    """
    stime = bars["stime"]
    years = 0.0
    idx0 = int(np.searchsorted(stime, int(warmup_until)))
    if idx0 < len(stime) and stime[-1] > stime[idx0]:
        years = ((encoded_to_epoch(int(stime[-1])) - encoded_to_epoch(int(stime[idx0])))
                 / (365.25 * 86400.0))
    out: dict = {}
    for j, i in enumerate(idxs):
        p = params_list[i]
        init_cash = float(p.get("init_cash", INIT_CASH))
        init_position = float(p.get("init_position", INIT_POSITION))
        baseline = init_cash + init_position * float(last[j])
        equity = float(cash[j]) + float(pos[j]) * float(last[j])
        diff = equity - baseline
        pct = (diff / baseline * 100.0) if baseline else 0.0
        ann = (pct / years) if years > 0 else 0.0
        sharpe = 0.0
        if dn[j] >= 2:
            mean_d = dsum[j] / dn[j]
            var_d = dsum2[j] / dn[j] - mean_d * mean_d
            if var_d > 0.0:
                sharpe = mean_d / var_d ** 0.5 * 252.0 ** 0.5
        # Sortino: 仅下行 (d<0)
        sortino = 0.0
        if dneg_n[j] >= 1 and dn[j] >= 1:
            mean_d = dsum[j] / dn[j]
            dn_var = dneg_sum2[j] / dneg_n[j]
            if dn_var > 0.0:
                sortino = mean_d / dn_var ** 0.5 * 252.0 ** 0.5
        # CAGR
        ieq = float(init_eq[j]) if init_eq[j] > 0.0 else (
            init_cash + init_position * float(last[j]))
        cagr = 0.0
        if ieq > 0.0 and years > 0.0:
            ratio = equity / ieq
            if ratio > 0.0:
                cagr = (ratio ** (1.0 / years) - 1.0) * 100.0
        # Calmar
        calmar = 0.0
        if mdd[j] > 1e-9:
            calmar = ann / (float(mdd[j]) * 100.0)
        # 最大回撤持续天数
        dd_days = 0.0
        if int(peak_ts[j]) > 0 and int(valley_ts[j]) >= int(peak_ts[j]):
            end_ts = int(stime[-1]) if int(recovered[j]) == 0 else int(valley_ts[j])
            if end_ts >= int(peak_ts[j]):
                dd_days = (encoded_to_epoch(end_ts) - encoded_to_epoch(int(peak_ts[j]))) / 86400.0
        out[i] = {
            "final_price": float(last[j]),
            "n_trades": int(ntr[j]),
            "n_buy": int(nbuy[j]),
            "n_sell": int(nsell[j]),
            "final_cash": float(cash[j]),
            "final_position": float(pos[j]),
            "final_equity": equity,
            "baseline": baseline,
            "excess": diff,
            "excess_pct": pct,
            "years": years,
            "ann_excess_pct": ann,
            "sharpe_excess": sharpe,
            "sortino_excess": sortino,
            "cagr": cagr,
            "calmar": calmar,
            "max_dd_days": dd_days,
            "max_dd_recovered": bool(int(recovered[j])),
            "x_mdd": float(xmdd[j]),
            "max_drawdown": float(mdd[j]),
            "turnover": float(turnover[j]),
        }
    return out


# ============ 通用 CUDA sweep kernel (步骤 1) ============
# 接受任意 strategies/ 子包策略 (需 DSL compute_signal docstring)
# ctx 字段约定: p0..p7 是策略参数 (按 params_spec 顺序); 其余字段与 kernel.py 同式

def cuda_sweep_window_generic(bars: dict, params_list: list[dict],
                                warmup_until: int,
                                strategy_name: str = None) -> list[dict]:
    """GPU 批量回测 (任意 DSL 策略; 通用 kernel)

    params 从 params["params"] dict 取 (按策略 params_spec 声明顺序映射到内核
    p0..p7); 策略段由 DSL 注入 (render_cuda_device_function, 编译期 {STRATEGY_BODY})。
    strategy_name: 策略 key; 缺省时取 params_list[0]["strategy_name"]。
    """
    from ..strategies import get_strategy_param_spec

    if not params_list:
        return []

    cp = _ensure_cupy()
    if not strategy_name:
        strategy_name = params_list[0].get("strategy_name")
    if not strategy_name:
        raise ValueError("cuda_sweep_window_generic: 缺 strategy_name")

    kernel = _compile_generic_kernel(strategy_name)

    # 提取策略参数 spec 顺序
    spec = get_strategy_param_spec(strategy_name)
    param_keys = list(spec.keys())  # strategy params in spec order
    if len(param_keys) > 8:
        raise ValueError(f"策略 {strategy_name} 参数 > 8 个, CUDA kernel 通用模板不支持")

    # 按周期分组
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(params_list):
        groups.setdefault(p.get("period", "5m"), []).append(i)

    d_o = cp.asarray(bars["open"])
    d_h = cp.asarray(bars["high"])
    d_l = cp.asarray(bars["low"])
    d_c = cp.asarray(bars["close"])
    d_v = cp.asarray(bars["volume"])
    day = bars["stime"] // 1_000_000
    _, day_inv = np.unique(day, return_inverse=True)
    d_day = cp.asarray(day_inv.astype(np.int32))
    d_stime = cp.asarray(bars["stime"])          # 回撤时间戳 (与 CPU 同口径)
    n = cp.int64(len(bars["stime"]))

    results: list = [None] * len(params_list)
    for period, idxs in groups.items():
        m = len(idxs)
        ts_np, mark_np = precompute_ts_mark(bars, period, int(warmup_until))
        d_ts = cp.asarray(ts_np)
        d_mark = cp.asarray(mark_np)

        # 构建 params 矩阵 [num_combos * num_params]
        params_mat = np.empty((m, len(param_keys)), dtype=np.float64)
        scales = np.empty(m, dtype=np.float64)
        tf1s = np.empty(m, dtype=np.int64)
        for j, i in enumerate(idxs):
            p = params_list[i]
            sp = p.get("params", {})
            for k_idx, k in enumerate(param_keys):
                params_mat[j, k_idx] = float(sp.get(k, spec[k].get("default", 0.0)))
            scales[j] = float(p.get("scale", 1.0))
            tf1s[j] = int(p.get("tf1", sp.get("tf1", 21)))

        d_params = cp.asarray(params_mat.flatten())  # 行优先
        d_scales = cp.asarray(scales)
        d_tf1s = cp.asarray(tf1s)

        # 输出数组
        _out = _alloc_gpu_outputs(m)
        out_cash = _out["out_cash"]; out_pos = _out["out_pos"]; out_last = _out["out_last"]
        out_ntrades = _out["out_ntrades"]; out_nbuy = _out["out_nbuy"]; out_nsell = _out["out_nsell"]
        out_mdd = _out["out_mdd"]; out_turnover = _out["out_turnover"]
        out_dsum = _out["out_dsum"]; out_dsum2 = _out["out_dsum2"]; out_dn = _out["out_dn"]
        out_xmdd = _out["out_xmdd"]; out_dneg_sum2 = _out["out_dneg_sum2"]; out_dneg_n = _out["out_dneg_n"]
        out_init_equity = _out["out_init_equity"]
        out_peak_ts = _out["out_peak_ts"]; out_valley_ts = _out["out_valley_ts"]; out_recovered = _out["out_recovered"]

        threads = 256
        blocks = (m + threads - 1) // threads
        # 兼容旧调用: 把 buy_pct/sell_pct/init_cash/init_position/trade_qty 提到首组
        first = params_list[idxs[0]]
        _buy_pct = float(first.get("buy_pct", 0.0))
        _sell_pct = float(first.get("sell_pct", 0.0))
        if first.get("all_in", False):
            _buy_pct = max(_buy_pct, 1.0)
            _sell_pct = max(_sell_pct, 1.0)
        kernel((blocks,), (threads,), (
            d_ts, d_mark, d_day, d_stime, d_o, d_h, d_l, d_c, d_v,
            n, cp.int32(m),
            cp.float64(float(first.get("init_cash", 200000.0))),
            cp.float64(float(first.get("init_position", 200000.0))),
            cp.float64(float(first.get("trade_qty", 10000.0))),
            cp.float64(_buy_pct),
            cp.float64(_sell_pct),
            d_scales, d_tf1s, d_params, cp.int32(len(param_keys)),
            out_cash, out_pos, out_last, out_ntrades, out_nbuy, out_nsell,
            out_mdd, out_turnover,
            out_dsum, out_dsum2, out_dn, out_xmdd,
            out_dneg_sum2, out_dneg_n, out_init_equity,
            out_peak_ts, out_valley_ts, out_recovered,
        ))
        cp.cuda.get_current_stream().synchronize()

        # 拉回 host, 公式集中在 _collect_gpu_results
        cash = out_cash.get(); pos = out_pos.get(); last = out_last.get()
        ntr = out_ntrades.get(); nbuy = out_nbuy.get(); nsell = out_nsell.get()
        mdd = out_mdd.get(); turnover = out_turnover.get()
        dsum = out_dsum.get(); dsum2 = out_dsum2.get(); dn = out_dn.get()
        xmdd = out_xmdd.get()
        dneg_sum2 = out_dneg_sum2.get(); dneg_n = out_dneg_n.get()
        init_eq = out_init_equity.get()
        peak_ts = out_peak_ts.get(); valley_ts = out_valley_ts.get()
        recovered = out_recovered.get()

        # 拉回 host, 公式集中在 _collect_gpu_results (公式集中在 _collect_gpu_results)
        for _i, _m in _collect_gpu_results(
            bars, params_list, idxs, warmup_until,
            cash, pos, last, ntr, nbuy, nsell, mdd, turnover,
            dsum, dsum2, dn, xmdd, dneg_sum2, dneg_n, init_eq,
            peak_ts, valley_ts, recovered).items():
            results[_i] = _m
    return results
