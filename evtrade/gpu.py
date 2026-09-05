from __future__ import annotations
"""GPU 参数扫描 (CUDA 内核) 与环境探测

与 kernel.py 的分工:
  * kernel.step (numba)  = 流式决策内核: 回测/实盘/扫描共用, 是语义的"唯一权威"。
  * 本模块的 CUDA kernel = 同一步进语义的 GPU 移植 (一行对一行), 只服务参数扫描。
    两者由 tests/test_gpu.py 差分锁定 (float64 + --fmad=false, 逐位一致)。

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

本机验证 (2026-09-05): RTX 5090 (sm_120, Blackwell) + 驱动 CUDA 13.1, 无本地
CUDA toolkit; 经 pip 的 nvidia-cuda-*-cu12 轮子提供 DLL, NVRTC 编译 PTX 由驱动 JIT。
"""

import os
import shutil
import subprocess

import numpy as np

from .config import INIT_CASH, INIT_POSITION
from .kernel import resolve_period_seconds

_CUDA_SOURCE = r"""
extern "C" __global__ void sweep_kernel(
    const long long* __restrict__ ts_arr,
    const signed char* __restrict__ mark_arr,
    const int* __restrict__ day_id,
    const double* __restrict__ o, const double* __restrict__ h,
    const double* __restrict__ l, const double* __restrict__ c,
    const double* __restrict__ v,
    long long n, int num_combos,
    double init_cash, double init_position, double trade_qty,
    const double* __restrict__ scales,
    const long long* __restrict__ tf1s,
    const double* __restrict__ low1s, const double* __restrict__ low2s,
    const double* __restrict__ high1s, const double* __restrict__ high2s,
    double* out_cash, double* out_pos, double* out_last,
    long long* out_ntrades, long long* out_nbuy, long long* out_nsell,
    double* out_mdd, double* out_turnover,
    double* out_dsum, double* out_dsum2, long long* out_dn, double* out_xmdd)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_combos) return;

    long long tf1 = tf1s[tid];
    double scale = scales[tid];
    double low1 = low1s[tid], low2 = low2s[tid];
    double high1 = high1s[tid], high2 = high2s[tid];
    double k_ema = 2.0 / ((double)tf1 + 1.0);
    double nan_bits = __longlong_as_double((long long)0x7ff8000000000000ULL);

    // ---- 状态 (寄存器), 与 KernelState 字段一一对应 ----
    int has_cur = 0; long long cur_ts = 0;
    double cur_high = 0.0, cur_low = 0.0, cur_close = 0.0, cur_vol = 0.0;
    long long cur_count = 0, cur_mark = 1;
    double up_sum = 0.0, dw_sum = 0.0, up_ema = nan_bits, dw_ema = nan_bits;
    long long up_count = 0, dw_count = 0;
    int low_hit = 0, high_hit = 0, lock_init = 0, low_acted = 0, high_acted = 0;
    long long lock_ts = 0;
    double cash = init_cash, position = init_position, last_price = 0.0;
    long long n_trades = 0, n_buy = 0, n_sell = 0;
    double turnover = 0.0, peak = 0.0, mdd = 0.0;
    long long last_side = 0; double cur_qty = trade_qty;   // 倍投状态
    // 超额曲线 (与 kernel.step 同式)
    int day_init = 0, day_end_init = 0, cur_day = -1;
    double x_day_end = 0.0, x_day_end_prev = nan_bits;
    double d_sum = 0.0, d_sum2 = 0.0, x_peak = -1.0e18, x_mdd = 0.0;
    long long d_n = 0;

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
            has_cur = 1; cur_ts = ts; cur_high = h[i]; cur_low = l[i];
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

        // ---- 通道值 (含未闭合桶) ----
        double up, dw;
        if (up_count < tf1 - 1) up = nan_bits;
        else if (up_count == tf1 - 1) up = (up_sum + cur_high) / (double)tf1;
        else up = cur_high * k_ema + up_ema * (1.0 - k_ema);
        if (dw_count < tf1 - 1) dw = nan_bits;
        else if (dw_count == tf1 - 1) dw = (dw_sum + cur_low) / (double)tf1;
        else dw = cur_low * k_ema + dw_ema * (1.0 - k_ema);

        // ---- 策略状态机 (kernel._strategy_check 同式) ----
        int signal = 0;
        if (!(up != up) && !(dw != dw) && up != 0.0 && dw != 0.0) {
            if (!lock_init || cur_ts != lock_ts) {
                lock_ts = cur_ts; lock_init = 1;
                low_acted = 0; high_acted = 0;
            }
            double low_dev = (dw - cur_low) / dw * 100.0;
            double high_dev = (cur_high - up) / up * 100.0;
            double low_dev_h = (dw - cur_high) / dw * 100.0;
            double high_dev_l = (cur_low - up) / up * 100.0;
            if (low_hit && low_dev_h < low2 && !low_acted) {
                signal = 1; low_hit = 0; low_acted = 1;
            } else if (high_hit && high_dev_l < high2 && !high_acted) {
                signal = -1; high_hit = 0; high_acted = 1;
            }
            if (low_dev > low1 && !low_acted) { low_hit = 1; low_acted = 1; }
            if (high_dev > high1 && !high_acted) { high_hit = 1; high_acted = 1; }
        }

        // ---- 模拟成交 (kernel._execute 同式, 含倍投; 仅在有信号时执行) ----
        if (signal != 0) {
            if (signal == last_side) {
                cur_qty = cur_qty * scale;
            } else {
                cur_qty = trade_qty; last_side = signal;
            }
            if (signal == 1) {
                double q = (price > 0.0)
                    ? ((cur_qty < cash / price) ? cur_qty : cash / price)
                    : 0.0;
                if (q > 0.0) {
                    cash -= q * price; position += q;
                    n_trades++; n_buy++; turnover += q * price;
                }
            } else if (signal == -1) {
                double q = (cur_qty < position) ? cur_qty : position;
                if (q > 0.0) {
                    cash += q * price; position -= q;
                    n_trades++; n_sell++; turnover += q * price;
                }
            }
        }

        // ---- 权益回撤 + 超额曲线 (与 kernel.step 第8步同式) ----
        double eq = cash + position * last_price;
        if (eq > peak) peak = eq;
        if (peak > 0.0) {
            double dd = (peak - eq) / peak;
            if (dd > mdd) mdd = dd;
        }
        double base_eq = init_cash + init_position * last_price;
        double x = (eq - base_eq) / base_eq;
        if (x > x_peak) x_peak = x;
        double xdd = x_peak - x;
        if (xdd > x_mdd) x_mdd = xdd;
        int day = day_id[i];
        if (!day_init) {
            day_init = 1; cur_day = day;
        } else if (day != cur_day) {
            if (day_end_init) {
                double d = x_day_end - x_day_end_prev;
                d_sum += d; d_sum2 += d * d; d_n++;
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
}
"""

_cp = None
_kernel_cache = {}


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


def _compile_kernel():
    """编译 sweep_kernel (按设备架构, 失败回退 compute_90 PTX + 驱动 JIT)"""
    cp = _ensure_cupy()
    cc = cp.cuda.device.get_compute_capability()
    if isinstance(cc, (tuple, list)):
        archs = [f"compute_{cc[0]}{cc[1]}", "compute_90"]
    else:
        archs = ["compute_90"]
    last_err = None
    for arch in archs:
        try:
            return cp.RawKernel(_CUDA_SOURCE, "sweep_kernel",
                                options=(f"--gpu-architecture={arch}", "--fmad=false"))
        except Exception as e:  # NVRTC 不认识该架构时回退
            last_err = e
    raise last_err


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


def precompute_ts_mark(bars: dict, period: str, warmup_until: int):
    """(周期, 预热阈值) -> (ts int64[n], mark int8[n]); 与策略参数无关, 每组共享

    桶算法与 kernel.bucket_ts_encoded 同式 (本地锚定 epoch 取整, 任意 m/h/d 周期)。
    """
    stime = bars["stime"]
    P = resolve_period_seconds(period)
    e = _encoded_to_epoch_np(stime)
    e0 = (e // P) * P
    r = e - e0
    ts = np.where(r == 0,
                  _epoch_to_encoded_np(e0),
                  _epoch_to_encoded_np(e0 + P))
    mark = np.where(stime < warmup_until, 0, 1).astype(np.int8)
    return ts.astype(np.int64), mark


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


def cuda_sweep_window(bars: dict, params_list: list[dict],
                      warmup_until: int) -> list[dict]:
    """GPU 批量回测: 一组参数 -> 一个线程, 按周期分组各一次 launch

    params_list 元素含 period/tf1/low1/low2/high1/high2/trade_qty/init_cash/init_position。
    返回与 kernel.summarize 同口径的绩效字典列表 (顺序同 params_list)。
    """
    cp = _ensure_cupy()
    if not params_list:
        return []
    kernel = _kernel_cache.get("sweep")
    if kernel is None:
        kernel = _compile_kernel()
        _kernel_cache["sweep"] = kernel

    num = len(params_list)
    results = [None] * num

    # 按周期分组: 同周期共享预计算的 ts/mark
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(params_list):
        groups.setdefault(p.get("period", "5m"), []).append(i)

    d_o = cp.asarray(bars["open"])
    d_h = cp.asarray(bars["high"])
    d_l = cp.asarray(bars["low"])
    d_c = cp.asarray(bars["close"])
    d_v = cp.asarray(bars["volume"])
    # 日 id (连续化), 供超额 Sharpe 的日切检测; 与 CPU 的 stime//10^6 切换点一致
    day = bars["stime"] // 1_000_000
    _, day_inv = np.unique(day, return_inverse=True)
    d_day = cp.asarray(day_inv.astype(np.int32))
    n = cp.int64(len(bars["stime"]))

    for period, idxs in groups.items():
        m = len(idxs)
        ts_np, mark_np = precompute_ts_mark(bars, period, int(warmup_until))
        d_ts = cp.asarray(ts_np)
        d_mark = cp.asarray(mark_np)
        tf1s = cp.empty(m, cp.int64)
        scales = cp.empty(m, cp.float64)
        low1s = cp.empty(m, cp.float64)
        low2s = cp.empty(m, cp.float64)
        high1s = cp.empty(m, cp.float64)
        high2s = cp.empty(m, cp.float64)
        for j, i in enumerate(idxs):
            p = params_list[i]
            tf1s[j] = int(p.get("tf1", 21))
            scales[j] = float(p.get("scale", 1.0))
            low1s[j] = float(p.get("low1", 1.5))
            low2s[j] = float(p.get("low2", 1.0))
            high1s[j] = float(p.get("high1", 1.5))
            high2s[j] = float(p.get("high2", 0.5))
        out_cash = cp.empty(m, cp.float64)
        out_pos = cp.empty(m, cp.float64)
        out_last = cp.empty(m, cp.float64)
        out_ntrades = cp.empty(m, cp.int64)
        out_nbuy = cp.empty(m, cp.int64)
        out_nsell = cp.empty(m, cp.int64)
        out_mdd = cp.empty(m, cp.float64)
        out_turnover = cp.empty(m, cp.float64)
        out_dsum = cp.empty(m, cp.float64)
        out_dsum2 = cp.empty(m, cp.float64)
        out_dn = cp.empty(m, cp.int64)
        out_xmdd = cp.empty(m, cp.float64)

        threads = 256
        blocks = (m + threads - 1) // threads
        kernel((blocks,), (threads,), (
            d_ts, d_mark, d_day, d_o, d_h, d_l, d_c, d_v,
            n, cp.int32(m),
            cp.float64(float(params_list[idxs[0]].get("init_cash", INIT_CASH))),
            cp.float64(float(params_list[idxs[0]].get("init_position", INIT_POSITION))),
            cp.float64(float(params_list[idxs[0]].get("trade_qty", 10000.0))),
            scales, tf1s, low1s, low2s, high1s, high2s,
            out_cash, out_pos, out_last, out_ntrades, out_nbuy,
            out_nsell, out_mdd, out_turnover,
            out_dsum, out_dsum2, out_dn, out_xmdd,
        ))
        cp.cuda.get_current_stream().synchronize()

        cash = out_cash.get()
        pos = out_pos.get()
        last = out_last.get()
        ntr = out_ntrades.get()
        nbuy = out_nbuy.get()
        nsell = out_nsell.get()
        mdd = out_mdd.get()
        turnover = out_turnover.get()
        dsum = out_dsum.get()
        dsum2 = out_dsum2.get()
        dn = out_dn.get()
        xmdd = out_xmdd.get()

        # 年数: 与 kernel.summarize 同式 (首末策略期 bar 的自然日差 / 365.25)
        stime = bars["stime"]
        years = 0.0
        idx0 = int(np.searchsorted(stime, int(warmup_until)))
        if idx0 < len(stime) and stime[-1] > stime[idx0]:
            from .kernel import encoded_to_epoch
            years = ((encoded_to_epoch(int(stime[-1])) - encoded_to_epoch(int(stime[idx0])))
                     / (365.25 * 86400.0))

        for j, i in enumerate(idxs):
            p = params_list[i]
            init_cash = float(p.get("init_cash", INIT_CASH))
            init_position = float(p.get("init_position", INIT_POSITION))
            baseline = init_cash + init_position * float(last[j])
            equity = float(cash[j]) + float(pos[j]) * float(last[j])
            diff = equity - baseline
            pct = (diff / baseline * 100) if baseline else 0
            ann = (pct / years) if years > 0 else 0.0
            sharpe = 0.0
            if dn[j] >= 2:
                mean_d = dsum[j] / dn[j]
                var_d = dsum2[j] / dn[j] - mean_d * mean_d
                if var_d > 0.0:
                    sharpe = mean_d / var_d ** 0.5 * 252.0 ** 0.5
            results[i] = {
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
                "x_mdd": float(xmdd[j]),
                "max_drawdown": float(mdd[j]),
                "turnover": float(turnover[j]),
            }
    return results


if __name__ == "__main__":
    for k, v in gpu_info().items():
        print(f"{k:12s}: {v}")
