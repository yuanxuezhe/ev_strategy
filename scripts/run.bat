@echo off
rem ============================================================
rem evtrade 启动脚本 - 策略常用命令速查 / 一键执行
rem 用法:
rem   双击执行       -> 打印菜单 (不跑命令)
rem   run.bat <编号> -> 执行对应编号的命令
rem   run.bat list   -> 列出全部命令
rem 各命令也可直接复制本文档末尾的命令段手动执行。
rem ============================================================
setlocal
cd /d "%~dp0\.."

set PYTHON=uv run python
set CODE=159992.SZ
set START=20250101
set END=20260903

if "%~1"=="" goto :menu
if /i "%~1"=="list" goto :menu
goto :cmd%~1

:menu
echo.
echo ============ evtrade 常用命令 ============
echo  [1] 单参数回测 - channel_deviation 默认参数, CPU, 5m
echo  [2] 单参数回测 - 显式传 --params (覆盖默认)
echo  [3] 单参数回测 - ma_crossover 双均线策略
echo  [4] 网格参数扫描 - 单窗, low1/tf1 网格, 输出 CSV
echo  [5] 网格参数扫描 - 滚动 WFO (--splits) + 自动落盘最优参数
echo  [6] 网格参数扫描 - 仅产 SIG + final_state (无手续费/置换检验参数)
echo  [7] GPU 回测 (无 CUDA 自动降级, 需 --device auto)
echo  [8] 信号轨迹导出 (--signals-out CSV, 桶级对齐)
echo  [9] 无库体验 - 合成数据回测
echo  [0] 退出
echo ==========================================
set /p choice="输入编号执行: "
if "%choice%"=="" goto :eof
goto :cmd%choice%

:cmd1
rem 单参数回测: 策略走 _defaults/channel_deviation.json 落盘参数
%PYTHON% -m evtrade backtest --strategy channel_deviation --device cpu ^
    --period 5m --start %START% --end %END%
goto :end

:cmd2
rem 单参数回测: --params 显式覆盖 (优先级最高)
%PYTHON% -m evtrade backtest --strategy channel_deviation --device cpu ^
    --period 5m --start %START% --end %END% ^
    --params "tf1:21;low1:1.5;low2:1.0;high1:1.5;high2:0.5"
goto :end

:cmd3
rem 单参数回测: 另一个策略 ma_crossover
%PYTHON% -m evtrade backtest --strategy ma_crossover --device cpu ^
    --period 5m --start %START% --end %END% ^
    --params "fast:5;slow:20"
goto :end

:cmd4
rem 网格参数扫描: 单窗, 笛卡尔积 4x5=20 组, 结果存 sweep_results.csv
%PYTHON% -m evtrade sweep --strategy channel_deviation --device cpu ^
    --period 5m --start %START% --end %END% ^
    --grid "low1=1.0,1.5,2.0,2.5" ^
    --grid "tf1=10,21,30,50,70" ^
    --out sweep_results.csv --top 20
goto :end

:cmd5
rem 网格参数扫描: 滚动 WFO 3 窗 + 自动落盘最优参数到 _defaults
%PYTHON% -m evtrade sweep --strategy channel_deviation --device cpu ^
    --period 5m --start %START% --end %END% ^
    --grid "low1=1.0,1.5,2.0" ^
    --grid "high2=0.3,0.5,0.8" ^
    --splits 20250601,20251201,20260601 ^
    --out sweep_results.csv --top 20 --save-defaults
goto :end

:cmd6
rem 网格参数扫描: 仅产 SIG + final_state (无手续费/置换检验参数, framework 不再做)
%PYTHON% -m evtrade sweep --strategy channel_deviation --device cpu ^
    --period 5m --start %START% --end %END% ^
    --grid "low1=1.0,1.5,2.0" ^
    --grid "tf1=21,50" ^
    --out sweep_results.csv --top 10
goto :end

:cmd7
rem GPU 回测: 无 CUDA 时 auto 自动降级 cpu
%PYTHON% -m evtrade backtest --strategy channel_deviation --device auto ^
    --period 5m --start %START% --end %END%
goto :end

:cmd8
rem 信号轨迹导出: backtest --signals-out 写 (ts, sig) CSV (桶级对齐; framework 不再有 replay 子命令)
%PYTHON% -m evtrade backtest --strategy channel_deviation --device cpu ^
    --period 5m --start %START% --end %END% ^
    --signals-out signals.csv --verbose
goto :end

:cmd9
rem 无库体验: 合成数据 120 天 (不连数据库)
%PYTHON% -m evtrade sweep --strategy channel_deviation --device cpu ^
    --period 5m --end 20250701 --synthetic-days 120 ^
    --grid "low1=1.0,1.5,2.0" --grid "tf1=10,21,30" ^
    --out sweep_results.csv
goto :end

:end
echo.
pause
exit /b 0
