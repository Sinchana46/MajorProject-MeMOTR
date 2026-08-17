@echo off
REM Run direct_eval_benchmark.py, logging stdout+stderr to direct_log.txt.
REM Uses %CONDA_PREFIX%\python.exe if the env is active, else falls back to python on PATH.

if defined CONDA_PREFIX (
    "%CONDA_PREFIX%\python.exe" tools/direct_eval_benchmark.py > direct_log.txt 2>&1
) else (
    python tools/direct_eval_benchmark.py > direct_log.txt 2>&1
)
