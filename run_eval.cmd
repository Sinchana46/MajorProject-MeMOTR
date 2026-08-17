@echo off
REM Run slice generation then fast evaluation, logging each to its own log file.
REM Uses %CONDA_PREFIX%\python.exe if the env is active, else falls back to python on PATH.

if defined CONDA_PREFIX (
    set PYTHON="%CONDA_PREFIX%\python.exe"
) else (
    set PYTHON=python
)

%PYTHON% tools/gen_slice_pil.py   > tools/gen_log.txt  2>&1
%PYTHON% tools/test_fast_eval.py  > tools/eval_log.txt 2>&1
