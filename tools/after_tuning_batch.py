"""
tools/after_tuning_batch.py — detached driver: runs corrected sweep settings S_B..S_G one after another via
tools/after_tuning_sweep_run.py (logs in results_after_tuning/logs/NN_<run>.txt). Skips a run whose summary exists.
Progress in results_after_tuning/logs/batch_status.txt. Keeps the machine awake while running.
"""
import os, subprocess, datetime, ctypes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = r"C:\Users\admin\anaconda3\envs\memotr_project\python.exe"
RUNS = ["S_B", "S_C", "S_D", "S_E", "S_F", "S_G"]
LOGS = os.path.join(ROOT, "results_after_tuning", "logs")
STATUS = os.path.join(LOGS, "batch_status.txt")

def note(m):
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {m}\n")

def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    note(f"batch start pid={os.getpid()}")
    try:
        for i, r in enumerate(RUNS, start=2):
            summ = os.path.join(ROOT, "results_after_tuning", "sweep", r, "trackeval", "trackeval_output",
                                f"AFTER_{r}", "pedestrian_summary.txt")
            if os.path.exists(summ):
                note(f"{r}: summary exists, skipped"); continue
            log = os.path.join(LOGS, f"{i:02d}_{r}.txt")
            note(f"{r}: start")
            with open(log, "w", encoding="utf-8") as f:
                rc = subprocess.run([PYTHON, "-u", os.path.join(ROOT, "tools", "after_tuning_sweep_run.py"), r],
                                    cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
            note(f"{r}: exit={rc}")
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        note("batch done")

if __name__ == "__main__":
    main()
