"""
tools/c2_eval_batch.py — detached driver: runs the remaining C2 evaluation jobs one after another through
tools/c2_eval_run.py, each with its own log check/logs/c2_<NN>_<seq>_<on|off>.txt. Skips a job whose
TrackEval summary already exists. Writes progress to results/c2_eval/batch_status.txt. Keeps the machine awake
while running (SetThreadExecutionState, no settings changed).
"""
import os, sys, subprocess, datetime, ctypes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = r"C:\Users\admin\anaconda3\envs\memotr_project\python.exe"
JOBS = [("05", "seq_01_1min", "off"), ("06", "seq_01_1min", "on"),
        ("07", "seq_03_30sec_b", "off"), ("08", "seq_03_30sec_b", "on")]
STATUS = os.path.join(ROOT, "results", "c2_eval", "batch_status.txt")

def note(msg):
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")

def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    note(f"batch start pid={os.getpid()}")
    try:
        for nn, seq, c2 in JOBS:
            summ = os.path.join(ROOT, "results", "c2_eval", f"{seq}_c2{c2}", "trackeval", "trackeval_output",
                                f"C2EVAL_{seq}_C2{c2.upper()}", "pedestrian_summary.txt")
            if os.path.exists(summ):
                note(f"{seq} {c2}: summary exists, skipped"); continue
            log = os.path.join(ROOT, "check", "logs", f"c2_{nn}_{seq}_{c2}.txt")
            note(f"{seq} {c2}: start -> {log}")
            with open(log, "w", encoding="utf-8") as f:
                r = subprocess.run([PYTHON, "-u", os.path.join(ROOT, "tools", "c2_eval_run.py"), seq, c2],
                                   cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            note(f"{seq} {c2}: exit={r.returncode}")
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        note("batch done")

if __name__ == "__main__":
    main()
