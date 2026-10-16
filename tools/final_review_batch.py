"""
tools/final_review_batch.py — detached driver for the final-review inference jobs.

    python -u tools/final_review_batch.py <job_id> [<job_id> ...]

For each job: appends a "Started" entry to results_after_tuning/tuning_execution_walkthrough.md and a row to
results_after_tuning/execution_log.csv, runs tools/final_review_run.py, then appends "Completed" or "Failed".
A failed job is retried once (the failed output folder is renamed *_failed_attempt1, never deleted).
Progress: results_after_tuning/final_review/logs/batch_status.txt. Keeps the machine awake while running.
"""
import os, sys, csv, re, subprocess, datetime, ctypes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = r"C:\Users\admin\anaconda3\envs\memotr_project\python.exe"
AT = os.path.join(ROOT, "results_after_tuning")
WALK, LOG = os.path.join(AT, "tuning_execution_walkthrough.md"), os.path.join(AT, "execution_log.csv")
FR = os.path.join(AT, "final_review")
STATUS = os.path.join(FR, "logs", "batch_status.txt")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_review_run import JOBS, CKPT  # noqa: E402

def now():
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

def next_id():
    ids = [int(r["entry_id"][1:]) for r in csv.DictReader(open(LOG, encoding="utf-8")) if r["entry_id"][1:].isdigit()]
    return f"E{max(ids) + 1:03d}"

def note(m):
    with open(STATUS, "a", encoding="utf-8") as f:
        f.write(f"{now()} {m}\n")

def describe(jid):
    j = JOBS[jid]
    if jid.startswith("C2_"):
        q = (f"With the corrected output filter, does C2 (scene-conditioned detection threshold) change tracking on "
             f"{j['seq']} when C2 is {j['c2']}?")
        why = "Earlier C2 numbers were from the unfiltered output and cannot be compared with corrected results."
    else:
        q = f"How does {j['model']} checkpoint_{j['ckpt']} score on seq_02_30sec_a under the corrected output filter?"
        why = ("Every model must be scored with the same corrected procedure before a best checkpoint or a best model "
               "can be named; the old scores were from unfiltered output.")
    return q, why

def row(eid, status, jid, out, key="", plain="", err="", nxt=""):
    j = JOBS[jid]; ck_dir, cfg = CKPT[j["model"]]
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f, quoting=csv.QUOTE_ALL).writerow([
            eid, now(), status, f"Corrected inference {jid}", f"tools/final_review_run.py {jid}",
            os.path.relpath(os.path.join(ck_dir, f"checkpoint_{j['ckpt']}.pth"), ROOT).replace(os.sep, "/"),
            "42", j["seq"], "0.5", "0.5", "30", "0.5", "100",
            f"S0 TrackEval DO_PREPROC=False; filtered runner; C2 {j['c2']}; config {os.path.basename(cfg)}",
            os.path.relpath(out, ROOT).replace(os.sep, "/"),
            os.path.relpath(os.path.join(FR, "logs", f"{jid}.txt"), ROOT).replace(os.sep, "/"),
            key, plain, err, nxt, os.path.relpath(out, ROOT).replace(os.sep, "/")])

def walk(text):
    with open(WALK, "a", encoding="utf-8") as f:
        f.write(text)

def run_one(jid):
    out = os.path.join(FR, "runs", jid); log = os.path.join(FR, "logs", f"{jid}.txt")
    j = JOBS[jid]; ck_dir, cfg = CKPT[j["model"]]
    q, why = describe(jid)
    eid = next_id()
    walk(f"\n### {eid} — Corrected inference `{jid}` — Started ({now()})\n"
         f"- **Question:** {q}\n- **Why:** {why}\n"
         f"- **Inputs:** checkpoint `{os.path.relpath(os.path.join(ck_dir, 'checkpoint_%d.pth' % j['ckpt']), ROOT)}`, "
         f"config `{os.path.relpath(cfg, ROOT)}`, sequence `{j['seq']}`, GT `{os.path.relpath(j['gt'], ROOT)}` "
         f"(class-1 evaluation copy of the original GT), det 0.5 / track 0.5 / miss 30, output filter score > 0.5 and "
         f"area > 100 px (`tools/run_inference_mot_filtered.py`), C2 {j['c2']}, C1 off, TrackEval DO_PREPROC=False.\n")
    row(eid, "Started", jid, out)
    for attempt in (1, 2):
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run([PYTHON, "-u", os.path.join(ROOT, "tools", "final_review_run.py"), jid], cwd=ROOT,
                                stdout=f, stderr=subprocess.STDOUT).returncode
        txt = open(log, encoding="utf-8", errors="replace").read()
        m = re.search(r"RESULT (.*)", txt)
        if rc == 0 and m:
            d = dict(kv.split("=") for kv in m.group(1).split())
            key = (f"HOTA {d['HOTA']} DetA {d['DetA']} AssA {d['AssA']} MOTA {d['MOTA']} IDF1 {d['IDF1']} "
                   f"IDSW {d['IDSW']} FP {d['CLR_FP']} FN {d['CLR_FN']} Prec {d['CLR_Pr']} Rec {d['CLR_Re']}")
            filt = re.search(r"output filter \(.*?\): (\{.*\})", txt); tau = re.search(r"tau over .*", txt)
            eid2 = next_id()
            walk(f"\n### {eid2} — Corrected inference `{jid}` — Completed ({now()}) — follows {eid}\n"
                 f"- **What was done:** ran `tools/final_review_run.py {jid}` (attempt {attempt}); inference with the "
                 f"filtered runner, then TrackEval.\n"
                 f"- **Result (Verified, TrackEval summary):** {key}.\n"
                 + (f"- **Output filter counts:** {filt.group(1)}\n" if filt else "")
                 + (f"- **C2 threshold:** {tau.group(0)}\n" if tau else "")
                 + f"- **Comparable with:** other corrected-procedure rows only (not with old unfiltered numbers).\n"
                 f"- **Outputs:** `{os.path.relpath(out, ROOT)}`; log `{os.path.relpath(log, ROOT)}`.\n"
                 + (f"- **Deviation:** first attempt failed; failed output kept as "
                    f"`{os.path.relpath(out, ROOT)}_failed_attempt1`.\n" if attempt == 2 else ""))
            row(eid2, "Completed", jid, out, key, "Scored under the corrected procedure",
                "retry after failed attempt 1" if attempt == 2 else "", "next queued job")
            note(f"{jid}: completed (attempt {attempt})")
            return True
        err = (re.findall(r"(REFUSE.*|BLOCKED.*|.*FAILED.*)", txt) or [f"rc={rc}"])[-1]
        note(f"{jid}: attempt {attempt} failed: {err}")
        if attempt == 1 and os.path.exists(out) and not err.startswith(("REFUSE", "BLOCKED")):
            os.rename(out, out + "_failed_attempt1"); os.rename(log, log[:-4] + "_failed_attempt1.txt")
            continue
        break
    eid2 = next_id()
    walk(f"\n### {eid2} — Corrected inference `{jid}` — Failed ({now()}) — follows {eid}\n"
         f"- **What happened:** {err}. No metrics are reported for this job.\n"
         f"- **Evidence:** `{os.path.relpath(log, ROOT)}`.\n")
    row(eid2, "Failed", jid, out, "", "No result", err, "reported as unavailable")
    return False

def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    note(f"batch start pid={os.getpid()} jobs={sys.argv[1:]}")
    try:
        for jid in sys.argv[1:]:
            run_one(jid)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        note("batch done")

if __name__ == "__main__":
    main()
