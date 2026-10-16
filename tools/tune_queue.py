"""
tools/tune_queue.py  —  Tuning Stage 1 queue driver
Runs 4 jobs sequentially: seed1, seed2, assoc_lr0, lr_half
Each job: train → verify checkpoints → evaluate all 5 ckpts → append to CSV → param-diff check

Launch detached (from project root):
    Start-Process -FilePath python -ArgumentList '-u','tools/tune_queue.py' `
        -WindowStyle Hidden `
        -RedirectStandardOutput results/tuning_stage1/logs/queue.out `
        -RedirectStandardError  results/tuning_stage1/logs/queue.err -PassThru

Rules:
 - No resume / no delete / no overwrite of anything
 - Backbone LR stays 0 in every job
 - Evaluation: thresh 0.5, miss_tol 30, use_c1=False, use_c2=False, DO_PREPROC=False
 - Scoring S0 (original GT SHA256 17ebe2782a0e9008f98e4e30e975b6a6dee5ffc72551b0e364e8ea4818cd3472)
"""
import os, sys, csv, json, time, hashlib, shutil, subprocess, ctypes, ctypes.wintypes
import datetime
import threading

# stdout/stderr are tee'd into queue_direct.log (append mode) once main() starts,
# so output is captured even when launched hidden/detached.
_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tuning_stage1", "logs", "queue_direct.log")

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            try: s.write(data); s.flush()
            except Exception: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass
    def fileno(self): return self.streams[0].fileno() if self.streams else -1

def _install_tee():
    os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
    fh = open(_LOG_FILE, "a", buffering=1, encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Use the memotr_project conda env which has torch — same as used for Arm 1.
# sys.executable resolves to the base conda env which does NOT have torch.
PYTHON = r"C:\Users\admin\anaconda3\envs\memotr_project\python.exe"

# CUDA bin must be on PATH so the MultiScaleDeformableAttention .pyd can load its DLLs
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"

def make_env():
    """Build subprocess env: inherit current, add CUDA bin to PATH, force UTF-8."""
    env = os.environ.copy()
    ops_dir = os.path.join(MEMOTR_DIR, "models", "ops")
    env["PYTHONPATH"] = MEMOTR_DIR + os.pathsep + ops_dir
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"   # force UTF-8 file I/O on Windows (fixes cp1252 YAML parse)
    if CUDA_BIN not in env.get("PATH", ""):
        env["PATH"] = CUDA_BIN + os.pathsep + env.get("PATH", "")
    return env

# ── Paths ──────────────────────────────────────────────────────────────────────
STAGE_DIR   = os.path.join(ROOT, "results", "tuning_stage1")
LOG_DIR     = os.path.join(STAGE_DIR, "logs")
EVAL_DIR    = os.path.join(STAGE_DIR, "eval")
STATUS_JSON = os.path.join(STAGE_DIR, "status.json")
RESULTS_CSV = os.path.join(STAGE_DIR, "stage1_results.csv")
PID_FILE    = os.path.join(STAGE_DIR, "queue.pid")
HASH_FILE   = os.path.join(ROOT, "data", "IOTD", "check", "hashes_before.json")
GT_PATH     = os.path.join(ROOT, "data", "IOTD_val", "seq_02_30sec_a", "gt", "gt.txt")
PHASE5_CKPT = os.path.join(ROOT, "results", "phase5_interim_baseline", "checkpoint_4.pth")
MEMOTR_DIR  = os.path.join(ROOT, "MeMOTR")
TRACKEVAL_DIR = os.path.join(ROOT, "TrackEval")
SEQINFO_VAL = os.path.join(ROOT, "data", "IOTD_val", "seq_02_30sec_a", "seqinfo.ini")

GT_SHA256_EXPECTED = "17ebe2782a0e9008f98e4e30e975b6a6dee5ffc72551b0e364e8ea4818cd3472"

# ── Jobs ───────────────────────────────────────────────────────────────────────
JOBS = [
    # seed1 uses the _s2 copy (output results/tuning_stage1/seed1_r2): the original seed1
    # OUTPUTS_DIR and results/tuning_stage1/seed1 hold partial outputs from the two killed
    # launches on 2026-10-15, which must not be reused or deleted.
    {"name": "seed1",     "config": "configs/train_iotd_real_seed1_s2.yaml",  "seed": 1234},
    {"name": "seed2",     "config": "configs/train_iotd_real_seed2.yaml",     "seed": 2024},
    {"name": "assoc_lr0", "config": "configs/train_iotd_real_assoc_lr0.yaml", "seed": 42},
    {"name": "lr_half",   "config": "configs/train_iotd_real_lr_half.yaml",   "seed": 42},
]
# Output directories per job (must match OUTPUTS_DIR in each config)
JOB_OUTPUTS = {
    "seed1":     os.path.join(ROOT, "results", "tuning_stage1", "seed1_r2"),
    "seed2":     os.path.join(ROOT, "results", "train_iotd_real_seed2"),
    "assoc_lr0": os.path.join(ROOT, "results", "tuning_stage1", "assoc_lr0"),
    "lr_half":   os.path.join(ROOT, "results", "tuning_stage1", "lr_half"),
}
NUM_CKPTS = 5

# ── CSV columns (match baseline registry) ─────────────────────────────────────
CSV_COLS = ["run","checkpoint","steps","epochs_completed","seed","config","scoring","threshold",
            "GT_sha256","HOTA","DetA","AssA","MOTA","IDF1","IDSW","TP","FP","FN",
            "recall","precision","summary_path","summary_sha256"]
STEPS_PER_EPOCH = 2528
CKPT_STEPS = [STEPS_PER_EPOCH * (i+1) for i in range(NUM_CKPTS)]

# ── Keep-awake ─────────────────────────────────────────────────────────────────
ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001

def keep_awake():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

def release_awake():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

# ── Helpers ────────────────────────────────────────────────────────────────────
def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def load_status():
    if os.path.exists(STATUS_JSON):
        with open(STATUS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {"jobs": {}, "queue_pid": None, "queue_start": None}

def save_status(status):
    status["heartbeat"] = now_utc()
    with open(STATUS_JSON, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

def log(msg, logfile=None):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def read_te_summary(path):
    """Read a TrackEval pedestrian_summary.txt."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < 2:
        return None
    d = dict(zip(lines[0].split(), lines[1].split()))
    return {
        "HOTA": float(d.get("HOTA", 0)),   "DetA": float(d.get("DetA", 0)),
        "AssA": float(d.get("AssA", 0)),   "MOTA": float(d.get("MOTA", 0)),
        "IDF1": float(d.get("IDF1", 0)),   "IDSW": float(d.get("IDSW", 0)),
        "TP":   float(d.get("CLR_TP", 0)), "FP":   float(d.get("CLR_FP", 0)),
        "FN":   float(d.get("CLR_FN", 0)),
        "recall":    float(d.get("CLR_Re", 0)),
        "precision": float(d.get("CLR_Pr", 0)),
    }

# ── Heartbeat thread ───────────────────────────────────────────────────────────
_status_lock = threading.Lock()
_status_global = {}

def heartbeat_thread(interval=60):
    while True:
        time.sleep(interval)
        with _status_lock:
            save_status(_status_global)

# ── Hash check ─────────────────────────────────────────────────────────────────
def _flatten_hash_entries(hash_json):
    """
    Flatten the nested hashes_before.json into a list of (relpath, sha256) tuples.

    JSON shape:
      { "sequences": { "<seq_name>": {
          "annotation_files": { "gt/gt.txt": sha, "seqinfo.ini": sha, ... },
          "img1":             { "000001.jpg": sha, ... }
      }}}

    On disk:
      annotation_files entries live at  data/IOTD/<seq>/<fname>
      img1 entries live at              data/IOTD/<seq>/img1/<fname>
    """
    entries = []
    sequences = hash_json.get("sequences", {})
    for seq_name, seq_data in sequences.items():
        for subdir, files in seq_data.items():
            if not isinstance(files, dict):
                continue
            for fname, sha in files.items():
                if subdir == "annotation_files":
                    # keys are already relative to the seq dir
                    relpath = f"data/IOTD/{seq_name}/{fname}"
                else:
                    relpath = f"data/IOTD/{seq_name}/{subdir}/{fname}"
                entries.append((relpath, sha))
    return entries


def verify_hashes():
    """Re-hash all original IOTD files against hashes_before.json."""
    if not os.path.exists(HASH_FILE):
        return None, "hashes_before.json not found — skipping"
    with open(HASH_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    entries = _flatten_hash_entries(raw)
    if not entries:
        return None, "hashes_before.json: no file entries found after flattening"
    changed = []; missing = []
    from concurrent.futures import ThreadPoolExecutor
    def check_one(item):
        relpath, exp_hash = item
        abspath = os.path.join(ROOT, relpath)
        if not os.path.exists(abspath):
            return "missing", relpath
        actual = sha256_file(abspath)
        if actual != exp_hash:
            return "changed", relpath
        return "ok", relpath
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(check_one, entries))
    for r, p in results:
        if r == "missing": missing.append(p)
        elif r == "changed": changed.append(p)
    if not changed and not missing:
        return True, f"Hash check PASS ({len(entries)} files)"
    return False, f"Hash check FAIL: changed={changed[:3]} missing={missing[:3]}"

# ── Disk and GPU check ─────────────────────────────────────────────────────────
def check_disk_and_gpu():
    import shutil as sh
    free_gb = sh.disk_usage(ROOT).free / 1e9
    if free_gb < 30:
        return False, f"Only {free_gb:.1f} GB free — need 30 GB"
    # GPU check via nvidia-smi
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        util = int(r.stdout.strip().replace("%","").strip())
        if util > 80:
            return False, f"GPU util {util}% — another process may be training"
    except Exception:
        pass  # non-fatal
    return True, f"Disk {free_gb:.1f} GB free; GPU check passed"

# ── Run training ──────────────────────────────────────────────────────────────
def run_training(job_name, config_path, train_log):
    """Run training subprocess. Returns (returncode, wall_seconds)."""
    cmd = [PYTHON, "-u", os.path.join(MEMOTR_DIR, "main.py"),
           "--config-path", os.path.join(ROOT, config_path)]
    env = make_env()
    t0 = time.time()
    with open(train_log, "w", encoding="utf-8") as flog:
        proc = subprocess.run(cmd, cwd=ROOT, env=env,
                              stdout=flog, stderr=subprocess.STDOUT, timeout=None)
    return proc.returncode, time.time() - t0

# ── Verify checkpoints ────────────────────────────────────────────────────────
def verify_checkpoints(out_dir, job_name, job_log):
    """Check that 5 checkpoints exist and each is ~575 MB."""
    ok = True
    for i in range(NUM_CKPTS):
        ckpt = os.path.join(out_dir, f"checkpoint_{i}.pth")
        if not os.path.exists(ckpt):
            log(f"  MISSING: {ckpt}", job_log)
            ok = False
        else:
            size_mb = os.path.getsize(ckpt) / 1e6
            if size_mb < 400:
                log(f"  WARNING: {ckpt} is only {size_mb:.0f} MB (expected ~575)", job_log)
                ok = False
            else:
                log(f"  OK ckpt_{i}: {size_mb:.0f} MB", job_log)
    return ok

# ── Run inference (one checkpoint) ────────────────────────────────────────────
# Same tools and settings as the pre-tuning baseline package (Arm 1 / Phase 5):
#   tools/run_inference_mot.py  --det-thresh 0.5 --track-thresh 0.5 --miss-tol 30
#   config configs/train_iotd_real.yaml (USE_C1=False, USE_C2=False), no C2 adapter
#   tools/run_trackeval.py      (direct TrackEval API, DO_PREPROC=False, THRESHOLD 0.5)
#   GT: results/eval_iotd_val/gt_eval (evaluation-only class-1 copy of the S0 GT)
VAL_SPLIT_DIR = os.path.join(ROOT, "data", "IOTD_val")
GT_EVAL_DIR   = os.path.join(ROOT, "results", "eval_iotd_val", "gt_eval")
INFER_CONFIG  = os.path.join(ROOT, "configs", "train_iotd_real.yaml")

def run_inference(job_name, ckpt_path, ckpt_idx, out_dir, job_log):
    """Run MeMOTR inference on data/IOTD_val (seq_02_30sec_a). Returns tracker dir or None."""
    ckpt_dir    = os.path.join(out_dir, f"ckpt_{ckpt_idx}")
    tracker_out = os.path.join(ckpt_dir, "tracker")
    os.makedirs(tracker_out, exist_ok=True)
    cmd = [PYTHON, "-u", os.path.join(ROOT, "tools", "run_inference_mot.py"),
           "--checkpoint",   ckpt_path,
           "--split-dirs",   VAL_SPLIT_DIR,
           "--output-dir",   tracker_out,
           "--config",       INFER_CONFIG,
           "--label",        f"{job_name.upper()}_CKPT{ckpt_idx}",
           "--det-thresh",   "0.5",
           "--track-thresh", "0.5",
           "--miss-tol",     "30"]
    inf_log = os.path.join(ckpt_dir, "inference.log")
    t0 = time.time()
    try:
        with open(inf_log, "w", encoding="utf-8") as flog:
            flog.write("CMD: " + " ".join(cmd) + chr(10)); flog.flush()
            r = subprocess.run(cmd, cwd=MEMOTR_DIR, env=make_env(),
                               stdout=flog, stderr=subprocess.STDOUT, timeout=1800)
    except subprocess.TimeoutExpired:
        log(f"    Inference TIMEOUT (see {inf_log})", job_log)
        return None
    trk_txt = os.path.join(tracker_out, "seq_02_30sec_a.txt")
    if r.returncode != 0 or not os.path.exists(trk_txt) or os.path.getsize(trk_txt) == 0:
        log(f"    Inference FAILED (code {r.returncode}; see {inf_log})", job_log)
        return None
    log(f"    inference done in {time.time()-t0:.0f}s", job_log)
    return tracker_out

# ── Run TrackEval ──────────────────────────────────────────────────────────────
def run_trackeval(tracker_out, eval_out, job_name, ckpt_idx, job_log):
    """Score with tools/run_trackeval.py. Returns pedestrian_summary.txt path or None."""
    label  = f"{job_name.upper()}_CKPT{ckpt_idx}"
    te_out = os.path.join(eval_out, f"ckpt_{ckpt_idx}", "trackeval")
    os.makedirs(te_out, exist_ok=True)
    cmd = [PYTHON, "-u", os.path.join(ROOT, "tools", "run_trackeval.py"),
           "--tracker-dir", tracker_out,
           "--gt-dirs",     GT_EVAL_DIR,
           "--output-dir",  te_out,
           "--label",       label,
           "--num-cores",   "1"]
    te_log = os.path.join(eval_out, f"ckpt_{ckpt_idx}", "trackeval.log")
    try:
        with open(te_log, "w", encoding="utf-8") as flog:
            flog.write("CMD: " + " ".join(cmd) + chr(10)); flog.flush()
            r = subprocess.run(cmd, cwd=ROOT, env=make_env(),
                               stdout=flog, stderr=subprocess.STDOUT, timeout=600)
    except subprocess.TimeoutExpired:
        log(f"    TrackEval TIMEOUT (see {te_log})", job_log)
        return None
    summary = os.path.join(te_out, "trackeval_output", label, "pedestrian_summary.txt")
    if r.returncode != 0 or not os.path.exists(summary):
        log(f"    TrackEval FAILED (code {r.returncode}; see {te_log})", job_log)
        return None
    return summary

# ── Append to CSV ──────────────────────────────────────────────────────────────
def append_csv(row: dict):
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

# ── Parameter-difference check ─────────────────────────────────────────────────
def param_diff_check(job_name, final_ckpt_path, job_log):
    """
    Compare final checkpoint parameters against Phase 5 checkpoint_4.
      backbone_max   : max |diff| over keys containing 'backbone.backbone' (must be 0)
      qu_max         : max |diff| over keys containing 'query_updater' (must be 0 for assoc_lr0)
      nonbb_mean_abs : mean |diff| over ALL elements of every non-backbone floating tensor
                       (same key set for every job, so values are comparable across jobs)
    """
    try:
        import torch
        ph5 = torch.load(PHASE5_CKPT, map_location="cpu", weights_only=False)
        new = torch.load(final_ckpt_path, map_location="cpu", weights_only=False)
        ph5_sd = ph5.get("model", ph5)
        new_sd = new.get("model", new)

        backbone_max = 0.0; qu_max = 0.0
        nonbb_sum = 0.0; nonbb_n = 0; nonbb_keys = 0; missing = []
        for k, v in ph5_sd.items():
            if k not in new_sd:
                missing.append(k); continue
            if not torch.is_floating_point(v):
                continue
            d = (v.double() - new_sd[k].double()).abs()
            if "backbone.backbone" in k:
                backbone_max = max(backbone_max, d.max().item())
                continue
            if "query_updater" in k:
                qu_max = max(qu_max, d.max().item())
            nonbb_sum += d.sum().item(); nonbb_n += d.numel(); nonbb_keys += 1

        mean_nonbb = nonbb_sum / nonbb_n if nonbb_n else 0.0
        log(f"  Param diff vs Phase5 ckpt4: backbone_max={backbone_max:.3e}  "
            f"query_updater_max={qu_max:.3e}  nonbackbone_mean_abs={mean_nonbb:.3e} "
            f"({nonbb_keys} tensors, {nonbb_n} elements; {len(missing)} keys missing)", job_log)
        backbone_ok = backbone_max == 0.0
        qu_ok = (qu_max == 0.0) if job_name == "assoc_lr0" else None
        if not backbone_ok:
            log(f"  FAIL: backbone changed (max diff={backbone_max})", job_log)
        if qu_ok is False:
            log(f"  FAIL: query_updater changed in assoc_lr0 (max diff={qu_max})", job_log)
        return {"backbone_max": backbone_max, "query_updater_max": qu_max,
                "nonbackbone_mean_abs": mean_nonbb, "nonbackbone_tensors": nonbb_keys,
                "nonbackbone_elements": nonbb_n, "missing_keys": len(missing),
                "backbone_ok": backbone_ok, "query_updater_ok": qu_ok}
    except Exception as e:
        log(f"  Param diff check ERROR: {e}", job_log)
        return {"error": str(e)}

# ── Training-log checks ────────────────────────────────────────────────────────
def check_train_log(train_log, job_log):
    """Loaded 2 sequences, 2,528 iterations per epoch, seq_02_30sec_a absent."""
    try:
        with open(train_log, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except Exception as e:
        return {"error": str(e)}
    res = {
        "loaded_2_sequences": "Loaded 2 sequences" in txt,
        "iters_2528":         "/2528 iters" in txt and "/2528" in txt,
        "val_seq_absent":     "seq_02_30sec_a" not in txt,
        "weights_loaded":     "Pretrained model is loaded" in txt,
    }
    log(f"  Train-log checks: {res}", job_log)
    return res

# ── Main queue ─────────────────────────────────────────────────────────────────
def main():
    _install_tee()
    os.makedirs(STAGE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Write PID
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    # Keep awake
    keep_awake()

    # Load/init status
    global _status_global
    _status_global = load_status()
    _status_global["queue_pid"] = os.getpid()
    _status_global["queue_start"] = now_utc()
    save_status(_status_global)

    # Start heartbeat
    t = threading.Thread(target=heartbeat_thread, args=(60,), daemon=True)
    t.start()

    # Print time estimate from measured step time
    step_time_s = 0.664
    train_per_job = step_time_s * 12640  # 5 epochs * 2528 steps
    infer_per_ckpt_s = 135               # measured 131 s inference + 4 s TrackEval (tune_eval_smoke.py)
    total_est_h = (train_per_job * 4 + infer_per_ckpt_s * NUM_CKPTS * 4) / 3600
    print(f"\n{'='*70}")
    print(f"TUNING STAGE 1 QUEUE — {len(JOBS)} jobs × 5 epochs × 2,528 steps/epoch")
    print(f"Measured step time: {step_time_s:.3f} s/step  (from Arm 1 training log)")
    print(f"Train time/job: ~{train_per_job/3600:.2f} h  |  Total est: ~{total_est_h:.1f} h")
    print(f"Job order: seed1, seed2, assoc_lr0, lr_half")
    print(f"Output dir: {STAGE_DIR}")
    print(f"PID: {os.getpid()}")
    print(f"{'='*70}\n")

    # Pre-launch hash check
    hash_ok, hash_msg = verify_hashes()
    print(f"Pre-launch hash check: {hash_msg}")
    if hash_ok is False:
        print("ABORT: hash check failed. Nothing started.")
        release_awake()
        sys.exit(1)

    # GT SHA256 check
    gt_sha = sha256_file(GT_PATH)
    if gt_sha != GT_SHA256_EXPECTED:
        print(f"ABORT: GT SHA256 mismatch! got={gt_sha}")
        release_awake()
        sys.exit(1)
    print(f"GT SHA256: {gt_sha}  [MATCH]")

    # ── Process each job ──────────────────────────────────────────────────────
    for job in JOBS:
        jname     = job["name"]
        jconfig   = job["config"]
        jseed     = job["seed"]
        jout_dir  = JOB_OUTPUTS[jname]
        jeval_dir = os.path.join(EVAL_DIR, jname)
        jlog_file = os.path.join(LOG_DIR, f"{jname}_train.log")
        job_log   = os.path.join(LOG_DIR, f"{jname}_queue.log")

        os.makedirs(jeval_dir, exist_ok=True)

        log(f"\n{'='*60}", job_log)
        log(f"JOB: {jname}  config={jconfig}  seed={jseed}", job_log)
        log(f"{'='*60}", job_log)

        with _status_lock:
            _status_global.setdefault("jobs", {})[jname] = {
                "status": "starting", "start": now_utc(), "error": None}
            save_status(_status_global)

        # ── Check output dir not already populated ────────────────────────────
        ckpt0 = os.path.join(jout_dir, "checkpoint_0.pth")
        all_ckpts_exist = all(
            os.path.exists(os.path.join(jout_dir, f"checkpoint_{i}.pth"))
            for i in range(NUM_CKPTS)
        )
        # Check if all CSV rows already present
        csv_rows_ok = False
        if os.path.exists(RESULTS_CSV):
            with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
                existing = [r for r in csv.DictReader(f) if r["run"] == jname]
            csv_rows_ok = len(existing) >= NUM_CKPTS

        if all_ckpts_exist and csv_rows_ok:
            log(f"  SKIP: {jname} already has all checkpoints and {len(existing)} CSV rows", job_log)
            with _status_lock:
                _status_global["jobs"][jname]["status"] = "skipped_complete"
                save_status(_status_global)
            continue

        if os.path.exists(jout_dir) and os.listdir(jout_dir) and not all_ckpts_exist:
            log(f"  WARNING: {jout_dir} exists but incomplete. Will not overwrite. Skipping.", job_log)
            with _status_lock:
                _status_global["jobs"][jname].update({"status": "skipped_incomplete", "error": "partial output"})
                save_status(_status_global)
            continue

        if os.path.exists(jout_dir) and os.listdir(jout_dir) and all_ckpts_exist:
            log(f"  Checkpoints complete; re-evaluating only.", job_log)
        else:
            # ── Phase 1: Disk + GPU check ─────────────────────────────────────
            disk_ok, disk_msg = check_disk_and_gpu()
            log(f"  Disk/GPU: {disk_msg}", job_log)
            if not disk_ok:
                log(f"  SKIP {jname}: {disk_msg}", job_log)
                with _status_lock:
                    _status_global["jobs"][jname].update({"status": "skipped_resources", "error": disk_msg})
                    save_status(_status_global)
                continue

            # ── Phase 2: Training ─────────────────────────────────────────────
            log(f"  Phase 2: training → {jlog_file}", job_log)
            with _status_lock:
                _status_global["jobs"][jname]["status"] = "training"
                _status_global["jobs"][jname]["train_start"] = now_utc()
                save_status(_status_global)
            try:
                retcode, wall = run_training(jname, jconfig, jlog_file)
                log(f"  Training done: retcode={retcode} wall={wall/3600:.2f}h", job_log)
                if retcode != 0:
                    raise RuntimeError(f"Training exited with code {retcode}")
            except Exception as e:
                log(f"  Training FAILED: {e}", job_log)
                with _status_lock:
                    _status_global["jobs"][jname].update({"status": "train_failed", "error": str(e)})
                    save_status(_status_global)
                continue

            # ── Phase 3: Verify checkpoints ───────────────────────────────────
            ckpt_ok = verify_checkpoints(jout_dir, jname, job_log)
            log_checks = check_train_log(jlog_file, job_log)
            with _status_lock:
                _status_global["jobs"][jname]["train_log_checks"] = log_checks
                _status_global["jobs"][jname]["train_wall_h"] = round(wall / 3600, 3)
                save_status(_status_global)
            if not ckpt_ok:
                log(f"  Checkpoint verification FAILED for {jname}", job_log)
                with _status_lock:
                    _status_global["jobs"][jname].update({"status": "ckpt_verify_failed"})
                    save_status(_status_global)
                continue

        # ── Phase 4+5: Evaluate all 5 checkpoints ─────────────────────────────
        log(f"  Phase 4+5: evaluating all {NUM_CKPTS} checkpoints", job_log)
        with _status_lock:
            _status_global["jobs"][jname]["status"] = "evaluating"
            _status_global["jobs"][jname]["eval_start"] = now_utc()
            save_status(_status_global)

        for ckpt_idx in range(NUM_CKPTS):
            ckpt_path  = os.path.join(jout_dir, f"checkpoint_{ckpt_idx}.pth")
            ckpt_label = f"ckpt_{ckpt_idx}"
            steps      = CKPT_STEPS[ckpt_idx]
            epoch_done = ckpt_idx + 1

            log(f"    ckpt_{ckpt_idx} ({steps} steps) ...", job_log)

            # Check if already in CSV
            if os.path.exists(RESULTS_CSV):
                with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
                    existing = [r for r in csv.DictReader(f)
                                if r["run"] == jname and r["checkpoint"] == ckpt_label
                                and r["scoring"] == "S0"]
                if existing:
                    log(f"    SKIP ckpt_{ckpt_idx}: already in CSV", job_log)
                    continue

            # Inference
            trk_out = run_inference(jname, ckpt_path, ckpt_idx,
                                    os.path.join(jeval_dir), job_log)
            if trk_out is None:
                log(f"    SKIP ckpt_{ckpt_idx}: inference failed", job_log)
                continue

            # TrackEval
            summary_path = run_trackeval(trk_out, jeval_dir, jname, ckpt_idx, job_log)
            if summary_path is None:
                log(f"    SKIP ckpt_{ckpt_idx}: TrackEval failed", job_log)
                continue

            metrics = read_te_summary(summary_path)
            if metrics is None:
                log(f"    SKIP ckpt_{ckpt_idx}: cannot read summary", job_log)
                continue

            summary_sha = sha256_file(summary_path)
            log(f"    ckpt_{ckpt_idx}: HOTA={metrics['HOTA']:.2f} "
                f"DetA={metrics['DetA']:.2f} AssA={metrics['AssA']:.2f} "
                f"MOTA={metrics['MOTA']:.2f} IDF1={metrics['IDF1']:.2f} "
                f"IDSW={int(metrics['IDSW'])} FP={int(metrics['FP'])} FN={int(metrics['FN'])}", job_log)

            append_csv({
                "run": jname, "checkpoint": ckpt_label,
                "steps": steps, "epochs_completed": epoch_done,
                "seed": jseed, "config": jconfig, "scoring": "S0", "threshold": 0.5,
                "GT_sha256": GT_SHA256_EXPECTED,
                "HOTA": round(metrics["HOTA"], 3), "DetA": round(metrics["DetA"], 3),
                "AssA": round(metrics["AssA"], 3), "MOTA": round(metrics["MOTA"], 3),
                "IDF1": round(metrics["IDF1"], 3), "IDSW": int(metrics["IDSW"]),
                "TP": int(metrics["TP"]), "FP": int(metrics["FP"]), "FN": int(metrics["FN"]),
                "recall": round(metrics["recall"], 3),
                "precision": round(metrics["precision"], 3),
                "summary_path": summary_path, "summary_sha256": summary_sha,
            })

        # ── Phase 6: Param-diff check on final checkpoint ─────────────────────
        final_ckpt = os.path.join(jout_dir, f"checkpoint_{NUM_CKPTS-1}.pth")
        if os.path.exists(final_ckpt):
            log(f"  Phase 6: param diff check vs Phase5 ckpt4", job_log)
            diff_result = param_diff_check(jname, final_ckpt, job_log)
        else:
            diff_result = {"error": "final checkpoint missing"}

        # ── Update status ─────────────────────────────────────────────────────
        with _status_lock:
            _status_global["jobs"][jname].update({
                "status": "done", "end": now_utc(),
                "param_diff": diff_result
            })
            save_status(_status_global)
        log(f"  Job {jname} DONE.", job_log)

    # ── Final hash check ──────────────────────────────────────────────────────
    hash_ok2, hash_msg2 = verify_hashes()
    print(f"\nPost-run hash check: {hash_msg2}")
    with _status_lock:
        _status_global["post_hash"] = {"result": hash_ok2, "msg": hash_msg2, "time": now_utc()}
        save_status(_status_global)

    release_awake()
    print("\nQueue DONE.")

if __name__ == "__main__":
    main()
