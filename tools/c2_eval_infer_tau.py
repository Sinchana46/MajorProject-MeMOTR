"""
tools/c2_eval_infer_tau.py — runs tools/run_inference_mot.py unchanged (its own main(), same CLI arguments)
in-process, with two read-only hooks that record, per frame, the scene vector from SceneAnalyzer and the
RuntimeTracker.tau_adaptive value after each update(). Writes <output-dir>/tau_per_frame.csv.
The hooks only observe return values; they do not change any input or output of the tracker.

Usage: same arguments as tools/run_inference_mot.py.
"""
import os, sys, csv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_inference_mot as rim                       # sets up MeMOTR / project paths
import models.runtime_tracker as rt
import modules.scene_analyzer as sa

RECORDS, _last_scene = [], [None]

_orig_scene = sa.SceneAnalyzer.forward
def _scene(self, *a, **k):
    v = _orig_scene(self, *a, **k)
    _last_scene[0] = [float(x) for x in v.tolist()]
    return v
sa.SceneAnalyzer.forward = _scene

_orig_update = rt.RuntimeTracker.update
def _update(self, *a, **k):
    _last_scene[0] = None
    out = _orig_update(self, *a, **k)
    s = _last_scene[0] or [None, None, None]
    RECORDS.append({"frame": len(RECORDS) + 1, "use_c2": self.use_c2, "tau": self.tau_adaptive,
                    "brightness": s[0], "contrast": s[1], "density": s[2]})
    return out
rt.RuntimeTracker.update = _update

if __name__ == "__main__":
    rim.main()
    out_dir = sys.argv[sys.argv.index("--output-dir") + 1]
    with open(os.path.join(out_dir, "tau_per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "use_c2", "tau", "brightness", "contrast", "density"])
        w.writeheader(); w.writerows(RECORDS)
    taus = [r["tau"] for r in RECORDS]
    print(f"tau over {len(taus)} frames: mean={sum(taus)/len(taus):.4f} min={min(taus):.4f} max={max(taus):.4f}")
