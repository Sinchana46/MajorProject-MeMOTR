"""
run_trackeval_only.py
Runs TrackEval against the already-written prediction file.
Prints all TrackEval output verbatim, then prints pedestrian_summary.txt verbatim.
No fallbacks. No reformatting. If it fails, exact error is shown.
"""
import os, sys

ROOT          = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACKEVAL_DIR = os.path.join(ROOT, "TrackEval")
MOT17_TRAIN   = os.path.join(ROOT, "data", "MOT17", "train")
TRACKER_FOLDER = os.path.join(ROOT, "results", "mot17_eval")
OUTPUT_FOLDER  = os.path.join(ROOT, "results", "mot17_eval_output")
SEQ_NAME       = "MOT17-02-FRCNN"
TRACKER_NAME   = "memotr"
PRED_FILE      = os.path.join(TRACKER_FOLDER, TRACKER_NAME, "data", SEQ_NAME + ".txt")

# Verify prediction file exists and is non-empty
assert os.path.isfile(PRED_FILE), f"Prediction file missing: {PRED_FILE}"
with open(PRED_FILE) as f:
    lines = f.readlines()
print(f"Prediction file  : {PRED_FILE}")
print(f"Prediction lines : {len(lines)}")
print(f"First 3 lines    : {lines[:3]}")

sys.path.insert(0, TRACKEVAL_DIR)
import trackeval

eval_config = {
    "USE_PARALLEL"        : False,
    "NUM_PARALLEL_CORES"  : 1,
    "BREAK_ON_ERROR"      : True,
    "RETURN_ON_ERROR"     : False,
    "LOG_ON_ERROR"        : os.path.join(OUTPUT_FOLDER, "trackeval_error.log"),
    "PRINT_RESULTS"       : True,
    "PRINT_ONLY_COMBINED" : False,
    "PRINT_CONFIG"        : True,
    "TIME_PROGRESS"       : False,
    "DISPLAY_LESS_PROGRESS": False,
    "OUTPUT_SUMMARY"      : True,
    "OUTPUT_EMPTY_CLASSES": True,
    "OUTPUT_DETAILED"     : True,
    "PLOT_CURVES"         : False,
}

dataset_config = {
    "GT_FOLDER"          : MOT17_TRAIN,
    "TRACKERS_FOLDER"    : TRACKER_FOLDER,
    "OUTPUT_FOLDER"      : OUTPUT_FOLDER,
    "TRACKERS_TO_EVAL"   : [TRACKER_NAME],
    "CLASSES_TO_EVAL"    : ["pedestrian"],
    "BENCHMARK"          : "MOT17",
    "SPLIT_TO_EVAL"      : "train",
    "DO_PREPROC"         : True,
    "TRACKER_SUB_FOLDER" : "data",
    "OUTPUT_SUB_FOLDER"  : "",
    "SKIP_SPLIT_FOL"     : True,
    "SEQ_INFO"           : {SEQ_NAME: 600},
    "GT_LOC_FORMAT"      : "{gt_folder}/{seq}/gt/gt.txt",
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

evaluator    = trackeval.Evaluator(eval_config)
metrics_list = [
    trackeval.metrics.HOTA(),
    trackeval.metrics.CLEAR(),
    trackeval.metrics.Identity(),
]
dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

print("\n" + "="*72)
print("TrackEval output (verbatim):")
print("="*72 + "\n")

raw_results, messages = evaluator.evaluate(dataset_list, metrics_list)

print("\n" + "="*72)
print("pedestrian_summary.txt — verbatim:")
print("="*72)
summary = os.path.join(OUTPUT_FOLDER, TRACKER_NAME, "pedestrian_summary.txt")
if os.path.isfile(summary):
    print(f"\nFile: {summary}\n")
    print(open(summary).read())
else:
    print(f"\nBLOCKER: {summary} not found. No metrics reported.")
    for dp, _, fns in os.walk(OUTPUT_FOLDER):
        for fn in fns:
            fp = os.path.join(dp, fn)
            print(f"  {os.path.relpath(fp, OUTPUT_FOLDER)}  ({os.path.getsize(fp)} bytes)")
    sys.exit(1)
