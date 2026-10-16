"""
tools/tune_eval_smoke.py — Tuning Stage 1 pre-launch smoke test (read-only on existing results)

1. Evaluates Arm 1 checkpoint_3 through the queue's own run_inference / run_trackeval /
   read_te_summary functions and compares with the baseline registry
   (Arm1 ckpt_3: HOTA 50.47 DetA 46.96 AssA 54.51 MOTA 29.63 IDF1 70.23 IDSW 9 FP 2629 FN 305).
2. Runs the queue's param_diff_check on Arm 1 checkpoint_4 (reference drift for lr_half).
3. Counts parameters per optimizer group from the Phase 5 checkpoint keys using the same
   keyword rules as MeMOTR/train_engine.py:get_param_groups.

Outputs only under results/tuning_stage1/smoke/.
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune_queue as q

OUT = os.path.join(q.STAGE_DIR, "smoke", "arm1")
LOG = os.path.join(q.STAGE_DIR, "smoke", "smoke.log")
REF = {"HOTA": 50.47, "DetA": 46.96, "AssA": 54.51, "MOTA": 29.63, "IDF1": 70.23,
       "IDSW": 9, "FP": 2629, "FN": 305}

def main():
    if os.path.exists(os.path.join(OUT, "ckpt_3")):
        print(f"REFUSE: {OUT}/ckpt_3 already exists"); sys.exit(1)
    os.makedirs(OUT, exist_ok=True)

    # 3. optimizer group parameter counts (from checkpoint keys, same keyword rules)
    import torch
    sd = torch.load(q.PHASE5_CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    groups = {"lr_backbone": 0, "lr_points": 0, "lr_query_updater": 0, "lr": 0}
    names = {k: [] for k in groups}
    for k, v in sd.items():
        if not torch.is_floating_point(v):
            continue
        if "backbone.backbone" in k: g = "lr_backbone"
        elif "reference_points" in k or "sampling_offsets" in k: g = "lr_points"
        elif "query_updater" in k: g = "lr_query_updater"
        else: g = "lr"
        groups[g] += v.numel(); names[g].append(k)
    print("Optimizer groups (from Phase5 ckpt4 state_dict, floating tensors incl. any buffers):")
    for g in groups:
        print(f"  {g:18s} tensors={len(names[g]):4d}  elements={groups[g]:>11,d}  e.g. {names[g][:2]}")
    print(f"  query_updater tensors: {names['lr_query_updater']}")

    # 1. evaluation smoke
    t0 = time.time()
    trk = q.run_inference("arm1", os.path.join(q.ROOT, "results", "train_iotd_real", "checkpoint_3.pth"),
                          3, OUT, LOG)
    t_inf = time.time() - t0
    print(f"inference: {trk}  ({t_inf:.0f}s)")
    if trk is None: sys.exit(1)
    t1 = time.time()
    summ = q.run_trackeval(trk, OUT, "arm1", 3, LOG)
    print(f"trackeval: {summ}  ({time.time()-t1:.0f}s)")
    if summ is None: sys.exit(1)
    m = q.read_te_summary(summ)
    ok = True
    print(f"{'metric':6s} {'smoke':>9s} {'registry':>9s} {'diff':>8s}")
    for k, ref in REF.items():
        d = m[k] - ref
        flag = "" if abs(d) <= 0.011 else "  <-- MISMATCH"
        ok &= flag == ""
        print(f"{k:6s} {m[k]:9.2f} {ref:9.2f} {d:8.3f}{flag}")
    print("SMOKE EVAL:", "PASS (matches registry within 0.01)" if ok else "FAIL")
    print(f"inference seconds per checkpoint: {t_inf:.0f}")

    # 2. Arm 1 drift reference
    diff = q.param_diff_check("arm1", os.path.join(q.ROOT, "results", "train_iotd_real", "checkpoint_4.pth"), LOG)
    print("Arm1 ckpt4 param diff vs Phase5:", json.dumps(diff))
    with open(os.path.join(q.STAGE_DIR, "smoke", "smoke_result.json"), "w") as f:
        json.dump({"eval_pass": ok, "metrics": m, "inference_s": t_inf, "arm1_param_diff": diff,
                   "group_elements": groups}, f, indent=2)

if __name__ == "__main__":
    main()
