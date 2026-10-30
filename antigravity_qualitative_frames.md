# Antigravity Instructions: Qualitative Test Video Overlay Frames for Paper

Written 18 Oct 2026. Goal: generate paper-quality overlay frames from the unseen test video `2_Take-1.mp4` using the best model's predicted tracks. No training, no evaluation metrics, no annotation. The output is a set of clean figures showing the tracker working on new footage.

---

## 0. Rules

1. **Do not modify** any existing file: datasets, annotations, checkpoints, configs, existing results, code.
2. New outputs go only under `results/qualitative_paper_figures/` (new folder). If it already exists and is non-empty, stop and ask.
3. No GPU training. Inference on the test video may already be done (`results/manual_test/2_Take_1/tracker/seq.txt`). Check first. Only re-run inference if that file is missing or empty.
4. Run each step as its own script with `python -u`, output to `check/logs/qualitative_<step>.txt`, timeout 300 s. Paste raw output. Label claims **verified** or **assumed**.
5. Re-hash all real IOTD files against `data/IOTD/check/hashes_before.json` at start and end: `changed`, `missing`, `added` must be 0.

---

## Part A. Verify existing inference output

1. Check whether `results/manual_test/2_Take_1/tracker/seq.txt` exists and is non-empty. Print its row count, the frame range (min and max frame index in column 1), the number of distinct track IDs, and the first three and last three rows.
2. Check whether the source frames exist. The video was presumably extracted to `results/manual_test/2_Take_1/seq/img1/` or similar. Print the folder path, the number of JPEG files, and the first and last filename. If frames do not exist, extract them now from `data/IOTD_manual_test/2_Take-1.mp4` using OpenCV sequential `cap.read()`, saving as `000001.jpg` onward at JPEG quality 98 under `results/qualitative_paper_figures/test_frames/img1/`. Print the extracted count.
3. Print the video resolution and FPS from the seqinfo or directly from the video file.

If the tracker output file is missing or has zero rows, run inference using `tools/run_inference_mot_filtered.py` with `configs/train_iotd_real.yaml`, checkpoint `results/train_iotd_real/checkpoint_3.pth`, det threshold 0.5, track threshold 0.5, miss tolerance 30, `use_c1=False`, `use_c2=False`. Output to `results/qualitative_paper_figures/tracker/`. Otherwise use the existing file.

---

## Part B. Select frames for the paper

Good paper figures show: multiple vehicles tracked simultaneously, persistent IDs maintained across at least two consecutive shown frames, a mix of vehicle sizes, and enough scene context to understand the overhead viewpoint. Bad figures show: only one vehicle, a nearly empty road, severe motion blur, or frames where the tracker obviously failed.

1. Parse the tracker output. For each frame, count the number of active tracks. Print a histogram of track counts per frame (binned by 5).
2. Find frames where at least 4 tracks are active simultaneously. From these, pick 6 frames spread across the video (beginning, two middle points, end, and two others). Print the chosen frame indices and their track counts.
3. Also pick 2 consecutive frame pairs (frames N and N+1) where at least 3 tracks appear in both frames, to show ID persistence. Print the chosen pairs.

Total selected: **8 individual frames + 2 pairs = 12 frames maximum.**

---

## Part C. Generate overlay images

Write `tools/qualitative_overlay.py`. For each selected frame:

1. Load the source JPEG.
2. Parse the tracker output for that frame: each row gives `frame, id, x, y, w, h` (columns 1 to 6, 1-based frame index, pixel coordinates).
3. Draw each bounding box with a thick border (line width 3). Use a fixed colour per track ID so the same vehicle has the same colour across frames. Use a colour map with at least 20 distinguishable colours (tab20 or a custom list); IDs beyond 20 wrap around.
4. Draw the track ID as a label in the top-left corner of the box: white text on a filled rectangle matching the box colour. Font size large enough to read at paper resolution (at least 18 pt equivalent).
5. Draw a clean header in the top-left of the image: `"Test video — unseen footage (no ground truth)"` in white text on a dark semi-transparent bar.
6. Do **not** draw confidence scores, class names, frame numbers or any other text. Keep the image clean.
7. Save at full resolution as PNG (lossless) under `results/qualitative_paper_figures/overlays/frame_XXXXXX.png`.

For the 2 consecutive frame pairs, save them side by side as a single wide image (`frame_XXXXXX_pair.png`) with a thin white separator line and the label `"Frame N"` and `"Frame N+1"` above each half. This shows ID persistence directly.

---

## Part D. Generate paper-ready figure composites

Write `tools/qualitative_compose.py`. Produce two composite images saved under `results/qualitative_paper_figures/paper_figures/`:

### fig_qualitative_grid.png (7.0 in wide, 300 dpi)

A 2×3 grid of 6 individual overlay frames. Each cell has a thin white border. No titles inside cells. One shared caption line at the bottom in small text: *"Qualitative tracking results on unseen test footage. Colours indicate persistent track IDs. No ground truth is available for this video."*

Layout: arrange frames in chronological order left to right, top to bottom. Resize each frame to fit the cell while preserving aspect ratio (letterbox with black padding if needed).

### fig_qualitative_pairs.png (7.0 in wide, 300 dpi)

The 2 consecutive frame pairs, each shown as a wide side-by-side image, stacked vertically. Label each pair: `"Pair 1: frames X and X+1"` and `"Pair 2: frames Y and Y+1"` in small text above each pair. Caption at bottom: *"Consecutive frame pairs showing ID persistence. The same colour indicates the same vehicle identity across frames."*

Also produce a **single combined figure** `fig_qualitative_combined.png` (3.4 in wide, 300 dpi) showing the 3 best individual frames stacked vertically, suitable for a single-column figure in the paper if space is tight.

---

## Part E. Figure caption drafts

Print the following ready-to-paste captions. Fill in the actual frame indices and track counts from Part B.

**For fig_qualitative_grid:**
> Fig. X. Qualitative tracking results on an unseen test video not used during training or validation. The model is Arm 1 checkpoint\_3, fine-tuned on IOTD. Each colour represents a unique persistent vehicle identity assigned by the tracker. No ground-truth annotations exist for this footage; quantitative evaluation was performed on the annotated validation sequence (see Table X).

**For fig_qualitative_pairs:**
> Fig. X. Consecutive frame pairs demonstrating identity persistence. The same vehicle receives the same colour and track ID in both frames of each pair, confirming that the tracker maintains consistent identities across frames without ground-truth supervision on this video.

---

## Part F. Quality check

For each composite image:
1. Open it with PIL and confirm pixel dimensions and DPI tag.
2. Confirm it is not blank (mean pixel value > 10).
3. Confirm at least 3 distinct colours are present in the overlay area (uses the track colour map).
4. Print the file size.

Print a summary table:

```
FILE                              | SIZE    | DIMS        | DPI | BLANK? | COLOURS
fig_qualitative_grid.png          | X KB    | WxH         | 300 | No     | N
fig_qualitative_pairs.png         | X KB    | WxH         | 300 | No     | N
fig_qualitative_combined.png      | X KB    | WxH         | 300 | No     | N
```

---



Final message:

```
QUALITATIVE FIGURES COMPLETE

Output folder: results/qualitative_paper_figures/paper_figures/
Figures:
  fig_qualitative_grid.png     (2x3 grid, 6 frames, 7.0 in, 300 dpi)
  fig_qualitative_pairs.png    (2 consecutive pairs, 7.0 in, 300 dpi)
  fig_qualitative_combined.png (3 frames stacked, 3.4 in, 300 dpi)

Source: results/manual_test/2_Take_1/tracker/seq.txt
        (Arm 1 checkpoint_3, det 0.5, track 0.5, miss tol 30)

Video: 2_Take-1.mp4 — unseen footage, no ground truth
IOTD hash check: changed=0, missing=0, added=0

Paste captions from Part E into the paper.
ACTION: Copy paper_figures/ to USB or cloud storage.
```

Do not modify any annotation, checkpoint, config or existing result.
Do not run training or metric evaluation on the test video.
