"""
Extract representative frames from each tracked output video for qualitative review.
Saves frames at regular intervals and at key moments (active-track changes).
"""
import cv2, os

ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR   = os.path.join(ROOT, "results", "iotd_qualitative")
FRAMES_DIR   = os.path.join(OUTPUT_DIR, "sample_frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

clips = {
    "day-1_tracked.mp4":   [1, 200, 600, 1000, 1400, 1600, 2000, 2400, 3000, 3600, 3820],
    "night-1_tracked.mp4": [1, 100, 200, 400, 600, 800, 1000, 1400, 1800, 1976],
}

for clip_name, frame_indices in clips.items():
    clip_path = os.path.join(OUTPUT_DIR, clip_name)
    stem      = clip_name.replace("_tracked.mp4", "")
    cap       = cv2.VideoCapture(clip_path)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n{clip_name}  ({total} frames)")

    fi_set = set(frame_indices)
    idx    = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx in fi_set:
            out = os.path.join(FRAMES_DIR, f"{stem}_f{idx:04d}.jpg")
            cv2.imwrite(out, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  Saved frame {idx:04d} -> {out}")
    cap.release()

print("\nDone.")
