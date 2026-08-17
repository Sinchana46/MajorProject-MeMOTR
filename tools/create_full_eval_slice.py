import os
import cv2
import numpy as np

def create_slice():
    seq_dir = os.path.abspath(os.path.join("data", "DanceTrack", "val", "dancetrack0001"))
    img_dir = os.path.join(seq_dir, "img1")
    gt_dir = os.path.join(seq_dir, "gt")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    print("Generating 30 frame images (1920x1080)...")
    for frame_idx in range(1, 31):
        img_name = f"{frame_idx:08d}.jpg"
        img_path = os.path.join(img_dir, img_name)
        # Create a textured image with 2 distinct solid rectangles (moving targets)
        img = np.ones((1080, 1920, 3), dtype=np.uint8) * 200
        
        # Target 1 (moving left -> right)
        x1, y1, w1, h1 = 200 + frame_idx * 15, 300, 100, 250
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (50, 50, 200), -1)
        
        # Target 2 (moving right -> left)
        x2, y2, w2, h2 = 1500 - frame_idx * 15, 500, 110, 270
        cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (200, 50, 50), -1)
        
        cv2.imwrite(img_path, img)

    print("Generating gt.txt ground truth file...")
    gt_path = os.path.join(gt_dir, "gt.txt")
    with open(gt_path, "w") as f:
        for frame_idx in range(1, 31):
            x1, y1, w1, h1 = 200 + frame_idx * 15, 300, 100, 250
            f.write(f"{frame_idx},1,{x1},{y1},{w1},{h1},1,1,1.0\n")
            x2, y2, w2, h2 = 1500 - frame_idx * 15, 500, 110, 270
            f.write(f"{frame_idx},2,{x2},{y2},{w2},{h2},1,1,1.0\n")

    # seqinfo.ini
    with open(os.path.join(seq_dir, "seqinfo.ini"), "w") as f:
        f.write("[Sequence]\nname=dancetrack0001\nimDir=img1\nframeRate=20\nseqLength=30\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n")

    # val_seqmap.txt
    val_seqmap = os.path.join("data", "DanceTrack", "val_seqmap.txt")
    with open(val_seqmap, "w") as f:
        f.write("name\ndancetrack0001\n")

    print("Dataset sequence slice successfully generated!")

if __name__ == "__main__":
    create_slice()
