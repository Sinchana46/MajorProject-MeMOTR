import os
import urllib.request
import json

def download_hf_seq(seq_name="dancetrack0001"):
    base_url = f"https://huggingface.co/datasets/noahcao/dancetrack/raw/main/val/{seq_name}"
    target_dir = os.path.join("data", "DanceTrack", "val", seq_name)
    os.makedirs(os.path.join(target_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "img1"), exist_ok=True)
    
    print(f"Downloading {seq_name} metadata and gt.txt...")
    # seqinfo.ini
    try:
        urllib.request.urlretrieve(f"{base_url}/seqinfo.ini", os.path.join(target_dir, "seqinfo.ini"))
    except Exception as e:
        print("seqinfo.ini error:", e)
        
    # gt/gt.txt
    try:
        urllib.request.urlretrieve(f"{base_url}/gt/gt.txt", os.path.join(target_dir, "gt", "gt.txt"))
    except Exception as e:
        print("gt.txt error:", e)

    # seqmap
    seqmap_path = os.path.join("data", "DanceTrack", "val_seqmap.txt")
    with open(seqmap_path, "w") as f:
        f.write("name\n")
        f.write(f"{seq_name}\n")

    print(f"Downloading images for {seq_name}...")
    # Download first 100 frames for quick evaluation
    img_url_base = f"https://huggingface.co/datasets/noahcao/dancetrack/resolve/main/val/{seq_name}/img1"
    for i in range(1, 151): # 150 frames
        img_name = f"{i:08d}.jpg"
        img_path = os.path.join(target_dir, "img1", img_name)
        if not os.path.exists(img_path):
            try:
                urllib.request.urlretrieve(f"{img_url_base}/{img_name}", img_path)
            except Exception as e:
                print(f"Failed frame {img_name}: {e}")
                break
        print(f"Downloaded {i}/150 frames...", end='\r')
    print(f"\nSequence {seq_name} setup completed!")

if __name__ == "__main__":
    download_hf_seq("dancetrack0001")
