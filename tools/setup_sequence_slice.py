import os
import sys
import zipfile
import urllib.request

def download_dancetrack_slice():
    print("Downloading DanceTrack sequence using huggingface_hub...")
    try:
        from huggingface_hub import snapshot_download
        local_dir = os.path.join("data", "DanceTrack")
        # Download val/dancetrack0001
        snapshot_download(
            repo_id="noahcao/dancetrack",
            repo_type="dataset",
            allow_patterns=["val/dancetrack0001/*"],
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print("Downloaded files successfully.")
        
        # Write val_seqmap.txt
        seqmap_path = os.path.join(local_dir, "val_seqmap.txt")
        with open(seqmap_path, "w") as f:
            f.write("name\n")
            f.write("dancetrack0001\n")
        print("Created val_seqmap.txt")
        return True
    except Exception as e:
        print("HuggingFace download failed:", e)

    print("Falling back to downloading MOT17-02 ground truth and synthetic frames for quick validation...")
    return False

if __name__ == "__main__":
    download_dancetrack_slice()
