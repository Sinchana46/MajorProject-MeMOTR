import os
import sys
import torch

def redownload():
    ckpt_path = os.path.abspath(os.path.join("checkpoints", "pretrained", "memotr_dancetrack.pth"))
    if os.path.exists(ckpt_path):
        print(f"Removing corrupted/incomplete checkpoint at {ckpt_path}...")
        os.remove(ckpt_path)
        
    file_id = "1_Xh-TDwwDIeacVEywwlYNvyRmhTKB5K2"
    print(f"Downloading MeMOTR pretrained weights from Google Drive ID: {file_id}...")
    
    # Try gdown
    cmd = f'"{sys.executable}" -m gdown {file_id} -O "{ckpt_path}"'
    print(f"Executing: {cmd}")
    res = os.system(cmd)
    
    if os.path.exists(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
        print(f"Downloaded file size: {size_mb:.2f} MB")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            print("VALIDATION SUCCESS: PyTorch checkpoint loaded successfully!")
            print("Keys in checkpoint:", list(ckpt.keys()) if isinstance(ckpt, dict) else "model tensor")
            return True
        except Exception as e:
            print("Validation error:", e)
    else:
        print("Download failed.")
    return False

if __name__ == "__main__":
    redownload()
