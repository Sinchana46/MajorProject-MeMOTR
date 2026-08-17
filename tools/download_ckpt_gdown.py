import os
import sys
import torch
import gdown

def main():
    ckpt_path = os.path.abspath(os.path.join("checkpoints", "pretrained", "memotr_dancetrack.pth"))
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    file_id = "1_Xh-TDwwDIeacVEywwlYNvyRmhTKB5K2"
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from {url} to {ckpt_path} using gdown.download()...")
    
    output = gdown.download(id=file_id, output=ckpt_path, quiet=False, fuzzy=True)
    print("gdown output returned:", output)

    if os.path.exists(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
        print(f"File downloaded! Size: {size_mb:.2f} MB")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            print("VALIDATION SUCCESS: PyTorch checkpoint file loaded cleanly!")
            print("Keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else "model")
        except Exception as e:
            print("Validation error:", e)
    else:
        print("Download failed.")

if __name__ == "__main__":
    main()
