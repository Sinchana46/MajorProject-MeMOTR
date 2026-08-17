import os
import urllib.request
import re

def download_gdrive_file(file_id, destination):
    print(f"Downloading Google Drive file {file_id} to {destination}...")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Direct confirmation URL for Google Drive large files
    url = f"https://drive.usercontent.google.com/download?id={file_id}&confirm=t"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
            chunk_size = 1024 * 1024
            downloaded = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                print(f"Downloaded {downloaded / 1024 / 1024:.2f} MB...", end='\r')
        print(f"\nDownload finished! Total size: {os.path.getsize(destination) / 1024 / 1024:.2f} MB")
        return True
    except Exception as e:
        print(f"\nError downloading from {url}: {e}")
        return False

if __name__ == "__main__":
    ckpt_path = os.path.join("checkpoints", "pretrained", "memotr_dancetrack.pth")
    file_id = "1_Xh-TDwwDIeacVEywwlYNvyRmhTKB5K2"
    success = download_gdrive_file(file_id, ckpt_path)
    if not success or os.path.getsize(ckpt_path) < 10000000:
        print("First URL pattern failed or produced small file. Trying secondary pattern...")
        url2 = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
        req = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req) as response, open(ckpt_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Secondary download completed: {os.path.getsize(ckpt_path) / 1024 / 1024:.2f} MB")
