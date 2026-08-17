import os
import sys

def download_gdrive(file_id, output_path):
    print(f"Attempting to download file {file_id} to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Try gdown if installed
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000000:
            print(f"Successfully downloaded with gdown ({os.path.getsize(output_path)} bytes)")
            return True
    except Exception as e:
        print(f"gdown download failed/not available: {e}")

    # Fallback to requests/urllib
    try:
        import urllib.request
        # Google Drive direct download link trick for small/medium files
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        print(f"Trying direct download URL: {url}")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            out_file.write(response.read())
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000000:
            print(f"Successfully downloaded ({os.path.getsize(output_path)} bytes)")
            return True
        else:
            print(f"Downloaded file too small or invalid: {os.path.getsize(output_path) if os.path.exists(output_path) else 0} bytes")
    except Exception as e:
        print(f"Direct download failed: {e}")
        
    return False

if __name__ == "__main__":
    # MeMOTR DanceTrack checkpoint ID from README: 1_Xh-TDwwDIeacVEywwlYNvyRmhTKB5K2
    # MeMOTR MOT17 checkpoint ID from README: 1MPZJfP91Pb1ThnX5dvxZ7tcjDH8t9hew
    dancetrack_id = "1_Xh-TDwwDIeacVEywwlYNvyRmhTKB5K2"
    mot17_id = "1MPZJfP91Pb1ThnX5dvxZ7tcjDH8t9hew"
    
    dt_path = os.path.join("checkpoints", "pretrained", "memotr_dancetrack.pth")
    if not download_gdrive(dancetrack_id, dt_path):
        print("Falling back to MOT17 checkpoint...")
        mot17_path = os.path.join("checkpoints", "pretrained", "memotr_mot17.pth")
        download_gdrive(mot17_id, mot17_path)
