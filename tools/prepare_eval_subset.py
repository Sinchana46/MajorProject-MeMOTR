"""
prepare_eval_subset.py — Download and setup a small benchmark slice for MeMOTR evaluation
"""
import os
import sys
import shutil

def setup_dancetrack_slice():
    print("=== Downloading DanceTrack val subset from Hugging Face (noahcao/dancetrack) ===")
    
    # We will use huggingface_hub to download specific files or a single sequence
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import hf_hub_download, snapshot_download

    target_dir = os.path.join("data", "DanceTrack")
    val_dir = os.path.join(target_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    
    # Sequence to download
    seq_name = "dancetrack0001"
    seq_dir = os.path.join(val_dir, seq_name)
    
    if os.path.exists(os.path.join(seq_dir, "gt", "gt.txt")) and os.path.exists(os.path.join(seq_dir, "img1")):
        print(f"Sequence {seq_name} already exists in {seq_dir}.")
    else:
        print(f"Downloading sequence {seq_name}...")
        # Download files matching val/dancetrack0001 pattern
        try:
            snapshot_download(
                repo_id="noahcao/dancetrack",
                repo_type="dataset",
                allow_patterns=[f"val/{seq_name}/*"],
                local_dir=target_dir,
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded {seq_name} to {seq_dir}")
        except Exception as e:
            print(f"Error downloading {seq_name} via snapshot_download: {e}")
            return False

    # Create val_seqmap.txt
    seqmap_path = os.path.join(target_dir, "val_seqmap.txt")
    with open(seqmap_path, "w") as f:
        f.write("name\n")
        f.write(f"{seq_name}\n")
    print(f"Created seqmap at {seqmap_path}")
    
    # Verify contents
    img_files = os.listdir(os.path.join(seq_dir, "img1")) if os.path.exists(os.path.join(seq_dir, "img1")) else []
    gt_file = os.path.join(seq_dir, "gt", "gt.txt")
    print(f"Sequence {seq_name}: {len(img_files)} frames, GT file exists: {os.path.exists(gt_file)}")
    return True

if __name__ == "__main__":
    setup_dancetrack_slice()
