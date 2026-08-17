import os
import urllib.request
import zipfile

def download_mot17_slice():
    print("Setting up MOT17-02 val sequence slice...")
    target_dir = os.path.join("data", "MOT17", "images", "val", "MOT17-02-FRCNN")
    os.makedirs(os.path.join(target_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "img1"), exist_ok=True)
    
    # Download seqinfo.ini from MOT17 reference repo
    seqinfo_url = "https://raw.githubusercontent.com/JonathonLuiten/TrackEval/master/data/gt/mot_challenge/MOT17-val/MOT17-02-FRCNN/seqinfo.ini"
    gt_url = "https://raw.githubusercontent.com/JonathonLuiten/TrackEval/master/data/gt/mot_challenge/MOT17-val/MOT17-02-FRCNN/gt/gt.txt"
    
    try:
        print("Downloading seqinfo.ini...")
        urllib.request.urlretrieve(seqinfo_url, os.path.join(target_dir, "seqinfo.ini"))
        print("Downloading gt.txt...")
        urllib.request.urlretrieve(gt_url, os.path.join(target_dir, "gt", "gt.txt"))
    except Exception as e:
        print("Error fetching ground truth from TrackEval repo:", e)
        
    # Write val_seqmap.txt
    seqmap_dir = os.path.join("data", "MOT17")
    os.makedirs(seqmap_dir, exist_ok=True)
    with open(os.path.join(seqmap_dir, "val_seqmap.txt"), "w") as f:
        f.write("name\n")
        f.write("MOT17-02-FRCNN\n")
        
    print("MOT17 slice GT setup completed!")

if __name__ == "__main__":
    download_mot17_slice()
