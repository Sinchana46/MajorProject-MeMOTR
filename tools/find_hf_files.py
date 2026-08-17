from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files("noahcao/dancetrack", repo_type="dataset")
print("Total files in noahcao/dancetrack:", len(files))
print("First 20 files:", files[:20])
val_files = [f for f in files if "dancetrack0001" in f]
print("dancetrack0001 files:", val_files[:10])
