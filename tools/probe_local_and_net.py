"""
Probe using requests library which handles SSL differently.
Also scan all local directories for any pre-existing MOT17 data.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --- 1. Check what's in the MeMOTR TrackEval data directory ---
print("=" * 60)
print("Scanning for pre-existing local GT data...")
print("=" * 60)

roots_to_scan = [
    os.path.join(_ROOT, "TrackEval", "data"),
    os.path.join(_ROOT, "TrackEval"),
    os.path.join(_ROOT, "data"),
]

for root in roots_to_scan:
    print(f"\n--- {root}")
    if os.path.exists(root):
        for dirpath, dirnames, filenames in os.walk(root):
            depth = dirpath.replace(root, '').count(os.sep)
            if depth > 3:
                continue
            indent = "  " * depth
            print(f"{indent}{os.path.basename(dirpath)}/")
            if depth <= 2:
                for f in filenames[:5]:
                    print(f"{indent}  {f}")
    else:
        print("  Not found.")

# --- 2. Try requests library with SSL verify=False ---
print("\n" + "=" * 60)
print("Testing motchallenge.net with requests (SSL verify=False)...")
print("=" * 60)
try:
    import requests
    # Just check the HEAD of the download page
    r = requests.head(
        "https://motchallenge.net/data/MOT17.zip",
        timeout=20,
        verify=False,
        allow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    print(f"Status: {r.status_code}")
    print(f"Headers: {dict(r.headers)}")
except Exception as e:
    print(f"BLOCKER — requests also failed: {type(e).__name__}: {e}")

# --- 3. Try gdown / wget approach via HTTP (not HTTPS) ---
print("\n" + "=" * 60)
print("Testing plain HTTP to motchallenge.net...")
print("=" * 60)
try:
    import requests
    r = requests.head(
        "http://motchallenge.net/data/MOT17.zip",
        timeout=20,
        verify=False,
        allow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    print(f"HTTP Status: {r.status_code}")
    print(f"Content-Length: {r.headers.get('content-length', 'N/A')}")
except Exception as e:
    print(f"BLOCKER — {type(e).__name__}: {e}")
