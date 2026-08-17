"""
Attempt to download MOT17 training data (which includes ground truth) from motchallenge.net.
Reports exact status and stops immediately if it cannot complete.
"""
import urllib.request
import urllib.error
import os

# MOT17 official download URL (no login required for training data)
MOT17_URL = "https://motchallenge.net/data/MOT17.zip"
TEST_URL_HEAD = "https://motchallenge.net/data/MOT17.zip"

print(f"Attempting HEAD request to: {MOT17_URL}")

try:
    req = urllib.request.Request(
        MOT17_URL,
        method="HEAD",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(f"Status         : {resp.status}")
        print(f"Content-Length : {resp.headers.get('Content-Length', 'unknown')} bytes")
        print(f"Content-Type   : {resp.headers.get('Content-Type', 'unknown')}")
        print(f"Final URL      : {resp.url}")
        print("=> DOWNLOAD IS ACCESSIBLE. Proceeding to download.")
except urllib.error.HTTPError as e:
    print(f"BLOCKER — HTTP {e.code}: {e.reason}")
    print(f"URL tried: {MOT17_URL}")
except urllib.error.URLError as e:
    print(f"BLOCKER — URLError: {e.reason}")
    print(f"URL tried: {MOT17_URL}")
except Exception as e:
    print(f"BLOCKER — Exception: {type(e).__name__}: {e}")
