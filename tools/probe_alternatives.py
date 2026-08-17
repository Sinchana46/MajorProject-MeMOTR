"""
Probe alternative MOT17 sources and also test plain HTTP fallback.
"""
import urllib.request
import urllib.error
import ssl
import os

# Try with SSL verification disabled (to see if it's purely a cert issue)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

URLS_TO_TRY = [
    # motchallenge.net direct (HTTP, no SSL)
    "http://motchallenge.net/data/MOT17.zip",
    # Static hosting alternative commonly used in the MOT community
    "https://opendatalab.com/OpenDataLab/MOT17/tree/main",
    # Check if TrackEval's own data dir has GT for MOT17 locally already
]

print("Testing alternative URL access patterns...\n")

for url in URLS_TO_TRY:
    print(f"--- Probing: {url}")
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
            print(f"  Status: {resp.status}")
            cl = resp.headers.get('Content-Length', 'N/A')
            print(f"  Content-Length: {cl}")
            print(f"  Final URL: {resp.url}")
    except urllib.error.HTTPError as e:
        print(f"  BLOCKER: HTTP {e.code} {e.reason}")
    except urllib.error.URLError as e:
        print(f"  BLOCKER: URLError — {e.reason}")
    except Exception as e:
        print(f"  BLOCKER: {type(e).__name__}: {e}")
    print()

# Also check if the bundled TrackEval repo already has any MOT17 GT locally
_ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
trackeval_dir = os.path.join(_ROOT, "TrackEval")
mot17_gt = os.path.join(trackeval_dir, "data", "gt", "mot_challenge")
print(f"\nChecking for local TrackEval GT data at: {mot17_gt}")
if os.path.exists(mot17_gt):
    entries = os.listdir(mot17_gt)
    print(f"  Found: {entries}")
else:
    print("  Not present.")
