"""
run_app.py  —  Launcher that patches the Windows SSL cert-store bug before
               importing streamlit/tornado, then starts the Streamlit server.

Usage (from project root, with memotr_project active):
    python run_app.py
"""
import ssl
import sys

# ── Patch: suppress the ASN1 NOT_ENOUGH_DATA error from the Windows cert store
_orig_load = ssl.SSLContext.load_default_certs
def _safe_load(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        _orig_load(self, purpose)
    except ssl.SSLError:
        pass   # skip corrupt certs silently
ssl.SSLContext.load_default_certs = _safe_load

# ── Now launch Streamlit programmatically ──────────────────────────────────────
from streamlit.web import cli as stcli

sys.argv = [
    "streamlit", "run", "app.py",
    "--server.port",             "8501",
    "--server.headless",         "true",
    "--browser.gatherUsageStats","false",
    "--server.fileWatcherType",  "none",
]

if __name__ == "__main__":
    sys.exit(stcli.main())
