
import ssl
import certifi
import sys

# Preserve original SSL function
_original_create_default_context = ssl.create_default_context

# Use Certifi instead of Windows certificate store
def fixed_create_default_context(*args, **kwargs):
    if not any(kwargs.get(k) for k in ("cafile", "capath", "cadata")):
        kwargs["cafile"] = certifi.where()

    return _original_create_default_context(*args, **kwargs)

# Apply SSL workaround before importing Streamlit
ssl.create_default_context = fixed_create_default_context

print("SSL workaround applied successfully.")
print("Starting MeMOTR dashboard...")

from streamlit.web import cli as stcli

if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        "app.py",
        "--server.port=8501"
    ]

    stcli.main()
