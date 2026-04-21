"""Run the Streamlit app from Google Colab with an ngrok tunnel.

Usage in a Colab cell:

    !pip install -q pyngrok
    # First time only: paste your free ngrok auth token from https://dashboard.ngrok.com
    import os; os.environ['NGROK_AUTHTOKEN'] = 'YOUR_TOKEN_HERE'
    !python scripts/colab_streamlit.py

The script:
  1. Starts the Streamlit server on port 8501 in the background.
  2. Opens an ngrok tunnel and prints the public URL.
  3. Streams the Streamlit logs until interrupted.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--app", default=str(_REPO_ROOT / "app" / "streamlit_app.py"))
    args = p.parse_args()

    try:
        from pyngrok import ngrok  # noqa: PLC0415
    except ImportError:
        sys.exit("pyngrok is required. Install with `pip install -q pyngrok` first.")

    token = os.environ.get("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)

    # Start Streamlit in the background.
    cmd = [
        "streamlit", "run", args.app,
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    print("→", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(4)

    # Open the tunnel.
    tunnel = ngrok.connect(args.port, "http")
    print(f"\n=== Streamlit is live ===\n→ {tunnel.public_url}\n")

    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(line.decode(errors="replace"))
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        proc.terminate()
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()


if __name__ == "__main__":
    main()
