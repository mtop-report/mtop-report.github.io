"""`python -m mtop studio`: the local data studio (FNO only, never part of the public app).

Starts the Streamlit app in src/mtop/studio_app.py on this computer only (127.0.0.1), behind the
studio password. Usage statistics of Streamlit are switched off. Needs `pip install -e ".[studio]"`.

  python -m mtop studio --set-password     set or change the password (stored as a scrypt hash)
  python -m mtop studio                    open the studio (default port 8765)
  python -m mtop studio --input other.xlsx use another registry export
"""
from __future__ import annotations

import getpass
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import paths
from .corrections import password_file, password_is_set, set_password

APP = Path(__file__).with_name("studio_app.py")


def _set_password() -> None:
    while True:
        a = getpass.getpass("New studio password (at least 8 characters): ")
        b = getpass.getpass("Repeat it: ")
        if a != b:
            print("The two entries differ, try again.")
            continue
        try:
            set_password(a)
        except ValueError as e:
            print(e)
            continue
        print(f"Password saved (hash only) in {password_file()}")
        return


def main(args) -> None:
    if args.set_password:
        _set_password()
        return
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit('The studio needs Streamlit: pip install -e ".[studio]"')
    if not password_is_set():
        raise SystemExit("Set a studio password first: python -m mtop studio --set-password")
    src = Path(args.input) if args.input else paths.DEFAULT_RAW_FILE
    if not src.exists():
        raise SystemExit(f"Registry export not found: {src}")
    env = dict(os.environ, MTOP_STUDIO_INPUT=str(src.resolve()), PYTHONPATH=os.pathsep.join(
        [str(paths.ROOT / "src")] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    url = f"http://127.0.0.1:{args.port}"
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP),
           "--server.address", "127.0.0.1", "--server.port", str(args.port), "--server.headless", "true",
           "--browser.gatherUsageStats", "false", "--server.fileWatcherType", "none",
           "--theme.base", "light", "--client.toolbarMode", "minimal"]
    print(f"MTOP data studio on {url} (this computer only). Stop with Ctrl+C.")
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(2.5), webbrowser.open(url)), daemon=True).start()
    subprocess.run(cmd, env=env, check=False)
