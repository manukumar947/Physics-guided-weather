"""
serve.py  —  Topo-Downscale SIH Demo Launcher
===============================================
Builds the React dashboard (if needed) and starts the FastAPI server.
The dashboard and the API are served from a single origin on port 8000.

Usage
-----
    python scripts/serve.py              # build + serve on :8000
    python scripts/serve.py --skip-build # skip npm build (use existing dist/)
    python scripts/serve.py --port 9000  # use a different port
    python scripts/serve.py --reload     # enable uvicorn hot-reload (dev mode)

What this does
--------------
1. Checks that Node/npm and Python packages are available.
2. Runs  `npm run build`  inside dashboard/ to compile the React app.
3. Starts  `uvicorn src.api.main:app`  at the repo root.
4. The built dist/ folder is served as static files by FastAPI.
5. Open  http://localhost:8000  in a browser — the full stack is live.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parent.parent
DASHBOARD_DIR  = REPO_ROOT / "dashboard"
DIST_DIR       = DASHBOARD_DIR / "dist"
INDEX_HTML     = DIST_DIR / "index.html"


def _run(cmd: list[str], cwd: Path, label: str) -> None:
    print(f"\n[build] {label}")
    print(f"[build] $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n[ERROR] '{label}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)


def build_dashboard() -> None:
    """Install npm deps (if needed) and build the production bundle."""
    npm = shutil.which("npm")
    if npm is None:
        print("[ERROR] npm not found on PATH. Install Node.js >= 18 and retry.")
        sys.exit(1)

    node_modules = DASHBOARD_DIR / "node_modules"
    if not node_modules.exists():
        _run([npm, "install"], cwd=DASHBOARD_DIR, label="npm install")
    else:
        print("[build] node_modules already present — skipping npm install.")

    _run([npm, "run", "build"], cwd=DASHBOARD_DIR, label="npm run build")

    if not INDEX_HTML.exists():
        print(f"[ERROR] Build succeeded but {INDEX_HTML} not found.")
        sys.exit(1)

    print(f"[build] Dashboard built successfully → {DIST_DIR}")


def start_server(port: int, reload: bool) -> None:
    """Start FastAPI + static dashboard on the given port."""
    uvicorn = shutil.which("uvicorn") or f"{sys.executable} -m uvicorn"
    cmd_str = (
        f"{sys.executable} -m uvicorn src.api.main:app "
        f"--host 0.0.0.0 --port {port}"
        + (" --reload" if reload else "")
    )
    print(f"\n{'='*55}")
    print(f"  Topo-Downscale is starting on http://localhost:{port}")
    print(f"  API docs      → http://localhost:{port}/docs")
    print(f"  Dashboard     → http://localhost:{port}/")
    try:
        subprocess.run(cmd_str, cwd=REPO_ROOT, shell=True)
    except KeyboardInterrupt:
        print("\n[INFO] Topo-Downscale server gracefully stopped.")
        sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Topo-Downscale demo launcher")
    parser.add_argument("--port",        type=int,  default=8000)
    parser.add_argument("--skip-build",  action="store_true",
                        help="Skip npm build and use existing dist/")
    parser.add_argument("--reload",      action="store_true",
                        help="Enable uvicorn auto-reload (development only)")
    args = parser.parse_args()

    if args.skip_build:
        if not INDEX_HTML.exists():
            print(f"[ERROR] --skip-build specified but {INDEX_HTML} does not exist.")
            print("       Run without --skip-build to trigger a fresh build.")
            sys.exit(1)
        print(f"[build] Skipping build — using existing {DIST_DIR}")
    else:
        build_dashboard()

    start_server(port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
