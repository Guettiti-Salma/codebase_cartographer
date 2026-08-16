"""
render.py
---------
Optional last step: convert the generated Mermaid TEXT into an actual
PNG image, using mermaid-cli (`mmdc`) under the hood.

This is deliberately isolated from graph.py/nodes.py and NOT part of the
LangGraph pipeline itself — rendering is a local, deterministic shell-out,
not something that benefits from being a graph node (no looping, no LLM
call, nothing to branch on). Keeping it here also means the core pipeline
still works even if `mmdc` isn't installed; only the picture is skipped.
"""

import subprocess   # to shell out to the `mmdc` command-line tool
import shutil        # to check whether `mmdc` is even on PATH before trying


def mermaid_cli_available() -> bool:
    return shutil.which("mmdc") is not None      # None if mmdc isn't installed / not on PATH


def render_mermaid_to_png(mmd_path: str, png_path: str) -> bool:
    """Renders mmd_path -> png_path. Returns True on success, False on any
    failure (not installed, no headless browser available, timeout, bad
    syntax) — callers should treat False as "skip the image, don't crash."""
    if not mermaid_cli_available():
        return False
    try:
        subprocess.run(
            ["mmdc", "-i", mmd_path, "-o", png_path, "-b", "white"],  # -b white: opaque background, not transparent
            check=True,          # raises CalledProcessError on a non-zero exit code
            capture_output=True,  # swallow mmdc's own logs — we report success/failure ourselves
            timeout=60,           # a stuck headless browser shouldn't hang the whole pipeline
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False              # any failure mode here just means "no image this time"
