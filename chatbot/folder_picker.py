"""Native folder selection dialog for local Streamlit sessions."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path


def _pick_folder_macos(initial_dir: str | None) -> str | None:
    prompt = "Select documents folder"
    if initial_dir and Path(initial_dir).is_dir():
        script = (
            f'POSIX path of (choose folder with prompt "{prompt}" '
            f'default location (POSIX file "{initial_dir}"))'
        )
    else:
        script = f'POSIX path of (choose folder with prompt "{prompt}")'

    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    folder = result.stdout.strip()
    return folder or None


def _pick_folder_tk(initial_dir: str | None) -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update_idletasks()
    folder = filedialog.askdirectory(
        initialdir=initial_dir if initial_dir and Path(initial_dir).is_dir() else None,
        mustexist=True,
        title="Select documents folder",
    )
    root.destroy()
    return folder or None


def pick_folder(initial_dir: str | None = None) -> str | None:
    """Open the OS folder picker and return the selected path."""
    initial = initial_dir.strip() if initial_dir else None

    if platform.system() == "Darwin":
        return _pick_folder_macos(initial)

    try:
        return _pick_folder_tk(initial)
    except Exception:
        return None
