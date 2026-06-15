from __future__ import annotations
import re
from typing import Optional

def _free_ram_bytes() -> Optional[int]:
    """Best-effort reading of currently-available system RAM. Returns
    ``None`` when we can't tell, in which case the caller should treat the
    answer as "don't know — proceed".

    Uses ``psutil`` when available; falls back to ``vm_stat`` on macOS.
    Never raises.
    """
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    # macOS fallback: parse vm_stat. Cheap; runs in <2ms typically.
    try:
        import subprocess

        out = subprocess.check_output(["vm_stat"], timeout=1.0).decode("utf-8")
        page_size = 16384  # default on Apple Silicon; refined below
        free_pages = 0
        inactive_pages = 0
        for line in out.splitlines():
            if "page size of" in line:
                # "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
                m = re.search(r"page size of (\d+) bytes", line)
                if m:
                    page_size = int(m.group(1))
            elif line.startswith("Pages free:"):
                free_pages = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages inactive:"):
                inactive_pages = int(line.split(":")[1].strip().rstrip("."))
        return (free_pages + inactive_pages) * page_size
    except Exception:
        return None

