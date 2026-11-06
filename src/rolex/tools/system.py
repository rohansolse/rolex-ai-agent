"""Tools that report on the machine.

All read-only, so they run without asking. They answer the questions a model
cannot: how much battery is left, how full the disk is, what this Mac is.

Deliberately three small tools rather than one that reports everything. A
single ``system_info`` was measured against `qwen2.5:3b` and lost: asked about
the battery, the model called ``get_time`` instead, for every phrasing tried.
Split into tools whose names say exactly what they do, it chose correctly
every time. A small model matches on the name, so the name has to be the
answer to the user's question.
"""

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from rolex.tools.registry import register

TIMEOUT = 5.0

GB = 1_000_000_000
"""Decimal gigabytes, matching what macOS shows the user."""


@register(description="Check the battery charge of this Mac.")
def battery_level() -> str:
    """How full the battery is, and whether it is charging."""
    return _battery() or "This Mac does not report a battery."


@register(description="Check how much free disk space this Mac has.")
def disk_space() -> str:
    """Free and total space on the startup disk."""
    usage = shutil.disk_usage(Path("/"))
    return f"Disk: {usage.free / GB:.0f} GB free of {usage.total / GB:.0f} GB"


@register(description="Check what this machine is: its memory, processor and macOS version.")
def machine_info() -> str:
    """What this computer is."""
    version = platform.mac_ver()[0] or platform.release()
    return f"Mac: {platform.machine()}, macOS {version}\n{_memory()}".strip()


def _memory() -> str:
    """Total RAM, from the page size and page count the kernel reports."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        return ""
    return f"Memory: {total / GB:.0f} GB total"


def _battery() -> str:
    """Battery state from ``pmset``, or nothing on a machine without one."""
    executable = shutil.which("pmset")
    if executable is None:
        return ""

    try:
        result = subprocess.run(
            [executable, "-g", "batt"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""

    if result.returncode != 0:
        return ""

    return _read_battery(result.stdout)


#: The percentage in pmset's battery line. Matched directly rather than by
#: splitting on the tab that usually precedes it: the id ahead of it contains
#: digits, so a missing tab would otherwise report the id as the charge.
_PERCENT = re.compile(r"(\d{1,3})%")


def _read_battery(output: str) -> str:
    """Pull the percentage and state out of pmset's output.

    Its second line looks like::

        -InternalBattery-0 (id=1234567)\t82%; discharging; 4:11 remaining
    """
    for line in output.splitlines():
        found = _PERCENT.search(line)
        if found is None:
            continue
        state = _state_after(line, found.end())
        return f"Battery: {found.group(1)}%{f' ({state})' if state else ''}"
    return ""


def _state_after(line: str, position: int) -> str:
    """The word between the percentage and the next semicolon, if any."""
    rest = line[position:].lstrip("; ")
    return rest.split(";")[0].strip()
