"""The read-only tools that report on the machine."""

import subprocess
from typing import Any

import pytest

from rolex.tools import get
from rolex.tools.system import _read_battery, battery_level, disk_space, machine_info

PMSET_DISCHARGING = """Now drawing from 'Battery Power'
 -InternalBattery-0 (id=12345)\t58%; discharging; 4:11 remaining present: true
"""

PMSET_CHARGED = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=12345)\t100%; charged; 0:00 remaining present: true
"""


def test_battery_percentage_and_state_are_read() -> None:
    assert _read_battery(PMSET_DISCHARGING) == "Battery: 58% (discharging)"


def test_a_charged_battery_reads_correctly() -> None:
    assert _read_battery(PMSET_CHARGED) == "Battery: 100% (charged)"


@pytest.mark.parametrize("output", ["", "no battery here", "Now drawing from 'AC Power'\n"])
def test_output_without_a_percentage_gives_nothing(output: str) -> None:
    assert _read_battery(output) == ""


def test_battery_level_reports_something(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/pmset")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_, **__: subprocess.CompletedProcess([], 0, PMSET_DISCHARGING, ""),
    )
    assert battery_level() == "Battery: 58% (discharging)"


def test_a_mac_without_a_battery_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Mac mini has no battery; that is an answer, not an error."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert "does not report a battery" in battery_level()


def test_battery_survives_pmset_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> None:
        raise subprocess.SubprocessError("pmset exploded")

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/pmset")
    monkeypatch.setattr("subprocess.run", boom)

    assert "does not report a battery" in battery_level()


def test_battery_survives_a_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/pmset")
    monkeypatch.setattr(
        "subprocess.run", lambda *_, **__: subprocess.CompletedProcess([], 1, "", "nope")
    )
    assert "does not report a battery" in battery_level()


def test_disk_space_reports_free_and_total() -> None:
    report = disk_space()
    assert report.startswith("Disk:")
    assert "GB free of" in report


def test_machine_info_reports_the_machine_and_memory() -> None:
    report = machine_info()
    assert "Mac:" in report
    assert "macOS" in report
    assert "Memory:" in report


def test_machine_info_survives_an_unreadable_memory_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(_: str) -> int:
        raise ValueError("unknown name")

    monkeypatch.setattr("os.sysconf", boom)
    assert "Mac:" in machine_info()


@pytest.mark.parametrize("name", ["battery_level", "disk_space", "machine_info"])
def test_each_one_is_read_only_so_it_never_asks(name: str) -> None:
    assert get(name).safety == "read_only"


@pytest.mark.parametrize("name", ["battery_level", "disk_space", "machine_info"])
def test_each_name_says_what_it_answers(name: str) -> None:
    """A 3B model matches on the name. One combined tool was measured to lose
    battery questions to get_time; three specific ones did not."""
    subject = name.split("_")[0]
    assert subject in get(name).description.lower()


PMSET_NO_TAB = " -InternalBattery-0 (id=4653155) 82%; discharging; 4:11 remaining present: true"


def test_the_percentage_is_not_confused_with_the_battery_id() -> None:
    """The id is full of digits, so splitting on a tab that is not there
    reported the id as the charge."""
    assert _read_battery(PMSET_NO_TAB) == "Battery: 82% (discharging)"


def test_a_percentage_with_no_state_still_reads() -> None:
    assert _read_battery(" -InternalBattery-0 (id=1)\t45%") == "Battery: 45%"


def test_a_line_whose_only_digits_are_the_id_is_skipped() -> None:
    assert _read_battery(" -InternalBattery-0 (id=4653155) present: true") == ""
