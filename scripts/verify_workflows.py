"""Run A-E through run_chat with the configured real local model.

Messaging uses an explicit deterministic fixture. --live-tabs adds and removes
only newly created about:blank tabs in the user's existing Chrome session.
"""

import argparse
import builtins
import contextlib
import io
import json
import sys
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from rolex.agent.workflow import NativeBackend  # noqa: E402
from rolex.core.config import load_config  # noqa: E402
from rolex.interface.chat import run_chat  # noqa: E402
from rolex.llm.base import Message, Reply, ToolSchema  # noqa: E402
from rolex.llm.ollama import OllamaProvider  # noqa: E402
from rolex.tools import browser as chrome  # noqa: E402
from workflow_fixture import FixtureBrowser  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--live-tabs", action="store_true")
parser.add_argument("--workflows", default="", help="Comma-separated workflow names to rerun")
parser.add_argument("--output", type=Path, default=Path("/private/tmp/rolex-workflows.json"))
args = parser.parse_args()
config = replace(load_config(), memory=False, gemini_fallback=False, provider="local")


class MeasuredProvider:
    def __init__(self) -> None:
        self.inner = OllamaProvider(config)
        self.generations = 0

    @property
    def name(self) -> str:
        return self.inner.name

    def is_available(self) -> bool:
        return self.inner.is_available()

    def structured(self, messages: list[Message], schema: dict[str, Any]) -> dict[str, Any]:
        self.generations += 1
        return self.inner.structured(messages, schema)

    def complete(self, messages: list[Message], tools: list[ToolSchema] | None = None) -> Reply:
        self.generations += 1
        return self.inner.complete(messages, tools)

    def stream(
        self, messages: list[Message], tools: list[ToolSchema] | None = None
    ) -> Iterator[str]:
        self.generations += 1
        yield from self.inner.stream(messages, tools)


provider = MeasuredProvider()
results: list[dict[str, Any]] = []


def exercise(
    name: str,
    backend: Any,
    instructions: list[str],
    expected: str,
    approvals: list[str] | None = None,
    before_input: Any = None,
) -> None:
    if args.workflows and name not in args.workflows.split(","):
        results.append({"workflow": name, "skipped": True})
        return
    inputs = iter([*instructions, "exit"])
    answers = iter(approvals or ["y"] * 50)
    transcript = io.StringIO()
    started = time.monotonic()
    generations_before = provider.generations

    def input_line(prompt: str) -> str:
        if "[y/N]" in prompt:
            return next(answers)
        line = next(inputs)
        if before_input:
            before_input(line)
        return line

    initial: dict[str, Any] = (
        {"count": backend.count}
        if isinstance(backend, FixtureBrowser)
        else {"window_and_count": backend.tabs()}
    )
    print(f"Running {name}...", flush=True, file=sys.stderr)
    with (
        patch("rolex.agent.workflow.NativeBackend", return_value=backend),
        patch.object(builtins, "input", side_effect=input_line),
        contextlib.redirect_stdout(transcript),
    ):
        code = run_chat(config, provider=provider, tools=[])
    if isinstance(backend, FixtureBrowser):
        observed = {
            "tabs": backend.count,
            "recipient_id": backend.current.conversation_id,
            "draft": backend.current.draft,
            "writes": backend.writes,
            "sent": backend.sent,
        }
    else:
        observed = {"window_and_count": backend.tabs()}
    results.append(
        {
            "workflow": name,
            "model": config.model,
            "model_generations": provider.generations - generations_before,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "environment": "live Chrome"
            if not isinstance(backend, FixtureBrowser)
            else "configured local provider + deterministic messaging fixture",
            "starting_state": initial,
            "input": instructions,
            "expected": expected,
            "observed": observed,
            "exit": code,
            "transcript": transcript.getvalue(),
        }
    )
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))


native = NativeBackend()
if args.live_tabs:
    window, before = native.tabs()
    try:
        exercise(
            "A",
            native,
            ["How many tabs are open?", "आणखी दोन टॅब उघड.", "किती टॅब उघडे आहेत?"],
            f"window {window}: {before} -> {before + 2}",
        )
        observed = int(
            chrome._osascript(
                f'tell application "Google Chrome" to count tabs of window id {window}'
            )
        )
        results[-1]["independent_tab_count"] = observed
        results[-1]["passed"] = observed == before + 2
    finally:
        # Close only IDs returned by our tab creation operation, and only if still blank.
        for tab_id in native.created:
            chrome._osascript(
                'tell application "Google Chrome"\n'
                "repeat with w in windows\nrepeat with t in tabs of w\n"
                f'if (id of t as text) = "{tab_id}" and URL of t = "about:blank" then\n'
                "close t\nexit repeat\nend if\nend repeat\nend repeat\nend tell"
            )
else:
    fixture = FixtureBrowser()
    exercise("A", fixture, ["How many tabs are open?", "Open four more tabs"], "3 -> 7 tabs")
    results[-1]["passed"] = fixture.count == 7

fixture = FixtureBrowser()
exercise(
    "B",
    fixture,
    ["WhatsApp वर जा.", "त्यात निशा शाहचं चॅट उघड.", "त्याला लिही, मी १२ मिनिटांत पोहोचतो."],
    "Nisha fixture, exact unsent Marathi draft",
)
results[-1]["passed"] = fixture.current.draft == "मी १२ मिनिटांत पोहोचतो." and not fixture.sent

fixture = FixtureBrowser()
payload = "Do not send ₹2,750 today"
exercise(
    "C",
    fixture,
    [f"WhatsApp kholo, Nisha Shah ka chat open karo aur type karo '{payload}'"],
    "Nisha fixture, exact payload, zero sends",
)
results[-1]["passed"] = fixture.current.draft == payload and not fixture.sent

fixture = FixtureBrowser()


def manual_change(line: str) -> None:
    if line.startswith("त्याला लिही"):
        fixture.current = replace(fixture.current, conversation_id="sam-work", draft="")


exercise(
    "D",
    fixture,
    ["Open WhatsApp, open Sam chat and type 'Meet at gate 9'", "2", "त्याला लिही, Call at 8."],
    "Choose home fixture, draft once; manual same-name change blocks follow-up typing",
    before_input=manual_change,
)
results[-1]["passed"] = fixture.writes == [("sam-home", "Meet at gate 9")] and not fixture.sent

fixture = FixtureBrowser()
exercise(
    "C-paraphrase",
    fixture,
    [
        "Get WhatsApp in front, locate Nisha Shah's conversation, and enter "
        "'Please send the revised invoice at 19:25'",
    ],
    "Alternate source-grounded wording; exact unsent draft",
)
results[-1]["passed"] = (
    fixture.current.draft == "Please send the revised invoice at 19:25" and not fixture.sent
)

fixture = FixtureBrowser()
exercise("model", fixture, ["Hello"], "A real local-model greeting without browser mutation")
results[-1]["passed"] = (
    results[-1].get("model_generations", 0) >= 1 and not fixture.writes and not fixture.sent
)

for variant in ("send", "deny", "cancel", "lost_ack", "unknown"):
    fixture = FixtureBrowser()
    fixture.submit_then_fail = variant == "lost_ack"
    fixture.lose_ack = variant == "unknown"
    controls = ["cancel"] if variant == "cancel" else ["पाठव."]
    if variant in ("lost_ack", "unknown"):
        controls.append("send it")
    exercise(
        "E-" + variant,
        fixture,
        [
            "WhatsApp kholo, Nisha Shah ka chat open karo aur type karo 'Ready at 18:40'",
            *controls,
        ],
        "Exactly one send for approved/submitted cases, zero otherwise",
        ["y", "y", "y", "n" if variant == "deny" else "y"],
    )
    results[-1]["passed"] = len(fixture.sent) == int(variant in {"send", "lost_ack"})

results = [row for row in results if not row.get("skipped")]
args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(
    json.dumps(
        [
            {"workflow": row["workflow"], "passed": row.get("passed"), "observed": row["observed"]}
            for row in results
        ],
        ensure_ascii=False,
        indent=2,
    )
)
raise SystemExit(int(not all(row.get("passed") for row in results)))
