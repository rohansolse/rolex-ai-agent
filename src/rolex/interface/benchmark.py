"""Compare provider tool selection on fixed synthetic commands.

Run with python -m rolex.interface.benchmark. No requested tool is executed.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from typing import Any

from rolex.core.config import Config, load_config
from rolex.core.errors import RolexError
from rolex.interface.chat import system_prompt
from rolex.llm.base import LLMProvider, Message, ToolCall, ToolSchema
from rolex.llm.factory import build_provider
from rolex.tools import registered


@dataclass(frozen=True, slots=True)
class Case:
    prompt: str
    expected: tuple[ToolCall, ...] = ()


CASES = (
    Case("What time is it?", (ToolCall("get_time", {}),)),
    Case("How much battery is left?", (ToolCall("battery_level", {}),)),
    Case("How much disk space is free?", (ToolCall("disk_space", {}),)),
    Case("Open Safari.", (ToolCall("open_app", {"name": "Safari"}),)),
    Case("Open https://example.com", (ToolCall("open_url", {"url": "https://example.com"}),)),
    Case("Hello!"),
    Case("What is a Python list?"),
)


def evaluate(provider: LLMProvider, config: Config) -> dict[str, Any]:
    """Measure first-response tool choice and arguments, without acting."""
    schemas = [ToolSchema(tool.name, tool.description, tool.parameters) for tool in registered()]
    rows: list[dict[str, Any]] = []
    for case in CASES:
        start = time.monotonic()
        row: dict[str, Any] = {"prompt": case.prompt}
        try:
            reply = provider.complete(
                [Message("system", system_prompt(config)), Message("user", case.prompt)],
                schemas,
            )
            row["passed"] = reply.tool_calls == case.expected and (
                bool(reply.tool_calls) or bool(reply.content.strip())
            )
            row["calls"] = [call.as_dict() for call in reply.tool_calls]
            row["answer"] = reply.content
        except RolexError as exc:
            row.update(passed=False, error=str(exc))
        row["seconds"] = round(time.monotonic() - start, 3)
        rows.append(row)
    return {
        "provider": provider.name,
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("local", "gemini", "fallback", "all"), default="all")
    args = parser.parse_args(argv)
    config = load_config()
    kinds = ("local", "gemini", "fallback") if args.provider == "all" else (args.provider,)
    reports: list[dict[str, Any]] = []
    for kind in kinds:
        selected = replace(
            config,
            provider="gemini" if kind == "gemini" else "local",
            gemini_fallback=kind == "fallback",
            # An unreachable endpoint exercises fallback without stopping the user's server.
            ollama_host="http://127.0.0.1:1" if kind == "fallback" else config.ollama_host,
        )
        try:
            provider = build_provider(selected)
            if not provider.is_available():
                reports.append({"provider": kind, "unavailable": True})
                continue
            print(f"Evaluating {kind}...", file=sys.stderr, flush=True)
            reports.append(evaluate(provider, selected))
        except RolexError as exc:
            reports.append({"provider": kind, "error": str(exc)})
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return int(any(report.get("passed") != len(CASES) for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
