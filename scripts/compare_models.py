"""Compare real local models on synthetic Rolex requests; never execute tools.

Unlike workflow fixtures, intent cases deliberately bypass the deterministic
extractor so every case measures an actual model generation. Results distinguish
raw model correctness from application validation. Models run sequentially.
"""

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from rolex.agent.intent import INSTRUCTIONS, SCHEMA, Step, validate
from rolex.core.config import load_config
from rolex.core.errors import RolexError
from rolex.interface.chat import system_prompt
from rolex.llm.base import Message, ToolCall, ToolSchema
from rolex.llm.ollama import OllamaProvider
from rolex.llm.transport import UrllibTransport
from rolex.tools import registered


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    expected: tuple[Step, ...]


INTENTS = (
    Case("count-en", "How many tabs are open?", (Step("count_tabs"),)),
    Case("add-mr", "आणखी तीन टॅब उघड.", (Step("open_tabs", 3),)),
    Case("add-roman", "char navin tabs open kara", (Step("open_tabs", 4),)),
    Case("focus-hi", "WhatsApp पर जाओ।", (Step("focus_whatsapp"),)),
    Case(
        "select-mr",
        "त्यात Leena Kulkarni चॅट उघड.",
        (Step("select_chat", recipient="Leena Kulkarni"),),
    ),
    Case(
        "draft-mr",
        "त्याला लिही, 'मी १७ मिनिटांत येतो.\nकृपया थांबा.'",
        (Step("draft", text="मी १७ मिनिटांत येतो.\nकृपया थांबा."),),
    ),
    Case(
        "compound-mixed",
        "WhatsApp kholo, Dev Mehta ka chat open karo aur type karo 'Please send ₹3,460 tomorrow'",
        (
            Step("focus_whatsapp"),
            Step("select_chat", recipient="Dev Mehta"),
            Step("draft", text="Please send ₹3,460 tomorrow"),
        ),
    ),
    Case(
        "compound-en",
        'Go to WhatsApp, select Tara Bose conversation and write "Don\'t send it before 21:15."',
        (
            Step("focus_whatsapp"),
            Step("select_chat", recipient="Tara Bose"),
            Step("draft", text="Don't send it before 21:15."),
        ),
    ),
    Case("send-mr", "पाठव.", (Step("send"),)),
    Case("conversation", "Why is the sky blue?", (Step("chat"),)),
    Case("missing-count", "Add a few more tabs", (Step("clarify"),)),
)
TOOLS = (
    ("time-mr", "आत्ता किती वाजले आहेत?", (ToolCall("get_time", {}),)),
    ("battery-roman", "battery kiti baki ahe?", (ToolCall("battery_level", {}),)),
    ("disk-hi", "डिस्क में कितनी खाली जगह है?", (ToolCall("disk_space", {}),)),
    ("app-en", "Open Safari.", (ToolCall("open_app", {"name": "Safari"}),)),
    ("greeting", "Hello!", ()),
)


class MeasuredTransport(UrllibTransport):
    def __init__(self) -> None:
        super().__init__()
        self.latest: dict[str, Any] = {}

    def post_json(self, url: str, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        # Same generation cap for every candidate. Intent settings already come
        # from production; bound general-tool output/context for this benchmark.
        body.setdefault("options", {}).setdefault("num_predict", 256)
        body["options"].setdefault("num_ctx", 4096)
        result = super().post_json(url, body, timeout=timeout)
        self.latest = result
        return result


def equivalent(actual: list[Step], expected: tuple[Step, ...]) -> bool:
    if len(actual) != len(expected):
        return False
    return all(
        a.action == e.action
        and a.count == e.count
        and a.recipient.casefold().strip() == e.recipient.casefold()
        and (a.action == "clarify" or a.text == e.text)
        for a, e in zip(actual, expected, strict=True)
    )


def raw_steps(raw: dict[str, Any]) -> list[Step]:
    return [
        Step(
            item["action"],
            int(item.get("value", "0")) if item["action"] == "open_tabs" else 0,
            item.get("value", "") if item["action"] == "select_chat" else "",
            item.get("value", "") if item["action"] in {"draft", "clarify"} else "",
        )
        for item in raw["steps"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["qwen3:4b", "qwen3.5:4b-q4_K_M", "granite4:3b"]
    )
    parser.add_argument("--output", type=Path, default=Path("verification/model-comparison.json"))
    parser.add_argument("--wait-downloads", action="store_true")
    args = parser.parse_args()
    config = replace(load_config(), provider="local", gemini_fallback=False, request_timeout=90)
    schemas = [ToolSchema(t.name, t.description, t.parameters) for t in registered()]
    report: dict[str, Any] = {
        "method": "One sequential first-response generation per case; no browser/tools executed",
        "intent_settings": "Production schema/prompt; temperature 0, context 2048, output 256",
        "tool_settings": "Production prompt/all tool schemas; context 4096, output 256",
        "tool_temperature": config.temperature,
        "platform": platform.platform(),
        "ollama": UrllibTransport().get_json(
            config.ollama_host.rstrip("/") + "/api/version", timeout=5
        ),
        "limitations": "Small synthetic sample; no live UI or speech evaluation; no retry",
        "models": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for model in args.models:
        selected = replace(config, model=model)
        transport = MeasuredTransport()
        provider = OllamaProvider(selected, transport)
        result: dict[str, Any] = {"model": model, "cases": []}
        report["models"].append(result)
        deadline = time.monotonic() + 3600
        while args.wait_downloads and not provider.is_available() and time.monotonic() < deadline:
            print(f"Waiting for download: {model}", flush=True)
            time.sleep(30)
        if not provider.is_available():
            result["unavailable"] = True
            save()
            continue
        details = transport.post_json(provider.host + "/api/show", {"model": model}, timeout=15)
        result["details"] = details.get("details")
        result["capabilities"] = details.get("capabilities")
        installed = transport.get_json(provider.host + "/api/tags", timeout=5)
        result["installed_model"] = next(
            (entry for entry in installed.get("models", []) if entry.get("name") == model), None
        )
        for kind, cases in (("intent", INTENTS), ("tools", TOOLS)):
            for case in cases:
                if isinstance(case, Case):
                    name, prompt = case.name, case.prompt
                    expected: object = [asdict(step) for step in case.expected]
                else:
                    name, prompt, calls = case
                    expected = [call.as_dict() for call in calls]
                print(f"{model} {kind}/{name}", flush=True)
                row: dict[str, Any] = {
                    "kind": kind,
                    "name": name,
                    "prompt": prompt,
                    "expected": expected,
                    "raw_passed": False,
                    "validated_passed": False,
                }
                transport.latest = {}
                started = time.monotonic()
                try:
                    if isinstance(case, Case):
                        raw = provider.structured(
                            [Message("system", INSTRUCTIONS), Message("user", prompt)], SCHEMA
                        )
                        row["raw"] = raw
                        row["raw_passed"] = equivalent(raw_steps(raw), case.expected)
                        checked = validate(raw, prompt)
                        row["validated"] = [asdict(step) for step in checked]
                        row["validated_passed"] = equivalent(checked, case.expected)
                        row["unexpected_send"] = any(
                            s.action == "send" for s in raw_steps(raw)
                        ) and not any(s.action == "send" for s in case.expected)
                    else:
                        reply = provider.complete(
                            [Message("system", system_prompt(selected)), Message("user", prompt)],
                            schemas,
                        )
                        row["calls"] = [call.as_dict() for call in reply.tool_calls]
                        row["answer"] = reply.content
                        row["raw_passed"] = reply.tool_calls == case[2] and (
                            bool(reply.tool_calls) or bool(reply.content.strip())
                        )
                        row["validated_passed"] = row["raw_passed"]
                except (RolexError, ValueError, KeyError, TypeError) as exc:
                    row["error"] = str(exc)
                row["seconds"] = round(time.monotonic() - started, 3)
                for field in ("load_duration", "prompt_eval_count", "eval_count", "eval_duration"):
                    if field in transport.latest:
                        row[field] = transport.latest[field]
                row["runtime"] = transport.get_json(provider.host + "/api/ps", timeout=5)
                result["cases"].append(row)
                print(
                    f"  raw={row['raw_passed']} validated={row['validated_passed']} "
                    f"{row['seconds']}s",
                    flush=True,
                )
                result["raw_passed"] = sum(r["raw_passed"] for r in result["cases"])
                result["validated_passed"] = sum(r["validated_passed"] for r in result["cases"])
                result["total"] = len(result["cases"])
                result["median_seconds"] = round(
                    statistics.median(r["seconds"] for r in result["cases"]), 3
                )
                save()
        # Avoid keeping several model allocations alive on an 8 GB machine.
        transport.post_json(
            provider.host + "/api/generate",
            {"model": model, "keep_alive": 0},
            timeout=15,
        )
    save()


if __name__ == "__main__":
    main()
