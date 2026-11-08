"""Memory survives restarts while keeping tool rounds and concurrent sessions intact."""

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from rolex.agent.memory import Context, Memory, MemoryError
from rolex.core.config import Config, load_config
from rolex.core.errors import ProviderError
from rolex.interface.chat import run_chat
from rolex.interface.listen import run_listen
from rolex.llm.base import Message, Reply, ToolCall
from rolex.voice.speak import Speaker
from test_chat import FakeProvider
from test_listen import FakeEars, FakeSpeaker


def test_signed_tool_history_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    messages = [
        Message("user", "time?"),
        Message(
            "assistant",
            "",
            (ToolCall("get_time", {}),),
            provider_data={"gemini_parts": [{"thoughtSignature": "opaque"}]},
        ),
        Message.tool_result("get_time", "5pm"),
        Message("assistant", "It is 5pm"),
    ]
    Memory(path).save(messages, "Earlier context")
    assert Memory(path).load() == (messages, "Earlier context")


def test_concurrent_sessions_keep_separate_snapshots(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    first, second = Memory(path), Memory(path)
    first.save([Message("user", "one")], "")
    second.save([Message("user", "two")], "")
    first.save([Message("user", "three")], "")
    assert Memory(path).load()[0] == [Message("user", "three")]
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2


def test_preferences_and_history_reach_chat_and_voice_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config(memory=True, memory_path=tmp_path / "memory.sqlite3")
    Memory(config.memory_path).remember("Use short answers")
    lines = iter(["My project is Rolex.", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(lines))
    first = FakeProvider(["Remembered"])
    assert run_chat(config, provider=first, tools=[]) == 0
    assert "Use short answers" in first.seen[0][0].content

    second = FakeProvider(["Your project is Rolex."])
    ears = FakeEars(["rolex what is my project"])
    assert (
        run_listen(
            config, provider=second, tools=[], ears=ears, speaker=cast(Speaker, FakeSpeaker())
        )
        == 0
    )
    assert Message("user", "My project is Rolex.") in second.seen[0]
    assert "Use short answers" in second.seen[0][0].content
    saved, _ = Memory(config.memory_path).load()
    assert any(message.content == "Your project is Rolex." for message in saved)


@pytest.mark.parametrize("failure", [False, True])
def test_compaction_retains_complete_recent_turns_and_bounds_summary(
    tmp_path: Path, failure: bool
) -> None:
    provider = FakeProvider([ProviderError("offline") if failure else "Project is Rolex."])
    store = Memory(tmp_path / "memory.sqlite3")
    context = Context(store, provider, "Current system prompt")
    history = [Message("system", "old")]
    for index in range(12):
        history.extend(
            [
                Message("user", f"question {index}"),
                Message("assistant", "", (ToolCall("get_time", {}),)),
                Message.tool_result("get_time", "now"),
                Message("assistant", f"answer {index}"),
            ]
        )
    compacted = context.commit(history)
    assert compacted[1].role == "user"
    assert len(compacted) < len(history)
    assert compacted[-4:] == history[-4:]
    messages, summary = store.load()
    assert len(summary) <= 2000
    assert summary
    assert messages == compacted[1:]
    assert provider.offered == [[]]


def test_failed_model_turn_does_not_replace_saved_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config(memory=True, memory_path=tmp_path / "memory.sqlite3")
    previous = [Message("user", "earlier"), Message("assistant", "earlier answer")]
    Memory(config.memory_path).save(previous, "")
    lines = iter(["new question", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(lines))
    run_chat(config, provider=FakeProvider([ProviderError("offline")]), tools=[])
    assert Memory(config.memory_path).load()[0] == previous


def test_invalid_memory_reports_error_without_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "bad.sqlite3"
    path.write_text("invalid sqlite data")
    with pytest.raises(MemoryError):
        Memory(path).load()
    assert path.read_text() == "invalid sqlite data"


def test_memory_setting_and_preference_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rolex.interface.cli import main

    config = load_config(
        root=tmp_path,
        env={
            "ROLEX_MEMORY": "true",
            "ROLEX_MEMORY_PATH": str(tmp_path / "memory.sqlite3"),
        },
    )
    assert config.memory
    monkeypatch.setattr("rolex.interface.cli.load_config", lambda: config)
    assert main(["remember", "Use Marathi"]) == 0
    assert main(["remember", "Use Marathi"]) == 0
    assert Memory(config.memory_path).preferences() == ["Use Marathi"]
    with pytest.raises(MemoryError):
        Memory(config.memory_path).remember(" ")


def test_older_messages_without_provider_metadata_still_load(tmp_path: Path) -> None:
    store = Memory(tmp_path / "memory.sqlite3")
    store.save([], "")
    with sqlite3.connect(store.path) as db:
        db.execute(
            "UPDATE sessions SET payload=?",
            (json.dumps({"messages": [{"role": "user", "content": "legacy"}], "summary": ""}),),
        )
    assert store.load() == ([Message("user", "legacy")], "")


def test_restored_history_does_not_authorize_a_new_mutating_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rolex.tools.base import Tool

    config = Config(memory=True, memory_path=tmp_path / "memory.sqlite3")
    Memory(config.memory_path).save(
        [
            Message("user", "yes"),
            Message("assistant", "The previous action completed."),
        ],
        "",
    )
    lines = iter(["Please do that again.", "n", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(lines))
    calls: list[object] = []

    def change(args: object) -> str:
        calls.append(args)
        return "changed"

    tool = Tool(
        name="change_setting",
        description="Change a setting",
        safety="mutating",
        run=change,
    )
    provider = FakeProvider(
        [
            Reply("", (ToolCall("change_setting", {}),)),
            Reply("The action was refused."),
        ]
    )
    assert run_chat(config, provider=provider, tools=[tool]) == 0
    assert calls == []


def test_large_older_turns_are_compacted_at_user_boundaries(tmp_path: Path) -> None:
    context = Context(Memory(tmp_path / "memory.sqlite3"), FakeProvider(["Summary"]), "system")
    history = [Message("system", "system")]
    for i in range(6):
        history += [Message("user", str(i)), Message("assistant", "x" * 4000)]
    compacted = context.commit(history)
    assert compacted[1:] == history[-4:]
    assert context.summary == "Summary"


def test_chat_continues_with_unreadable_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bad.sqlite3"
    path.write_text("invalid")
    config = Config(memory=True, memory_path=path)
    lines = iter(["hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(lines))
    provider = FakeProvider(["Hello"])
    assert run_chat(config, provider=provider, tools=[]) == 0
    assert path.read_text() == "invalid"


def test_voice_continues_with_unreadable_memory(tmp_path: Path) -> None:
    path = tmp_path / "bad.sqlite3"
    path.write_text("invalid")
    config = Config(memory=True, memory_path=path)
    speaker = FakeSpeaker()
    assert (
        run_listen(
            config,
            provider=FakeProvider(["Hello"]),
            tools=[],
            ears=FakeEars(["rolex hello"]),
            speaker=cast(Speaker, speaker),
        )
        == 0
    )
    assert "Hello" in speaker.said
    assert path.read_text() == "invalid"
