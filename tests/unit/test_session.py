"""Unit tests for PipelineSessionState."""

import pytest

from murmur.session import PipelineSessionState, PipelineState


def test_add_and_retrieve_messages() -> None:
    s = PipelineSessionState(system_prompt="You are helpful.")
    s.add_user("Hello")
    s.add_assistant("Hi there!")

    msgs = s.to_messages()
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"


def test_history_trimming() -> None:
    s = PipelineSessionState(max_history_turns=2)
    for i in range(5):
        s.add_user(f"msg {i}")
        s.add_assistant(f"resp {i}")

    # Should keep only last 2 turns = 4 messages
    assert len(s.history) == 4


def test_clear() -> None:
    s = PipelineSessionState()
    s.add_user("test")
    s.clear()
    assert s.history == []


def test_state_transition() -> None:
    s = PipelineSessionState()
    assert s.state == PipelineState.IDLE
    s.transition(PipelineState.LISTENING)
    assert s.state == PipelineState.LISTENING


def test_no_system_prompt_in_messages_when_empty() -> None:
    s = PipelineSessionState(system_prompt="")
    s.add_user("hi")
    msgs = s.to_messages()
    assert all(m["role"] != "system" for m in msgs)
