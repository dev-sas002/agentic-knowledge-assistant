"""
Smoke tests for the Streamlit console, run headless through ``AppTest``.

They drive the real script: a question renders an answer and its sources, and
a command lands at the approval gate, where approving dispatches exactly once
through the test's recording dispatcher.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents import retrieval
from tests.fakes import FakeRetriever

APP = str(Path(__file__).resolve().parent.parent / "apps" / "ui" / "streamlit_app.py")


@pytest.fixture
def app():
    from streamlit.testing.v1 import AppTest

    retrieval.set_retriever(FakeRetriever())
    return AppTest.from_file(APP, default_timeout=30)


def test_the_console_renders_an_empty_state(app):
    app.run()
    assert not app.exception
    assert any("Try" in block.value for block in app.info)


def test_a_question_renders_an_answer_with_sources(app):
    app.run()
    app.chat_input[0].set_value("When are payouts issued?").run()
    assert not app.exception
    rendered = " ".join(block.value for block in app.markdown)
    assert "finance.pdf p7" in rendered
    assert "Retrieved sources" in rendered


def test_a_command_waits_at_the_gate_until_approved(app, isolated_gate):
    app.run()
    app.chat_input[0].set_value("pause 'Chicken Biryani'").run()
    assert not app.exception
    assert isolated_gate.dispatcher.dispatched == []

    approve = next(button for button in app.button if button.label.startswith("Approve"))
    approve.click().run()
    assert not app.exception
    assert len(isolated_gate.dispatcher.dispatched) == 1
    assert isolated_gate.dispatcher.dispatched[0]["endpoint"] == "/pause_item"


def test_a_retrieval_failure_is_shown_rather_than_hidden(app):
    retrieval.set_retriever(FakeRetriever(error="NotFoundError"))
    app.run()
    app.chat_input[0].set_value("When are payouts issued?").run()
    assert not app.exception
    assert any("could not be searched" in block.value for block in app.warning)
