"""
Tests for environment handling.

A malformed environment variable used to crash at import time, taking the
whole Streamlit process down with a ValueError that named no variable. The
rule now is: a bad value degrades one setting, never the app.
"""

from __future__ import annotations

import pytest

from agents.config import env_float, env_int, env_str


class TestCoercion:
    def test_a_valid_integer_is_read(self, monkeypatch):
        monkeypatch.setenv("TEST_MAX_TOP_K", "9")
        assert env_int("TEST_MAX_TOP_K", 4) == 9

    def test_a_malformed_integer_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("TEST_MAX_TOP_K", "lots")
        assert env_int("TEST_MAX_TOP_K", 4) == 4

    def test_a_malformed_value_is_logged_not_raised(self, monkeypatch, caplog):
        monkeypatch.setenv("TEST_MAX_TOP_K", "lots")
        with caplog.at_level("WARNING"):
            env_int("TEST_MAX_TOP_K", 4)
        assert "TEST_MAX_TOP_K" in caplog.text, "the warning must name the variable at fault"

    def test_a_valid_float_is_read(self, monkeypatch):
        monkeypatch.setenv("TEST_TIMEOUT", "2.5")
        assert env_float("TEST_TIMEOUT", 30.0) == 2.5

    def test_a_malformed_float_falls_back(self, monkeypatch):
        monkeypatch.setenv("TEST_TIMEOUT", "soon")
        assert env_float("TEST_TIMEOUT", 30.0) == 30.0

    @pytest.mark.parametrize("helper,default", [(env_str, "d"), (env_int, 1), (env_float, 1.0)])
    def test_an_unset_variable_uses_the_default(self, monkeypatch, helper, default):
        monkeypatch.delenv("TEST_UNSET", raising=False)
        assert helper("TEST_UNSET", default) == default

    @pytest.mark.parametrize("helper,default", [(env_str, "d"), (env_int, 1), (env_float, 1.0)])
    def test_an_empty_variable_uses_the_default(self, monkeypatch, helper, default):
        # `GROQ_API_KEY=` in a .env file means "no key", not "the empty key".
        monkeypatch.setenv("TEST_EMPTY", "")
        assert helper("TEST_EMPTY", default) == default


class TestDocumentedDefaults:
    def test_every_setting_in_env_example_is_read_by_config(self):
        """.env.example is documentation, and documentation drifts.

        Every name it documents must actually be read somewhere in the
        package, or the configuration table in the README is describing a
        variable that does nothing.
        """
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        documented = {
            line.split("=", 1)[0].strip()
            for line in (root / ".env.example").read_text().splitlines()
            if line.strip() and not line.startswith("#") and "=" in line
        }
        source = "\n".join(
            path.read_text()
            for path in root.rglob("*.py")
            if ".venv" not in str(path) and "project_venv" not in str(path)
        ) + (root / "docker-compose.yml").read_text()
        missing = {name for name in documented if not re.search(rf"\b{re.escape(name)}\b", source)}
        assert not missing, f".env.example documents variables nothing reads: {sorted(missing)}"
