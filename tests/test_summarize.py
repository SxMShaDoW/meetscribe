"""Tests for meet.summarize — system prompt construction."""

from __future__ import annotations

import pytest

from meet.summarize import _build_system_prompt
from meet.languages import SECTION_HEADERS as _SECTION_HEADERS


class TestBuildSystemPrompt:
    def test_english_default(self):
        prompt = _build_system_prompt("en")
        assert "Meeting Overview" in prompt
        assert "Key Topics Discussed" in prompt
        assert "Action Items" in prompt
        assert "Decisions Made" in prompt
        assert "Open Questions" in prompt

    def test_farsi_headers(self):
        prompt = _build_system_prompt("fa")
        h = _SECTION_HEADERS["fa"]
        assert h["overview"] in prompt  # "خلاصه جلسه"
        assert h["topics"] in prompt
        assert h["actions"] in prompt
        # Should contain the "write ENTIRE summary in Persian" instruction
        assert "Persian" in prompt or "Farsi" in prompt

    def test_all_supported_languages(self):
        """Every language in _SECTION_HEADERS should produce a valid prompt."""
        for lang in _SECTION_HEADERS:
            prompt = _build_system_prompt(lang)
            h = _SECTION_HEADERS[lang]
            assert h["overview"] in prompt
            assert h["topics"] in prompt
            assert h["actions"] in prompt
            assert h["decisions"] in prompt
            assert h["questions"] in prompt

    def test_unknown_language_falls_back_to_english(self):
        prompt = _build_system_prompt("xx")
        assert "Meeting Overview" in prompt
