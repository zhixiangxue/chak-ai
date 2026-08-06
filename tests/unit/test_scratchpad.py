"""Unit tests for the std Scratchpad tool."""

import json

import pytest

from chak.tools.native.object import NativeObjectTool
from chak.tools.std import Scratchpad


pytestmark = pytest.mark.unit


def test_scratchpad_saves_reads_lists_and_persists_json(tmp_path):
    path = tmp_path / "notes.json"
    scratchpad = Scratchpad(path)

    result = scratchpad.save_section("summary", "- fact one\n- fact two")

    assert "Saved new section 'summary'" in result
    assert '1. "summary" (2 lines)' in scratchpad.list_sections()
    assert scratchpad.read_section("summary") == "- fact one\n- fact two"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "summary": "- fact one\n- fact two"
    }

    reopened = Scratchpad(path)
    assert reopened.read_section("summary") == "- fact one\n- fact two"


def test_scratchpad_overwrites_removes_and_clears_sections(tmp_path):
    scratchpad = Scratchpad(tmp_path / "notes.json")

    scratchpad.save_section("summary", "old")
    update = scratchpad.save_section("summary", "new")
    assert "Updated section 'summary'" in update
    assert scratchpad.read_section("summary") == "new"

    removed = scratchpad.remove_section("summary")
    assert "Removed section 'summary'" in removed
    assert "scratchpad is empty" in scratchpad.read_section("summary")

    scratchpad.save_section("a", "one")
    scratchpad.save_section("b", "two")
    cleared = scratchpad.clear()
    assert cleared == "Cleared all 2 section(s). Scratchpad is now empty."
    assert scratchpad.list_sections() == "(empty — no notes saved yet)"


def test_scratchpad_searches_headings_and_content(tmp_path):
    scratchpad = Scratchpad(tmp_path / "notes.json")
    scratchpad.save_section("alpha_notes", "The first topic is stored here.")
    scratchpad.save_section("beta", "Contains a unique keyword in the body.")

    heading_match = scratchpad.search_sections("alpha")
    content_match = scratchpad.search_sections("unique keyword")

    assert '"alpha_notes"' in heading_match
    assert '"beta"' in content_match
    assert "No sections matched" in scratchpad.search_sections("missing")


def test_scratchpad_read_only_mode_exposes_read_methods_only(tmp_path):
    path = tmp_path / "notes.json"
    writer = Scratchpad(path)
    writer.save_section("summary", "content")

    reader = Scratchpad(path, mode="r")

    assert reader.read_section("summary") == "content"
    assert reader.__available__() == frozenset({
        "list_sections",
        "read_section",
        "search_sections",
    })
    assert reader.save_section("new", "content") == "Error: scratchpad is read-only."
    assert reader.remove_section("summary") == "Error: scratchpad is read-only."
    assert reader.clear() == "Error: scratchpad is read-only."


def test_scratchpad_native_object_tool_method_names(tmp_path):
    tool = NativeObjectTool(Scratchpad(tmp_path / "notes.json"))

    assert set(tool.method_names) == {
        "scratchpad-list_sections",
        "scratchpad-read_section",
        "scratchpad-search_sections",
        "scratchpad-save_section",
        "scratchpad-remove_section",
        "scratchpad-clear",
    }


def test_scratchpad_handles_invalid_json_as_empty(tmp_path):
    path = tmp_path / "notes.json"
    path.write_text("not json", encoding="utf-8")

    scratchpad = Scratchpad(path)

    assert scratchpad.list_sections() == "(empty — no notes saved yet)"
    scratchpad.save_section("summary", "content")
    assert json.loads(path.read_text(encoding="utf-8")) == {"summary": "content"}


def test_scratchpad_rejects_directory_path(tmp_path):
    with pytest.raises(ValueError, match="must be a file"):
        Scratchpad(tmp_path)


def test_scratchpad_rejects_invalid_mode(tmp_path):
    with pytest.raises(ValueError, match="must be 'r' or 'rw'"):
        Scratchpad(tmp_path / "notes.json", mode="w")
