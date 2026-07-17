"""Unit tests for MoonshotProvider._apply_reasoning_params.

The hook exists to work around Moonshot's HTTP 400::

    tool_choice 'specified' is incompatible with thinking enabled

Same shape as DeepSeek's thinking-mode restriction; same fix pattern.
These tests pin down the branch matrix so regressions don't silently
resurface when structured output stops working on Kimi models.
"""

from chak.providers.llm.moonshot import (
    MoonshotProvider,
    _thinking_can_be_disabled,
)


def _apply(kwargs: dict) -> dict:
    """Call the hook the same way OpenAICompatibleProvider._send_complete does.

    Bypass ``__init__`` via ``object.__new__`` so we don't need real config /
    an SDK client — the hook only reads/writes the kwargs dict and doesn't
    touch self, so a raw instance is enough to satisfy the ``super()`` chain.
    """
    provider = object.__new__(MoonshotProvider)
    provider._apply_reasoning_params(kwargs)
    return kwargs


# ── model classification --------------------------------------------------


def test_thinking_can_be_disabled_matrix():
    """Whitelist-by-blacklist: only K3 and K2.7-code are always-on."""
    # Always-on families — chak must NOT try to disable thinking.
    assert _thinking_can_be_disabled("kimi-k3") is False
    assert _thinking_can_be_disabled("kimi-k3-vision") is False  # hypothetical future variant
    assert _thinking_can_be_disabled("kimi-k2.7-code") is False
    assert _thinking_can_be_disabled("kimi-k2.7-code-highspeed") is False

    # Disable-supported families — safe to inject.
    assert _thinking_can_be_disabled("kimi-k2.6") is True
    assert _thinking_can_be_disabled("kimi-k2.5") is True

    # Non-thinking legacy models — injection is a no-op on the wire, treat
    # as "safe to inject" so the caller's request still goes through.
    assert _thinking_can_be_disabled("moonshot-v1-8k") is True
    assert _thinking_can_be_disabled("moonshot-v1-32k") is True

    # Defensive: empty / None-ish inputs must not crash and must default
    # to "safe" — otherwise a stray call with unset model would silently
    # skip the injection and 400 the user.
    assert _thinking_can_be_disabled("") is True
    assert _thinking_can_be_disabled(None) is True  # type: ignore[arg-type]


# ── the four branches of the hook ----------------------------------------


def test_no_injection_when_tool_choice_is_auto():
    """Auto/none/absent tool_choice → the 400 doesn't apply, don't touch anything."""
    for tc in (None, "auto", "none", "any"):
        kwargs = {"model": "kimi-k2.6", "tool_choice": tc}
        out = _apply(dict(kwargs))
        assert "extra_body" not in out, f"unexpected injection for tool_choice={tc!r}"


def test_injects_thinking_disabled_for_forced_tool_choice_on_k26():
    """The main path: k2.6 + specific-function tool_choice → inject."""
    kwargs = {
        "model": "kimi-k2.6",
        "tool_choice": {"type": "function", "function": {"name": "extract_city"}},
    }
    out = _apply(kwargs)
    assert out["extra_body"] == {"thinking": {"type": "disabled"}}


def test_injects_thinking_disabled_for_required_tool_choice():
    """`tool_choice="required"` is also a forced form — same treatment."""
    kwargs = {"model": "kimi-k2.5", "tool_choice": "required"}
    out = _apply(kwargs)
    assert out["extra_body"] == {"thinking": {"type": "disabled"}}


def test_does_not_inject_on_kimi_k3_even_with_forced_tool_choice():
    """K3 cannot disable thinking; injecting would mask the real error."""
    kwargs = {
        "model": "kimi-k3",
        "tool_choice": {"type": "function", "function": {"name": "extract_city"}},
    }
    out = _apply(kwargs)
    # No extra_body means the request goes through as-is and Moonshot's
    # native 400 surfaces — that's the correct behavior for K3 until chak
    # grows a response_format-based structured-output path.
    assert "extra_body" not in out


def test_does_not_inject_on_k27_code():
    """K2.7-code family also has thinking permanently on."""
    for model in ("kimi-k2.7-code", "kimi-k2.7-code-highspeed"):
        kwargs = {
            "model": model,
            "tool_choice": {"type": "function", "function": {"name": "extract"}},
        }
        out = _apply(kwargs)
        assert "extra_body" not in out, f"unexpected injection for {model}"


# ── caller-config preservation -------------------------------------------


def test_respects_caller_thinking_config():
    """If the caller already set extra_body.thinking, leave it alone.

    The caller may deliberately want thinking on (e.g. to test a k2.6 quirk),
    and chak must not second-guess an explicit choice.
    """
    kwargs = {
        "model": "kimi-k2.6",
        "tool_choice": "required",
        "extra_body": {"thinking": {"type": "enabled", "keep": "all"}},
    }
    out = _apply(kwargs)
    # Untouched — exact same dict content.
    assert out["extra_body"] == {"thinking": {"type": "enabled", "keep": "all"}}


def test_shallow_merge_preserves_other_extra_body_keys():
    """Injection must not clobber unrelated extra_body settings."""
    kwargs = {
        "model": "kimi-k2.6",
        "tool_choice": "required",
        "extra_body": {"foo": "bar", "custom_flag": True},
    }
    out = _apply(kwargs)
    assert out["extra_body"] == {
        "foo": "bar",
        "custom_flag": True,
        "thinking": {"type": "disabled"},
    }


def test_original_extra_body_is_not_mutated_in_place():
    """The hook copies before mutating so caller-owned dicts stay clean.

    If chak mutated the caller's dict, a retry (or a shared config) would
    leak the injected 'thinking' key and cause spooky action at a distance
    on subsequent requests.
    """
    original = {"foo": "bar"}
    kwargs = {
        "model": "kimi-k2.6",
        "tool_choice": "required",
        "extra_body": original,
    }
    _apply(kwargs)
    # The dict we passed in is untouched; only kwargs["extra_body"] points
    # to the new merged copy.
    assert original == {"foo": "bar"}
