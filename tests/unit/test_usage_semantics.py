"""Unit tests locking chak's canonical disjoint-bucket Usage semantics.

Both Anthropic (natively disjoint) and OpenAI (natively subset) provider
adapters must normalize their raw ``usage`` payloads into the same shape
before storing them on ``Metadata.usage``:

    total_tokens == prompt_tokens
                  + completion_tokens
                  + cache_creation_input_tokens
                  + cache_read_input_tokens

If either provider stops honouring the invariant, cost math and cross-
provider dashboards silently break \u2014 hence the assertions below.
"""
from types import SimpleNamespace

import pytest

from chak.metadata import Money, Usage
from chak.providers.llm.anthropic_compat import AnthropicCompatibleMessageConverter
from chak.providers.llm.openai_compat import OpenAICompatibleMessageConverter

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers to build fake provider responses without pulling in real SDK types.
# ---------------------------------------------------------------------------

def _openai_response(prompt: int, completion: int, cached: int = 0, cache_write: int = 0):
    """Fake an OpenAI Chat Completions response object.

    OpenAI reports ``prompt_tokens`` as a SUPERSET already containing
    cached_tokens and cache_write_tokens \u2014 mirror that here so the tests
    exercise the subtraction path in the adapter.
    """
    details = SimpleNamespace(cached_tokens=cached, cache_write_tokens=cache_write)
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=details,
    )
    choice = SimpleNamespace(finish_reason="stop", message=SimpleNamespace())
    return SimpleNamespace(usage=usage, model="gpt-x", choices=[choice]), choice


def _anthropic_response(inp: int, out: int, cache_c: int = 0, cache_r: int = 0):
    """Fake an Anthropic Messages API response with four disjoint buckets."""
    usage = SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_creation_input_tokens=cache_c,
        cache_read_input_tokens=cache_r,
    )
    return SimpleNamespace(usage=usage, model="claude-x", stop_reason="end_turn")


# ---------------------------------------------------------------------------
# Invariant tests
# ---------------------------------------------------------------------------

class TestOpenAIDisjointNormalization:
    """OpenAI's superset prompt_tokens must be split into disjoint buckets."""

    def test_cache_read_subtracted_from_prompt(self):
        # 1000 raw prompt, of which 800 came from cache.
        response, choice = _openai_response(prompt=1000, completion=200, cached=800)
        md = OpenAICompatibleMessageConverter()._build_metadata(response, choice)

        u = md.usage
        assert u.prompt_tokens == 200, "prompt_tokens must exclude cached_tokens"
        assert u.cache_read_input_tokens == 800
        assert u.cache_creation_input_tokens == 0
        assert u.completion_tokens == 200
        assert u.total_tokens == 1200
        # Invariant
        assert u.total_tokens == (
            u.prompt_tokens + u.completion_tokens
            + u.cache_creation_input_tokens + u.cache_read_input_tokens
        )

    def test_cache_write_also_subtracted(self):
        response, choice = _openai_response(
            prompt=1500, completion=300, cached=200, cache_write=500,
        )
        md = OpenAICompatibleMessageConverter()._build_metadata(response, choice)

        u = md.usage
        # 1500 - 200 - 500 = 800 fresh input
        assert u.prompt_tokens == 800
        assert u.cache_read_input_tokens == 200
        assert u.cache_creation_input_tokens == 500
        assert u.completion_tokens == 300
        assert u.total_tokens == 1800

    def test_no_cache_case_unchanged(self):
        response, choice = _openai_response(prompt=500, completion=100)
        md = OpenAICompatibleMessageConverter()._build_metadata(response, choice)

        u = md.usage
        assert u.prompt_tokens == 500
        assert u.cache_read_input_tokens == 0
        assert u.cache_creation_input_tokens == 0
        assert u.total_tokens == 600

    def test_negative_prompt_clamped_to_zero(self):
        """Guard against pathological servers where cached > prompt."""
        response, choice = _openai_response(prompt=100, completion=50, cached=200)
        md = OpenAICompatibleMessageConverter()._build_metadata(response, choice)

        u = md.usage
        assert u.prompt_tokens == 0  # max(100 - 200 - 0, 0)


class TestAnthropicDisjointPassthrough:
    """Anthropic buckets are already disjoint; we just compute total."""

    def test_first_call_with_cache_write(self):
        response = _anthropic_response(inp=640, out=6635, cache_c=12208, cache_r=0)
        md = AnthropicCompatibleMessageConverter()._build_metadata(response)

        u = md.usage
        assert u.prompt_tokens == 640
        assert u.cache_creation_input_tokens == 12208
        assert u.cache_read_input_tokens == 0
        assert u.completion_tokens == 6635
        assert u.total_tokens == 19483  # 640 + 6635 + 12208 + 0

    def test_cache_hit_subsequent_call(self):
        response = _anthropic_response(inp=50, out=200, cache_c=0, cache_r=12208)
        md = AnthropicCompatibleMessageConverter()._build_metadata(response)

        u = md.usage
        assert u.prompt_tokens == 50
        assert u.cache_read_input_tokens == 12208
        assert u.total_tokens == 12458


class TestUsageInvariantHolds:
    """The four-bucket disjoint invariant must hold on every extraction path."""

    @pytest.mark.parametrize("prompt,completion,cached,cache_write", [
        (1000, 200, 0, 0),
        (1000, 200, 800, 0),
        (1000, 200, 0, 300),
        (1000, 200, 400, 300),
    ])
    def test_openai_invariant(self, prompt, completion, cached, cache_write):
        response, choice = _openai_response(
            prompt=prompt, completion=completion,
            cached=cached, cache_write=cache_write,
        )
        u = OpenAICompatibleMessageConverter()._build_metadata(response, choice).usage
        assert u.total_tokens == (
            u.prompt_tokens + u.completion_tokens
            + u.cache_creation_input_tokens + u.cache_read_input_tokens
        )

    @pytest.mark.parametrize("inp,out,cc,cr", [
        (100, 50, 0, 0),
        (100, 50, 2000, 0),
        (100, 50, 0, 2000),
        (100, 50, 500, 1500),
    ])
    def test_anthropic_invariant(self, inp, out, cc, cr):
        response = _anthropic_response(inp=inp, out=out, cache_c=cc, cache_r=cr)
        u = AnthropicCompatibleMessageConverter()._build_metadata(response).usage
        assert u.total_tokens == (
            u.prompt_tokens + u.completion_tokens
            + u.cache_creation_input_tokens + u.cache_read_input_tokens
        )


class TestEstimateCost:
    """Uniform cost formula: (pt*p_in + ct*p_out + cc*p_cw + cr*p_cr) / 1e6."""

    def test_basic_input_output_only(self):
        u = Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
        cost = u.estimate_cost(input_price=3.0, output_price=15.0)
        # Return type is Money — access .amount for the numeric value.
        assert isinstance(cost, Money)
        assert cost.currency == "USD"  # default
        assert cost.amount == pytest.approx(3.0 + 7.5)  # $3 + $7.5 = $10.5

    def test_with_cache_buckets(self):
        # Reproduce the real dump case: 640 fresh in, 12208 cache_write,
        # 0 cache_read, 6635 output at Claude Sonnet 4 unit prices.
        u = Usage(
            prompt_tokens=640,
            completion_tokens=6635,
            cache_creation_input_tokens=12208,
            cache_read_input_tokens=0,
            total_tokens=19483,
        )
        cost = u.estimate_cost(
            input_price=3.0,
            output_price=15.0,
            cache_write_price=3.75,
            cache_read_price=0.30,
        )
        expected = (
            640 * 3.0
            + 6635 * 15.0
            + 12208 * 3.75
            + 0 * 0.30
        ) / 1_000_000
        assert cost.amount == pytest.approx(expected)
        assert cost.currency == "USD"

    def test_cache_read_default_zero_prices_do_not_charge(self):
        u = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            cache_read_input_tokens=1_000_000,
        )
        # Caller passes only input/output prices — cache buckets default to 0.
        cost = u.estimate_cost(input_price=3.0, output_price=15.0)
        expected = (100 * 3.0 + 50 * 15.0) / 1_000_000
        assert cost.amount == pytest.approx(expected)

    def test_custom_currency_tag_flows_through(self):
        """CNY pricing (typical for Chinese providers) tags the result."""
        u = Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
        cost = u.estimate_cost(
            input_price=1.0, output_price=2.0, currency="CNY",
        )
        assert cost.currency == "CNY"
        assert cost.amount == pytest.approx(1.0 + 1.0)  # ¥2

    def test_int_prices_accepted_and_coerced_to_float(self):
        """Real pricing pages mix ints ($3/MTok) and floats ($0.15/MTok).

        Per PEP 484 int is a subtype of float, so ``input_price: float`` must
        accept both. Locking this behavior in so a future stricter signature
        cannot silently break callers writing prices as bare ints.
        """
        u = Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
        cost = u.estimate_cost(
            input_price=3,           # int
            output_price=15,         # int
            cache_write_price=3,     # int
            cache_read_price=0,      # int (also exercises default 0-price path)
        )
        assert isinstance(cost.amount, float)  # Money.amount is always float
        assert cost.amount == pytest.approx(3.0 + 7.5)

    def test_mixed_int_and_float_prices(self):
        u = Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
        cost = u.estimate_cost(input_price=3, output_price=0.5)  # int + float
        assert cost.amount == pytest.approx(3.0 + 0.25)


class TestMoneyArithmetic:
    """Money enforces same-currency arithmetic and supports common ops."""

    def test_add_same_currency(self):
        a = Money(amount=1.5, currency="USD")
        b = Money(amount=2.5, currency="USD")
        assert (a + b) == Money(amount=4.0, currency="USD")

    def test_add_different_currency_raises(self):
        a = Money(amount=1.0, currency="USD")
        b = Money(amount=1.0, currency="CNY")
        with pytest.raises(ValueError, match="different currencies"):
            _ = a + b

    def test_sub_same_currency(self):
        a = Money(amount=5.0, currency="USD")
        b = Money(amount=1.5, currency="USD")
        assert (a - b) == Money(amount=3.5, currency="USD")

    def test_sub_different_currency_raises(self):
        a = Money(amount=5.0, currency="USD")
        b = Money(amount=1.0, currency="EUR")
        with pytest.raises(ValueError):
            _ = a - b

    def test_mul_by_scalar_scales_amount(self):
        m = Money(amount=1.5, currency="USD")
        assert (m * 3) == Money(amount=4.5, currency="USD")
        assert (2.0 * m) == Money(amount=3.0, currency="USD")

    def test_div_by_scalar(self):
        m = Money(amount=10.0, currency="USD")
        assert (m / 4) == Money(amount=2.5, currency="USD")

    def test_neg(self):
        m = Money(amount=3.0, currency="USD")
        assert (-m) == Money(amount=-3.0, currency="USD")

    def test_sum_of_moneys_works_via_radd(self):
        """Python's built-in sum() starts with 0; __radd__ handles that."""
        totals = [
            Money(amount=0.10, currency="USD"),
            Money(amount=0.20, currency="USD"),
            Money(amount=0.05, currency="USD"),
        ]
        s = sum(totals)
        # Compare via .amount + pytest.approx because floating-point sum
        # yields 0.35000000000000003, not exactly 0.35 (Money.__eq__ is
        # exact field comparison, so we can't compare Money objects here).
        assert isinstance(s, Money)
        assert s.currency == "USD"
        assert s.amount == pytest.approx(0.35)

    def test_sum_mixed_currency_raises(self):
        totals = [
            Money(amount=0.10, currency="USD"),
            Money(amount=1.00, currency="CNY"),
        ]
        with pytest.raises(ValueError):
            sum(totals)

    def test_add_plain_number_is_not_allowed(self):
        """Adding a bare float would drop the currency tag — must fail."""
        m = Money(amount=1.0, currency="USD")
        with pytest.raises(TypeError):
            _ = m + 1.5

    def test_str_format(self):
        m = Money(amount=0.14676, currency="USD")
        assert str(m) == "0.146760 USD"

    def test_frozen_and_hashable(self):
        m = Money(amount=1.0, currency="USD")
        # Frozen — mutation must fail (pydantic v2 raises ValidationError).
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            m.amount = 2.0
        # Hashable — can go into sets.
        s = {m, Money(amount=1.0, currency="USD"), Money(amount=2.0, currency="USD")}
        assert len(s) == 2


class TestMoneyCurrencyValidation:
    """Currency codes are normalized to uppercase and shape-checked."""

    def test_lowercase_input_is_normalized_to_uppercase(self):
        m = Money(amount=1.0, currency="usd")
        assert m.currency == "USD"

    def test_mixed_case_input_is_normalized(self):
        m = Money(amount=1.0, currency="Cny")
        assert m.currency == "CNY"

    def test_whitespace_is_stripped(self):
        m = Money(amount=1.0, currency="  USD  ")
        assert m.currency == "USD"

    def test_case_insensitive_addition_succeeds_after_normalization(self):
        """The whole point of case normalization: these should add cleanly."""
        a = Money(amount=1.0, currency="usd")
        b = Money(amount=2.0, currency="USD")
        assert (a + b).amount == pytest.approx(3.0)

    @pytest.mark.parametrize(
        "bad",
        [
            "",         # empty
            "US",       # too short
            "USDD",     # too long
            "US1",      # contains digit
            "US$",      # contains symbol
            "US D",     # contains space in middle
            "人民币",     # non-ASCII
            "CN¥",      # mixed ascii + non-ascii
        ],
    )
    def test_malformed_codes_are_rejected(self, bad):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Money(amount=1.0, currency=bad)

    def test_non_string_input_is_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Money(amount=1.0, currency=123)  # type: ignore[arg-type]

    def test_estimate_cost_normalizes_currency_kwarg(self):
        """Currency passed via estimate_cost() must flow through the same validator."""
        u = Usage(prompt_tokens=1_000_000, completion_tokens=0)
        cost = u.estimate_cost(input_price=1.0, output_price=0.0, currency="cny")
        assert cost.currency == "CNY"

    def test_estimate_cost_rejects_bad_currency(self):
        from pydantic import ValidationError
        u = Usage(prompt_tokens=100, completion_tokens=0)
        with pytest.raises(ValidationError):
            u.estimate_cost(input_price=1.0, output_price=0.0, currency="US")
