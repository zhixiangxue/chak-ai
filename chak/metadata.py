"""Message metadata types

This module defines metadata structures for messages.
Separated from providers to avoid circular imports.
"""

from typing import Any, Dict, List, Optional

import re

from pydantic import BaseModel, Field, field_validator

# ISO 4217 currency codes are three uppercase A–Z letters. We validate the
# shape only, not membership in the actual ISO list — chak is a
# cost-annotation library, not a financial system, and a hard whitelist
# would need to track additions/retirements plus block reasonable
# extensions like ``XBT`` (bitcoin) or ``XAU`` (gold).
_CURRENCY_CODE_RE = re.compile(r"^[A-Z]{3}$")


class Money(BaseModel):
    """A tagged monetary amount — an ``amount`` bound to a ``currency`` code.

    Deliberately NOT a full-blown finance abstraction: no FX conversion, no
    locale-aware formatting, no Decimal arithmetic, no sub-unit handling. Its
    only job is to keep an amount together with its currency tag so that
    LLM-cost figures cannot be silently added across currencies. If a caller
    later wants FX or precise accounting, they can convert to whatever
    library they use — chak stays out of that space.

    Arithmetic:
      * ``a + b`` and ``a - b`` require matching currency; mismatched
        currencies raise ``ValueError`` with a clear message.
      * ``money * scalar`` and ``money / scalar`` scale the amount, keeping
        the currency intact (useful for "average cost per call", etc.).
      * ``sum([m1, m2, …])`` works because ``__radd__(0)`` treats ``0`` as
        the additive identity.

    Formatting:
      * ``str(m)`` → ``"0.146760 USD"`` (six decimals, currency suffix). No
        locale, no currency symbol — the ISO code stays explicit so logs and
        dashboards are unambiguous.

    Currency codes should be 3-letter ISO 4217 strings (``USD``, ``CNY``,
    ``EUR``, …). We do NOT enum-restrict the set because new codes appear
    and the string tag is enough for our purposes.
    """

    amount: float
    currency: str = "USD"

    # Frozen so Money instances are immutable and hashable — lets callers
    # stash them in sets, dict keys, or Grand-Total bookkeeping structures.
    model_config = {"frozen": True}

    @field_validator("currency", mode="before")
    @classmethod
    def _normalize_currency(cls, v: Any) -> str:
        """Normalize + shape-check the currency code.

        Runs BEFORE pydantic's default string coercion so we control the
        exact input surface. Applies two fixes that prevent the two most
        common real-world bugs:

        1. Case mismatch — without this, ``Money('usd') + Money('USD')`` would
           blow up with a currency-mismatch error even though the caller
           obviously meant the same currency.
        2. Stray whitespace — config files and env-vars leak spaces.

        Everything else (non-string, wrong length, non-letters) becomes a
        loud ``ValueError`` — better a startup crash than a silent bogus tag.
        """
        if not isinstance(v, str):
            raise ValueError(
                f"currency must be a string ISO 4217 code, got {type(v).__name__}: {v!r}"
            )
        normalized = v.strip().upper()
        if not _CURRENCY_CODE_RE.match(normalized):
            raise ValueError(
                f"currency must be a 3-letter ISO 4217 code "
                f"(e.g. 'USD', 'CNY', 'EUR'); got {v!r}"
            )
        return normalized

    # -- internal helpers --------------------------------------------------
    def _check_same_currency(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise ValueError(
                f"Cannot combine Money values with different currencies: "
                f"{self.currency!r} vs {other.currency!r}. Convert one side "
                f"to a common currency before combining."
            )

    # -- arithmetic --------------------------------------------------------
    def __add__(self, other: Any) -> "Money":
        if isinstance(other, Money):
            self._check_same_currency(other)
            return Money(amount=self.amount + other.amount, currency=self.currency)
        return NotImplemented

    def __radd__(self, other: Any) -> "Money":
        # Support built-in ``sum([m1, m2, …])`` which starts with the int 0.
        # Only 0 is treated as the additive identity; any other number is a
        # bug (adding a plain number would drop the currency tag silently).
        if other == 0:
            return self
        return NotImplemented

    def __sub__(self, other: Any) -> "Money":
        if isinstance(other, Money):
            self._check_same_currency(other)
            return Money(amount=self.amount - other.amount, currency=self.currency)
        return NotImplemented

    def __mul__(self, scalar: Any) -> "Money":
        if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
            return Money(amount=self.amount * scalar, currency=self.currency)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, scalar: Any) -> "Money":
        if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
            return Money(amount=self.amount / scalar, currency=self.currency)
        return NotImplemented

    def __neg__(self) -> "Money":
        return Money(amount=-self.amount, currency=self.currency)

    # -- formatting --------------------------------------------------------
    def __str__(self) -> str:
        return f"{self.amount:.6f} {self.currency}"


class Usage(BaseModel):
    """Normalized token usage information for provider responses.

    Field names follow OpenAI's official naming convention, but the field
    *semantics* have been normalized to a single canonical model across all
    providers: **the four token buckets are disjoint**.

    -------------------------------------------------------------------
    Canonical semantics (uniform across providers)
    -------------------------------------------------------------------

        prompt_tokens                  : fresh input tokens (NOT cached).
                                         The tokens the model had to process
                                         from scratch this call.
        cache_creation_input_tokens    : input tokens written to cache this
                                         call (Anthropic "cache_write").
        cache_read_input_tokens        : input tokens served from cache this
                                         call (cache hit).
        completion_tokens              : output tokens generated.
        total_tokens                   : true total tokens processed, always
                                         equal to the sum of the four
                                         disjoint buckets above.

    Invariant:
        total_tokens == prompt_tokens
                      + completion_tokens
                      + cache_creation_input_tokens
                      + cache_read_input_tokens

    Provider mapping notes:
      * Anthropic Messages API returns these buckets already disjoint
        (`input_tokens`, `cache_creation_input_tokens`,
        `cache_read_input_tokens`). We map 1:1 and compute total ourselves
        because Anthropic does not send total_tokens.
      * OpenAI (and every OpenAI-compat provider: deepseek, moonshot,
        siliconflow, bailian, baidu, tencent, iflytek, mistral, minimax,
        vllm, ollama, azure, …) natively returns `prompt_tokens` as a
        SUPERSET that includes `cached_tokens`/`cache_write_tokens`. In
        `openai_compat._build_metadata` we subtract those subsets so the
        stored `prompt_tokens` matches the canonical (disjoint) meaning.
        This is why chak's `usage.prompt_tokens` may differ from the raw
        OpenAI response's `prompt_tokens` when caching is active.

    Cost estimation:
        Because the buckets are disjoint, cost is a single uniform formula
        regardless of provider — see :meth:`estimate_cost`.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_creation_input_tokens: int = 0  # Tokens written to cache
    cache_read_input_tokens: int = 0      # Tokens served from cache

    def estimate_cost(
        self,
        *,
        input_price: float,
        output_price: float,
        cache_write_price: float = 0.0,
        cache_read_price: float = 0.0,
        currency: str = "USD",
    ) -> Money:
        """Estimate call cost given per-million-token unit prices.

        All four prices are expressed in the caller-supplied ``currency`` per
        million tokens (matching the convention on OpenAI's and Anthropic's
        public pricing pages, plus most Chinese providers' CNY pricing).
        Cache prices default to 0 so callers who don't care about caching can
        pass only ``input_price`` and ``output_price``.

        Prices may be passed as ``int`` OR ``float`` — real-world pricing
        pages use both (``$3/MTok`` vs ``$0.15/MTok``). Per PEP 484 int is a
        subtype of float, so the ``float`` annotation accepts both and
        pydantic coerces the resulting ``Money.amount`` to a float regardless.

        Because chak normalizes all provider usage into four disjoint
        buckets (see class docstring), the cost formula is uniform:

            amount = ( prompt_tokens               * input_price
                    + completion_tokens            * output_price
                    + cache_creation_input_tokens  * cache_write_price
                    + cache_read_input_tokens      * cache_read_price ) / 1e6

        Returns:
            A :class:`Money` instance so downstream aggregation (``sum``,
            comparison, printing) cannot accidentally mix currencies.
            Callers who just want a number can access ``.amount``.
        """
        amount = (
            self.prompt_tokens * input_price
            + self.completion_tokens * output_price
            + self.cache_creation_input_tokens * cache_write_price
            + self.cache_read_input_tokens * cache_read_price
        ) / 1_000_000
        return Money(amount=amount, currency=currency)


class FailureRecord(BaseModel):
    """A single failed provider attempt within a request."""

    attempt_index: int = 0
    provider: str = ""
    model: str = ""
    base_url: str = ""
    error: str = ""
    status_code: Optional[int] = None
    error_type: Optional[str] = None


class ProviderTrace(BaseModel):
    """Provider routing trace for a single model request.

    Records which providers were attempted, which failed, and which
    ultimately resolved the request. Always populated — even for direct
    (non-resilient) calls — so developers never need to check for None.
    """

    primary_provider: str = ""
    primary_model: str = ""
    fallback_used: bool = False
    failover_attempts: int = 0
    failed_providers: List[FailureRecord] = Field(default_factory=list)
    resolved_provider: str = ""
    resolved_model: str = ""


class Metadata(BaseModel):
    """Standardized metadata for provider responses.

    This structure is used by BaseMessage.metadata and provides a stable
    contract across all providers.
    """

    provider: str = ""
    model: Optional[str] = None
    usage: Optional[Usage] = None
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None
    provider_trace: Optional[ProviderTrace] = None
