"""
Multimodal (text + image) integration tests.

Uses a tiny self-contained base64 PNG so no network dependency is needed.
Verifies that Chat Completions → Responses API content-type conversion
(``text`` → ``input_text``, ``image_url`` → ``input_image``) works
correctly for vision-capable providers.
"""

import pytest

import chak
from chak import HumanMessage, Image

pytestmark = [pytest.mark.live, pytest.mark.multimodal]

# 10×10 pixel solid-red PNG, base64-encoded (~126 chars).
# Self-contained — no network dependency, no external file.
_TINY_RED_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFElEQVR4nGP4"
    "z8DwnxjMMKqQvgoBksPHOas6/LEAAAAASUVORK5CYII="
)


# ─── async + Image attachment ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_multimodal_attachment(multimodal_provider):
    """Most common path: ``Image`` attachment with base64 data URI."""
    conv = chak.Conversation(
        multimodal_provider.model_uri,
        api_key=multimodal_provider.api_key,
        timeout=60,
    )

    response = await conv.asend(
        "What color is this image? One word only.",
        attachments=[Image(_TINY_RED_PNG)],
    )

    assert response.content is not None
    assert len(response.content.strip()) > 0


# ─── async + manual multimodal content ──────────────────────────────────

@pytest.mark.asyncio
async def test_multimodal_manual_content(multimodal_provider):
    """Raw multimodal content list — exercises type conversion edge cases."""
    conv = chak.Conversation(
        multimodal_provider.model_uri,
        api_key=multimodal_provider.api_key,
        timeout=60,
    )

    manual_content = [
        {"type": "text", "text": "What color is this image? One word only."},
        {
            "type": "image_url",
            "image_url": {"url": _TINY_RED_PNG, "detail": "high"},
        },
    ]

    response = await conv.asend(HumanMessage(content=manual_content))

    assert response.content is not None
    assert len(response.content.strip()) > 0


# ─── streaming + Image attachment ───────────────────────────────────────

@pytest.mark.asyncio
async def test_multimodal_streaming(multimodal_provider):
    """Streaming mode — verifies Responses API event parsing with image."""
    conv = chak.Conversation(
        multimodal_provider.model_uri,
        api_key=multimodal_provider.api_key,
        timeout=60,
    )

    stream = await conv.asend(
        "What color is this image? One word only.",
        attachments=[Image(_TINY_RED_PNG)],
        stream=True,
    )

    chunks = []
    async for chunk in stream:
        if hasattr(chunk, "content") and chunk.content:
            chunks.append(chunk.content)

    content = "".join(chunks)
    assert len(content.strip()) > 0
