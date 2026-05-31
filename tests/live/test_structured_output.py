import pytest
from pydantic import BaseModel

import chak

pytestmark = [pytest.mark.live, pytest.mark.structured]


class CityInfo(BaseModel):
    city: str
    country: str


@pytest.mark.asyncio
async def test_core_provider_structured_output(core_provider):
    conv = chak.Conversation(core_provider.model_uri, api_key=core_provider.api_key, timeout=90)

    result = await conv.asend(
        "Return structured data for Paris, France.",
        returns=CityInfo,
        timeout=90,
    )

    assert isinstance(result, CityInfo)
    assert result.city
    assert result.country
