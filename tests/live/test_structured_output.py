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


@pytest.mark.asyncio
async def test_core_provider_structured_output_list_return(core_provider):
    conv = chak.Conversation(core_provider.model_uri, api_key=core_provider.api_key, timeout=90)

    result = await conv.asend(
        "Return structured data for exactly two cities: Paris, France and Tokyo, Japan.",
        returns=list[CityInfo],
        timeout=90,
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(item, CityInfo) for item in result)
    assert all(item.city for item in result)
    assert all(item.country for item in result)
