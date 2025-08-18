import pytest
from no_llm.providers.provider_configs.deepseek import DeepseekProvider


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_deepseek_provider_connection():
    """Test that DeepSeek provider can successfully connect to the API."""
    provider = DeepseekProvider()
    result = await provider.test()
    assert result is True, "DeepSeek provider test should return True with valid API key"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_deepseek_provider_invalid_key(monkeypatch):
    """Test that DeepSeek provider returns False with invalid API key."""
    monkeypatch.setenv("INVALID_DEEPSEEK_KEY", "invalid-api-key")
    from no_llm.providers.env_var import EnvVar
    provider = DeepseekProvider(api_key=EnvVar[str]("$INVALID_DEEPSEEK_KEY"))
    result = await provider.test()
    assert result is False, "DeepSeek provider test should return False with invalid API key"
