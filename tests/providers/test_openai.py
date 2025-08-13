import pytest
from no_llm.providers.provider_configs.openai import OpenAIProvider


@pytest.mark.vcr
def test_openai_provider_connection():
    """Test that OpenAI provider can successfully connect to the API."""
    provider = OpenAIProvider()
    result = provider.test()
    assert result is True, "OpenAI provider test should return True with valid API key"


@pytest.mark.vcr
def test_openai_provider_invalid_key(monkeypatch):
    """Test that OpenAI provider returns False with invalid API key."""
    monkeypatch.setenv("INVALID_OPENAI_KEY", "invalid-api-key")
    from no_llm.providers.env_var import EnvVar
    provider = OpenAIProvider(api_key=EnvVar[str]("$INVALID_OPENAI_KEY"))
    result = provider.test()
    assert result is False, "OpenAI provider test should return False with invalid API key"

