import pytest

from no_llm.providers.provider_configs.openrouter import OpenRouterProvider


@pytest.mark.vcr()
def test_openrouter_provider_connection():
    """Test that OpenRouter provider can successfully connect to the API."""
    provider = OpenRouterProvider()
    result = provider.test()
    assert result is True, "OpenRouter provider test should return True with valid API key"


@pytest.mark.vcr()
def test_openrouter_provider_invalid_key(monkeypatch):
    """Test that OpenRouter provider returns False with invalid API key."""
    monkeypatch.setenv("INVALID_OPENROUTER_KEY", "invalid-api-key")
    from no_llm.providers.env_var import EnvVar
    provider = OpenRouterProvider(api_key=EnvVar[str]("$INVALID_OPENROUTER_KEY"))
    result = provider.test()
    assert result is False, "OpenRouter provider test should return False with invalid API key"