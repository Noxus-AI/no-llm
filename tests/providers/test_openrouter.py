import pytest

from no_llm.providers.provider_configs.openrouter import OpenRouterProvider


@pytest.mark.vcr()
def test_openrouter_provider_connection():
    """Test that OpenRouter provider can successfully connect to the API."""
    provider = OpenRouterProvider()
    result = provider.test()
    assert result is True, "OpenRouter provider test should return True with valid API key"