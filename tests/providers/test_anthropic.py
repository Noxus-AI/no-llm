import pytest

from no_llm.providers.provider_configs.anthropic import AnthropicProvider


@pytest.mark.vcr()
@pytest.mark.skip(reason="Anthropic provider is not available in the free tier")
def test_anthropic_provider_connection():
    """Test that Anthropic provider can successfully connect to the API."""
    provider = AnthropicProvider()
    result = provider.test()
    assert result is True, "Anthropic provider test should return True with valid API key"