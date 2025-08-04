import pytest

from no_llm.providers.provider_configs.openai import OpenAIProvider


@pytest.mark.vcr()
def test_openai_provider_connection():
    """Test that OpenAI provider can successfully connect to the API."""
    provider = OpenAIProvider()
    result = provider.test()
    assert result is True, "OpenAI provider test should return True with valid API key"

