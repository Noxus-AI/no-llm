import pytest

from no_llm.providers.provider_configs.groq import GroqProvider


@pytest.mark.vcr()
def test_groq_provider_connection():
    """Test that Groq provider can successfully connect to the API."""
    provider = GroqProvider()
    result = provider.test()
    assert result is True, "Groq provider test should return True with valid API key"