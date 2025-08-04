import pytest

from no_llm.providers.provider_configs.perplexity import PerplexityProvider


@pytest.mark.vcr()
def test_perplexity_provider_connection():
    """Test that Perplexity provider can successfully connect to the API."""
    provider = PerplexityProvider()
    result = provider.test()
    assert result is True, "Perplexity provider test should return True with valid API key"