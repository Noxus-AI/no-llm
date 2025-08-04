import pytest

from no_llm.providers.provider_configs.mistral import MistralProvider


@pytest.mark.vcr()
def test_mistral_provider_connection():
    """Test that Mistral provider can successfully connect to the API."""
    provider = MistralProvider()
    result = provider.test()
    assert result is True, "Mistral provider test should return True with valid API key"