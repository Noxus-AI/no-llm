import pytest
from no_llm.providers.provider_configs.mistral import MistralProvider


@pytest.mark.vcr
def test_mistral_provider_connection():
    """Test that Mistral provider can successfully connect to the API."""
    provider = MistralProvider()
    result = provider.test()
    assert result is True, "Mistral provider test should return True with valid API key"


@pytest.mark.vcr
def test_mistral_provider_invalid_key(monkeypatch):
    """Test that Mistral provider returns False with invalid API key."""
    monkeypatch.setenv("INVALID_MISTRAL_KEY", "invalid-api-key")
    from no_llm.providers.env_var import EnvVar
    provider = MistralProvider(api_key=EnvVar[str]("$INVALID_MISTRAL_KEY"))
    result = provider.test()
    assert result is False, "Mistral provider test should return False with invalid API key"
