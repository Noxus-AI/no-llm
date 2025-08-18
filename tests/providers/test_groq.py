import pytest
from no_llm.providers.provider_configs.groq import GroqProvider


@pytest.mark.vcr
@pytest.mark.skip(reason="Groq provider is not working")
async def test_groq_provider_connection():
    """Test that Groq provider can successfully connect to the API."""
    provider = GroqProvider()
    result = await provider.test()
    assert result is True, "Groq provider test should return True with valid API key"


@pytest.mark.vcr
@pytest.mark.skip(reason="Groq provider is not working")
def test_groq_provider_invalid_key(monkeypatch):
    """Test that Groq provider returns False with invalid API key."""
    monkeypatch.setenv("INVALID_GROQ_KEY", "invalid-api-key")
    from no_llm.providers.env_var import EnvVar
    provider = GroqProvider(api_key=EnvVar[str]("$INVALID_GROQ_KEY"))
    result = provider.test()
    assert result is False, "Groq provider test should return False with invalid API key"
