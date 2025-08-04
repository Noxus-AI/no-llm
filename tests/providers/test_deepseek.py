import pytest

from no_llm.providers.provider_configs.deepseek import DeepseekProvider


@pytest.mark.vcr()
def test_deepseek_provider_connection():
    """Test that DeepSeek provider can successfully connect to the API."""
    provider = DeepseekProvider()
    result = provider.test()
    assert result is True, "DeepSeek provider test should return True with valid API key"