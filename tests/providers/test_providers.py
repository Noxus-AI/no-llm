from __future__ import annotations

from pydantic_ai.providers.openai import OpenAIProvider
from typing import Literal

from no_llm.providers import EnvVar
from no_llm.providers.config import ProviderConfiguration


class TestProvider(ProviderConfiguration):
    """Test provider for unit tests"""

    type: Literal["test"] = "test"  # type: ignore
    id: str = "test"
    name: str = "Test Provider"
    api_key: EnvVar[str] = EnvVar[str]("$TEST_API_KEY")
    _iterator_index: int = 0

    def to_pydantic(self) -> OpenAIProvider:
        # Simple implementation for testing
        return OpenAIProvider(api_key=str(self.api_key))

    def reset_iterator(self) -> None:
        self._iterator_index = 0



def test_provider_reset_iterator():
    provider = TestProvider()
    provider._iterator_index = 5
    provider.reset_iterator()
    assert provider._iterator_index == 0

