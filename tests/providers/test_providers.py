from __future__ import annotations

from pydantic_ai.providers.openai import OpenAIProvider

from no_llm.providers import EnvVar
from no_llm.providers.base import ParameterMapping, Provider
from no_llm.providers.vertex import VertexProvider


class TestProvider(Provider):
    """Test provider for unit tests"""

    type: str = "test"
    name: str = "Test Provider"
    api_key: EnvVar[str] = EnvVar[str]("$TEST_API_KEY")
    _iterator_index: int = 0

    def to_pydantic(self) -> OpenAIProvider:
        # Simple implementation for testing
        return OpenAIProvider(api_key=str(self.api_key))

    def reset_iterator(self) -> None:
        self._iterator_index = 0

    def map_parameters(self, params: dict) -> dict:
        mapped = {}
        for key, value in params.items():
            if key in self.parameter_mappings:
                mapping = self.parameter_mappings[key]
                if mapping.supported and mapping.name:
                    mapped[mapping.name] = value
            else:
                mapped[key] = value
        return mapped


def test_provider_reset_iterator():
    provider = TestProvider()
    provider._iterator_index = 5
    provider.reset_iterator()
    assert provider._iterator_index == 0


def test_provider_map_parameters():
    provider = TestProvider()

    # Add some parameter mappings
    provider.parameter_mappings = {
        "temperature": ParameterMapping(name="temp", supported=True),
        "max_tokens": ParameterMapping(name="max_output_tokens", supported=True),
        "unsupported_param": ParameterMapping(name="unused", supported=False),
    }

    params = {
        "temperature": 0.7,
        "max_tokens": 100,
        "unsupported_param": "test",
        "unmapped_param": "keep",
        "direct_param": "direct",
    }

    mapped = provider.map_parameters(params)

    assert mapped == {
        "temp": 0.7,
        "max_output_tokens": 100,
        "unmapped_param": "keep",
        "direct_param": "direct",
    }
