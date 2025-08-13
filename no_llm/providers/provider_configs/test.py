from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from no_llm.providers.config import ProviderConfiguration
from no_llm.providers.env_var import EnvVar


class TestProvider(ProviderConfiguration):
    """Test provider configuration for testing purposes only"""

    type: Literal["test"] = "test"  # type: ignore
    id: str = "test"
    name: str = "Test Provider"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$TEST_API_KEY"),
        description="Test API key - any value works for testing",
    )
    base_url: str = Field(
        default="https://api.test.example/v1/",
        description="Test base URL for testing purposes"
    )

    def test(self) -> bool:
        """Always returns True for testing purposes"""
        return True

    def has_valid_env(self) -> bool:
        """Always returns True for testing - doesn't require real env vars"""
        return True

    def to_pydantic(self) -> PydanticOpenAIProvider:
        """Returns a mock OpenAI provider for testing"""
        return PydanticOpenAIProvider(
            api_key="test-key",
            base_url=str(self.base_url),
        )
