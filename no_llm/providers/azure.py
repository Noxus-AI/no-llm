from collections.abc import Iterator
from typing import Literal

from pydantic import Field, PrivateAttr
from pydantic_ai.providers.azure import AzureProvider as PydanticAzureProvider

from no_llm.providers.base import Provider
from no_llm.providers.env_var import EnvVar


class AzureProvider(Provider):
    """Azure provider configuration"""

    type: Literal["azure"] = "azure"
    name: str = "Azure"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$AZURE_API_KEY"),
        description="Name of environment variable containing API key",
    )
    base_url: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$AZURE_BASE_URL"),
        description="Optional base URL override",
    )
    locations: list[str] = Field(default=["eastus", "eastus2"], description="Azure regions")
    _value: str | None = PrivateAttr(default=None)

    def iter(self) -> Iterator[Provider]:
        """Yield provider variants for each location"""
        if not self.has_valid_env():
            return
        for location in self.locations:
            provider = self.model_copy()
            provider._value = location  # noqa: SLF001
            yield provider

    @property
    def current(self) -> str:
        """Get current value, defaulting to first location if not set"""
        return self._value or self.locations[0]

    def reset_variants(self) -> None:
        self._value = None

    def to_pydantic(self) -> PydanticAzureProvider:
        return PydanticAzureProvider(
            api_key=str(self.api_key),
            azure_endpoint=str(self.base_url),
        )
