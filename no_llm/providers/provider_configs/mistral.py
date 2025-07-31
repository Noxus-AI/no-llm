from typing import Literal

from pydantic import Field
from pydantic_ai.providers.mistral import MistralProvider as PydanticMistralProvider

from no_llm.providers.config import ProviderConfiguration
from no_llm.providers.env_var import EnvVar


class MistralProvider(ProviderConfiguration):
    """Mistral provider configuration"""

    type: Literal["mistral"] = "mistral"  # type: ignore
    id: str = "mistral"
    name: str = "Mistral AI"
    api_key: EnvVar[str] = Field(
        default_factory=lambda: EnvVar[str]("$MISTRAL_API_KEY"),
        description="Name of environment variable containing API key",
    )

    def to_pydantic(self) -> PydanticMistralProvider:
        return PydanticMistralProvider(
            api_key=str(self.api_key),
        )
