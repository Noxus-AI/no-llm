"""Re-exports of provider classes for backward compatibility."""

from typing import Annotated

import pydantic

from no_llm.providers.anthropic import AnthropicProvider
from no_llm.providers.azure import AzureProvider
from no_llm.providers.base import ParameterMapping, Provider
from no_llm.providers.bedrock import BedrockProvider
from no_llm.providers.deepseek import DeepseekProvider
from no_llm.providers.env_var import EnvVar
from no_llm.providers.fireworks import FireworksProvider
from no_llm.providers.grok import GrokProvider
from no_llm.providers.groq import GroqProvider
from no_llm.providers.mistral import MistralProvider
from no_llm.providers.openai import OpenAIProvider
from no_llm.providers.openrouter import OpenRouterProvider
from no_llm.providers.perplexity import PerplexityProvider
from no_llm.providers.together import TogetherProvider
from no_llm.providers.vertex import VertexProvider

Providers = Annotated[
    OpenAIProvider
    | AnthropicProvider
    | VertexProvider
    | AzureProvider
    | MistralProvider
    | GroqProvider
    | PerplexityProvider
    | DeepseekProvider
    | TogetherProvider
    | OpenRouterProvider
    | GrokProvider
    | FireworksProvider
    | BedrockProvider,
    pydantic.Discriminator("type"),
]

__all__ = [
    "EnvVar",
    "Provider",
    "Providers",
    "ParameterMapping",
    "OpenAIProvider",
    "AnthropicProvider",
    "VertexProvider",
    "MistralProvider",
    "GroqProvider",
    "PerplexityProvider",
    "DeepseekProvider",
    "AzureProvider",
    "BedrockProvider",
    "TogetherProvider",
    "OpenRouterProvider",
    "GrokProvider",
    "FireworksProvider",
]
