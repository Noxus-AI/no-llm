from typing import Annotated

import pydantic

from no_llm.providers.provider_configs.anthropic import AnthropicProvider
from no_llm.providers.provider_configs.azure import AzureProvider
from no_llm.providers.provider_configs.bedrock import BedrockProvider
from no_llm.providers.provider_configs.deepseek import DeepseekProvider
from no_llm.providers.provider_configs.fireworks import FireworksProvider
from no_llm.providers.provider_configs.gemini import GeminiProvider
from no_llm.providers.provider_configs.grok import GrokProvider
from no_llm.providers.provider_configs.groq import GroqProvider
from no_llm.providers.provider_configs.mistral import MistralProvider
from no_llm.providers.provider_configs.openai import OpenAIProvider
from no_llm.providers.provider_configs.openrouter import OpenRouterProvider
from no_llm.providers.provider_configs.perplexity import PerplexityProvider
from no_llm.providers.provider_configs.together import TogetherProvider
from no_llm.providers.provider_configs.vertex import VertexProvider

AnyProvider = Annotated[
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
    | GeminiProvider
    | BedrockProvider,
    pydantic.Discriminator("type"),
]
