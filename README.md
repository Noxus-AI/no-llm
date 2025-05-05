<div align="center">
  <h1>no/llm</h1>
  <em>Standard Interface for Large Language Models</em>
</div>

<div align="center">
  <a href="https://pypi.python.org/pypi/no_llm"><img src="https://img.shields.io/pypi/v/no_llm.svg" alt="PyPI"></a>
  <a href="https://github.com/Noxus-AI/no-llm"><img src="https://img.shields.io/pypi/pyversions/no_llm.svg" alt="versions"></a>
  <a href="https://github.com/Noxus-AI/no-llm/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Noxus-AI/no-llm.svg" alt="license"></a>
</div>

---

`no/llm` is a Python library that provides a unified interface for working with LLMs, with built-in support for model configuration, parameter validation, and provider management.

> [!IMPORTANT]
> **⚠️ Early Stage Development**  
> This project is in early stages and under active development. While we're working hard to maintain stability, APIs and features may change as we improve the library. We encourage you to try it out and provide feedback, but please be aware that production use should be carefully considered.

## Quick Install

```bash
uv pip install "no_llm[pydantic-ai]"
```

## Quick Example with Pydantic AI


> [!TIP]
> Get a free API key from [OpenRouter](https://openrouter.ai/keys) to test various models without individual provider accounts.


```python
import os

from no_llm.integrations.pydantic_ai import NoLLMModel
from no_llm.registry import ModelRegistry
from no_llm.settings import ValidationMode, settings
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

settings.validation_mode = ValidationMode.ERROR
# or ValidationMode.WARN, ValidationMode.CLAMP

os.environ["OPENROUTER_API_KEY"] = "..."

# Get model from registry
registry = ModelRegistry()
openrouter_models = list(registry.list_models(provider="openrouter"))
print([m.identity.id for m in openrouter_models])
# > ['claude-3.5-haiku', 'claude-3.5-sonnet-v2', 'claude-3.7-sonnet', 'deepseek-chat', 'deepseek-r1-llama-70b-distilled', 'deepseek-reasoner', ...]
no_llm_model = NoLLMModel(*openrouter_models)

# Use with Pydantic AI
agent = Agent(no_llm_model, model_settings=ModelSettings(temperature=1.2))
result = agent.run_sync("What is the capital of France?")
print(result.data)
# > 2025-04-09 09:50:51.375 | WARNING  | no_llm.integrations.pydantic_ai:request:220 - Model deepseek-chat failed, trying next fallback. Error: Invalid value for parameter 'temperature'
# > Current value: 1.2
# > Valid range: (0.0, 1.0)
# > Error: Value 1.2 outside range [0.0, 1.0]
# > 2025-04-09 09:50:51.375 | WARNING  | no_llm.integrations.pydantic_ai:request:220 - Model deepseek-r1-llama-70b-distilled failed, trying next fallback. Error: Invalid value for parameter 'temperature'
# > Current value: 1.2
# > Valid range: (0.0, 1.0)
# > Error: Value 1.2 outside range [0.0, 1.0]
# ✅ gemini-2.0-flash
# > The capital of France is **Paris**.
```

## Why no/llm?

* __Provider Agnostic__: Support for OpenAI, Anthropic, Google, Mistral, Groq, and more through a single interface

* __Built-in Validation__: Type-safe parameter validation and capability checking

* __Provider Fallbacks__: Automatic fallback between providers and data centers

* __Configuration System__: YAML-based model configurations with inheritance support

* __Model Registry__: Central management of models with capability-based filtering

* __Integration Ready__: Works with Pydantic AI, and more frameworks coming soon


## Next Steps

- [Configuration Guide](docs/configs/overview.md)
- [Parameter System](docs/parameters/overview.md)
- [Provider Documentation](docs/providers/overview.md)
- [Registry System](docs/registry.md)
