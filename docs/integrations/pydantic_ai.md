# Pydantic AI Integration

no_llm provides seamless integration with [Pydantic AI](https://github.com/pydantic/pydantic-ai), allowing you to use any no_llm model through the Pydantic AI interface.

## Installation

```bash
pip install "no_llm[pydantic-ai]"
```

## Usage

The integration provides a `NoLLMModel` class that wraps no_llm models for use with Pydantic AI:

```python
from no_llm.integrations.pydantic_ai import NoLLMModel
from no_llm.registry import ModelRegistry
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

# Get models from registry
registry = ModelRegistry()
models = list(registry.list_models(provider="openai"))

# Create NoLLMModel with fallbacks
model = NoLLMModel(*models)

# Use with Pydantic AI
agent = Agent(model, model_settings=ModelSettings(temperature=0.7))
result = agent.run_sync("What is the capital of France?")
```

## Features

- **Model Fallbacks**: Automatically tries alternative models if the primary model fails
- **Parameter Validation**: Validates and merges parameters from both no_llm and Pydantic AI settings
- **Provider Support**: Works with all no_llm supported providers including:
  - OpenAI
  - Anthropic
  - Google Vertex AI
  - Mistral
  - Groq
  - And more

### Model Settings

The integration merges model settings from both no_llm and Pydantic AI:

```python
# no/llm parameters are merged with Pydantic AI settings, and validated with no/llm
agent = Agent(
    model,
    model_settings=ModelSettings(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1000
    )
)
```
