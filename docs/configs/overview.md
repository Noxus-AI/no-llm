# Configuration Overview

The stdllm configuration system provides a comprehensive way to define and manage model configurations. This overview will help you understand how the different components work together.

## Core Components

The configuration system consists of several key components:

- **[Model Identity](model_identity.md)**: Basic model information like ID, name, and version
- **[Model Mode](mode.md)**: Primary function of the model (chat, completion, embedding, etc.)
- **[Capabilities](capabilities.md)**: Features supported by the model
- **[Constraints](constraints.md)**: Technical limitations like context window and token limits
- **[Properties](properties.md)**: Performance and quality metrics
- **[Privacy](privacy.md)**: Compliance and data protection levels
- **[Pricing](pricing.md)**: Cost calculation configuration

## Built-in Models

stdllm includes configurations for popular models:

### Anthropic Models
- Claude 3 (Opus, Sonnet, Haiku)
- Claude 3.5 (Sonnet, Haiku)
- Claude 3.7 Sonnet

### Google Models
- Gemini 1.5 (Pro, Flash)
- Gemini 2.0 (Pro, Flash, Flash Lite, Flash Thinking)
- Gemini 2.5 Pro

### OpenAI Models
- GPT-4 (Base, O, O Mini)
- GPT-3.5 Turbo
- O1/O3 Mini

### Other Providers
- DeepSeek (Chat, Reasoner, R1 Llama 70B)
- Llama 3 (405B, 70B)
- Mistral (Large, Nemo)
- Groq Mixtral
- Perplexity Sonar (Large, Small)

## Model Configuration API

```python
from stdllm.config.model import ModelConfiguration
from stdllm.config.enums import ModelMode, ModelCapability

model = ModelConfiguration(
    identity=ModelIdentity(id="gpt-4", name="GPT-4", version="1.0.0"),
    mode=ModelMode.CHAT,
    capabilities={ModelCapability.STREAMING},
    providers=[OpenAIProvider(), AzureProvider()]  # Multiple providers supported
)


# Parameter Management
model.set_parameters(ModelParameters(temperature=0.7))
params = model.get_parameters()  # Get current parameters
new_model = model.from_parameters(temperature=0.8)  # Create new config with parameters

# Capability Checking
model.check_capabilities({ModelCapability.STREAMING})  # Returns bool
model.assert_capabilities({ModelCapability.STREAMING})  # Raises if missing

# Cost Calculation
input_cost, output_cost = model.calculate_cost(
    input_tokens=1000,
    output_tokens=500
)
```

## YAML Configuration

Models can also be configured using YAML files:

```yaml
identity:
  id: gpt-4
  name: GPT-4
  version: 1.0.0

mode: chat
capabilities: 
  - streaming
  - function_calling

providers:
  - type: openai
    api_key: ${OPENAI_API_KEY}
  - type: azure
    api_key: ${AZURE_API_KEY}
    deployment: gpt4

constraints:
  context_window: 8192
  max_input_tokens: 7000
  max_output_tokens: 4000
```

## Provider Iteration

Models support iterating through providers and their variants (e.g., different locations for cloud providers):

```python
# Iterate through all providers and their variants
for provider in model.iter():
    try:
        response = call_model_with_provider(provider)
        break  # Success, stop trying other providers
    except Exception:
        continue  # Try next provider
```

For cloud providers like Vertex AI, each location becomes a variant:

```python
vertex_provider = VertexProvider(
    project_id="my-project",
    locations=["us-central1", "europe-west1"]
)

# Will yield a provider instance for each location
for provider in vertex_provider.iter():
    print(provider.current)  # Access current location
```

See the specific component documentation for detailed information about each configuration aspect.