# Model Capabilities

Capabilities define what features a model supports, enhancing its core functionality defined by its [mode](mode.md).

## Available Capabilities

Models can support various capabilities that enhance their functionality:

| Capability | Description |
|------------|-------------|
| `STREAMING` | Ability to stream responses token by token |
| `FUNCTION_CALLING` | Support for function/tool calling in responses |
| `PARALLEL_FUNCTION_CALLING` | Ability to call multiple functions simultaneously |
| `VISION` | Support for processing image inputs |
| `SYSTEM_PROMPT` | Support for system-level prompting |
| `TOOLS` | Support for using external tools |
| `JSON_MODE` | Ability to output responses in JSON format |
| `STRUCTURED_OUTPUT` | Support for generating structured data outputs |
| `REASONING` | Advanced reasoning capabilities |
| `WEB_SEARCH` | Ability to perform web searches |

## Runtime Capability Validation

Capability checking is crucial for runtime validation before making API requests. Different models support different features, and attempting to use unsupported capabilities can result in errors or unexpected behavior. For example:

- OpenAI O1 models don't support system prompts
- Some models don't support function calling
- Vision capabilities are only available in specific model versions

By checking capabilities before making requests, you can:
- Prevent failed API calls
- Provide better error messages
- Adapt your application's behavior based on available features
- Handle model upgrades gracefully

## Filtering Models by Capability

You can list models that match specific capability requirements using the registry:

```python
from stdllm.config.enums import ModelCapability
from stdllm.registry import ModelRegistry, SetFilter

registry = ModelRegistry()

# List models with any of the specified capabilities
streaming_or_json_models = list(registry.list_models(
    capabilities={ModelCapability.STREAMING, ModelCapability.JSON_MODE}
))

# List models that have ALL specified capabilities
advanced_models = list(registry.list_models(
    capabilities=SetFilter(
        values={ModelCapability.STREAMING, ModelCapability.FUNCTION_CALLING, ModelCapability.JSON_MODE},
        mode="all"
    )
))
```

## Parameter Validation

Parameters are validated against model capabilities:

```python
from stdllm.config.model import ModelConfiguration
from stdllm.config.enums import ModelCapability

model_config = registry.get_model("gpt-4")

# Parameters will be validated against available capabilities
parameters = model_config.parameters
parameters.set_parameters(
    capabilities={ModelCapability.STREAMING, ModelCapability.FUNCTION_CALLING},
    temperature=0.7,
    include_reasoning=True  # This will be dropped if REASONING capability is not present
)

# Get validated parameters with explicit handling of unsupported features
validated_params = parameters.validate_parameters(
    capabilities={ModelCapability.STREAMING, ModelCapability.FUNCTION_CALLING},
    temperature=0.7,
    include_reasoning=True,  # Will raise UnsupportedParameterError if REASONING not available
    drop_unsupported=False   # Default is True which silently drops unsupported parameters
)
```

!!! note "Capability-Dependent Parameters"
    Some parameters are only available when specific capabilities are enabled:
    - `include_reasoning`: Requires `REASONING` capability
    - `reasoning_effort`: Requires `REASONING` capability
    - Function calling parameters: Require `FUNCTION_CALLING` capability

    When a capability-dependent parameter is used without the required capability, it will be either:
    1. Dropped silently (if `drop_unsupported=True`)
    2. Raise an `UnsupportedParameterError` (if `drop_unsupported=False`)

