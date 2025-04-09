# Configuration Inheritance

stdllm supports inheritance for model configurations, allowing you to create custom model configurations with specialized parameters and behaviors.

## Creating Custom Configurations

To create a custom model configuration, inherit from `ModelConfiguration` and optionally define custom parameters:

```python
from stdllm.config import (
    ModelConfiguration,
    ConfigurableModelParameters,
    ModelIdentity,
    ModelMode,
    ParameterValue,
    ParameterVariant,
    RangeValidation
)

class CustomModelConfiguration(ModelConfiguration):
    """Custom model configuration with specific parameters"""
    
    # Define custom parameters class for YAML serialization
    class Parameters(ConfigurableModelParameters):
        temperature: ParameterValue[float] = Field(
            default_factory=lambda: ParameterValue(
                variant=ParameterVariant.VARIABLE,
                value=0.0,
                validation_rule=RangeValidation(min_value=0.0, max_value=2.0)
            )
        )
        top_p: ParameterValue[float] = Field(
            default_factory=lambda: ParameterValue(
                variant=ParameterVariant.FIXED,
                value=1.0
            )
        )
        # Mark unsupported parameters
        frequency_penalty: ParameterValue[float] = Field(
            default_factory=lambda: ParameterValue(
                variant=ParameterVariant.UNSUPPORTED,
                value=None
            )
        )

    # Define model configuration
    identity: ModelIdentity = ModelIdentity(
        id="custom-model",
        name="Custom Model",
        version="1.0",
        description="Custom model configuration",
        creator="Your Organization"
    )
    mode: ModelMode = ModelMode.CHAT
    parameters: ConfigurableModelParameters = Field(default_factory=Parameters)
```

## YAML Serialization

The custom Parameters class enables YAML configuration:

```yaml
parameters:
  temperature: 0.7  # Variable parameter
  top_p: 
    variant: fixed
    value: 1.0
  frequency_penalty: unsupported
```

!!! note "Parameter Class Inheritance"
    Inheriting from `ConfigurableModelParameters` ensures your custom parameters can be:
    - Serialized to/from YAML
    - Validated properly
    - Used with the standard parameter system

## Real-World Example

Here's a simplified version of the Claude 3.5 Haiku configuration:

```python
class Claude35HaikuConfiguration(ModelConfiguration):
    class Parameters(ConfigurableModelParameters):
        temperature: ParameterValue[float] = Field(
            default_factory=lambda: ParameterValue(
                variant=ParameterVariant.VARIABLE,
                value=0.0,
                validation_rule=RangeValidation(min_value=0.0, max_value=2.0)
            )
        )
        top_k: ParameterValue[int] = Field(
            default_factory=lambda: ParameterValue(
                variant=ParameterVariant.FIXED,
                value=40
            )
        )

    identity: ModelIdentity = ModelIdentity(
        id="claude-3.5-haiku",
        name="Claude 3.5 Haiku",
        version="2024.02",
        description="Fast and compact model for instant responses",
        creator="Anthropic"
    )

    mode: ModelMode = ModelMode.CHAT
    capabilities: set[ModelCapability] = {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.VISION
    }

    parameters: ConfigurableModelParameters = Field(default_factory=Parameters)
```