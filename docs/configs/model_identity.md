# Model Identity

Model identity provides essential identification and descriptive information for each model in stdllm.

## Identity Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier for the model |
| `name` | str | Human-readable display name |
| `version` | str | Model version identifier |
| `description` | str | Detailed description of the model |
| `creator` | str | Organization or entity that created the model |
| `model_api_name` | str \| None | Optional provider-specific API name |

## Usage Example

```python
from stdllm.config.model import ModelIdentity

identity = ModelIdentity(
    id="gpt-4-turbo",
    name="GPT-4 Turbo",
    version="1.0",
    description="Advanced language model with improved performance",
    creator="OpenAI",
    model_api_name="gpt-4-0125-preview"  # Provider-specific name if different
)
```

!!! note "Model ID vs API Name"
    The `id` field is stdllm's internal identifier, while `model_api_name` is used when the provider's API requires a different name. If `model_api_name` is not set, the `id` is used for API calls.

!!! tip "Versioning"
    Use semantic versioning in the `version` field to track model updates and ensure compatibility:
    ```python
    identity = ModelIdentity(
        id="gpt-4",
        version="1.2.3",  # major.minor.patch
        # ... other fields ...
    )
    ```
