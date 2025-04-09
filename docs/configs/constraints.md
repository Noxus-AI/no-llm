# Model Constraints

Model constraints define the technical limitations and boundaries of a model's operation. These constraints are crucial for managing token usage and ensuring requests stay within model capabilities.

## Available Constraints

| Constraint | Type | Description |
|------------|------|-------------|
| `context_window` | int | Maximum total length of context (input + output) in tokens |
| `max_input_tokens` | int | Maximum number of tokens allowed in input |
| `max_output_tokens` | int | Maximum number of tokens allowed in output |

All constraints must be positive integers (>0).

## Usage Example

```python
from stdllm.config.model import ModelConstraints

constraints = ModelConstraints(
    context_window=8192,      # 8K context window
    max_input_tokens=7000,    # Maximum input size
    max_output_tokens=4000    # Maximum output size
)
```

## Input Size Estimation

The constraints system provides a quick way to estimate if text might exceed the input token limit:

```python
# Quick check if text might be too long
long_text = "..." # Your input text
if constraints.estimate_exceeds_input_limit(long_text):
    print("Warning: Text likely exceeds model's input token limit")
```

!!! note "Estimation Accuracy"
    The estimation uses a simple heuristic of 4 characters per token. While this provides a quick check, actual token counts may vary based on the specific tokenizer used by the model.
