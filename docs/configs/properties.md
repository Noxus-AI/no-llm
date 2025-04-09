# Model Properties

Model properties provide informational metrics about model performance and quality. These properties are intended for display purposes to help users make informed decisions about model selection.

!!! note "Display Only"
    Properties are informational metrics and do not affect model behavior. They are designed to help users understand and compare models.

## Speed Properties

Speed properties indicate the model's processing performance:

```python
from stdllm.config.properties import SpeedProperties

speed = SpeedProperties(
    score=1000.0,        # Tokens per second
    label="Fast",        # Quick reference label
    description="Optimized for real-time applications"
)
```

## Quality Properties

Quality properties reflect the model's output quality on a standardized scale:

```python
from stdllm.config.properties import QualityProperties

quality = QualityProperties(
    score=95.0,          # Score from 0-100
    label="High",        # Quality tier
    description="State-of-the-art performance on standard benchmarks"
)
```

!!! tip "Custom Properties"
    You can extend the properties system by inheriting from the base Pydantic models:
    ```python
    from stdllm.config.properties import ModelProperties
    from pydantic import BaseModel, Field

    class CustomMetrics(BaseModel):
        accuracy: float = Field(description="Model accuracy score")
        latency: float = Field(description="Average response time")

    class ExtendedProperties(ModelProperties):
        metrics: CustomMetrics
    ```
