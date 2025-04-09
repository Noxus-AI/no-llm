# Model Pricing

stdllm supports both token-based and character-based pricing models for LLM usage calculations.

## Pricing Models

### Token-Based Pricing

Token-based pricing is the most common model, where costs are calculated per thousand tokens:

```python
from stdllm.config.metadata import TokenPrices

pricing = TokenPrices(
    input_price_per_1k=0.01,  # $0.01 per 1000 input tokens
    output_price_per_1k=0.03   # $0.03 per 1000 output tokens
)
```

### Character-Based Pricing

Some models use character-based pricing instead:

```python
from stdllm.config.metadata import CharacterPrices

pricing = CharacterPrices(
    input_price_per_1k=0.001,  # $0.001 per 1000 input characters
    output_price_per_1k=0.002  # $0.002 per 1000 output characters
)
```

## Cost Calculation

You can calculate costs using either the model configuration or pricing models directly:

```python
# Using ModelConfiguration
model_config = registry.get_model("gpt-4")
input_cost, output_cost = model_config.calculate_cost(
    input_tokens=1000,
    output_tokens=500
)

# Using pricing models directly
pricing = ModelPricing(
    token_prices=TokenPrices(
        input_price_per_1k=0.01,
        output_price_per_1k=0.03
    )
)
input_cost, output_cost = pricing.calculate_cost(
    input_size=1000,    # tokens or characters depending on pricing type
    output_size=500
)
```

!!! note "Pricing Configuration"
    Models must have either token-based or character-based pricing configured. Attempting to configure both or neither will raise an `InvalidPricingConfigError`.

!!! warning "Token Counting Limitations"
    The current version has some limitations in token counting:
    - Reasoning steps tokens are not counted separately
    - Cached response tokens are not tracked
    
    These features will be added in future releases.
