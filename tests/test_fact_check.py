import pytest
import warnings
from no_llm.registry import ModelRegistry


@pytest.fixture
def registry():
    return ModelRegistry()

@pytest.fixture
def litellm_facts():
    import requests
    response = requests.get("https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json")
    response.raise_for_status()
    return response.json()

# @pytest.mark.skip(reason="To be run locally")
def test_litellm_facts(subtests, registry: ModelRegistry, litellm_facts: dict):
    """Test that model configurations match LiteLLM facts"""
    
    for model in registry._models.values():
        with subtests.test(model=model.identity.id):
            if not model.integration_aliases:
                pytest.skip("No integration aliases")
                
            litellm_alias = model.integration_aliases.litellm
            if not litellm_alias:
                pytest.skip("No LiteLLM alias")
                
            if litellm_alias not in litellm_facts:
                warnings.warn(f"Model {model.identity.id} has litellm alias {litellm_alias} but no facts found in litellm")
                pytest.skip(f"No facts for {litellm_alias}")
                
            facts = litellm_facts[litellm_alias]
            
            required_keys = ["max_input_tokens", "max_output_tokens", "input_cost_per_token", "output_cost_per_token"]
            missing_keys = [key for key in required_keys if key not in facts]
            if missing_keys:
                warnings.warn(f"Model {litellm_alias} missing required keys in litellm facts: {missing_keys}")
                pytest.skip(f"Missing required keys: {missing_keys}")

            if model.metadata.pricing.token_prices:
                expected_input_price = facts["input_cost_per_token"] * 1000
                expected_output_price = facts["output_cost_per_token"] * 1000
                if expected_input_price == 0 or expected_output_price == 0:
                    pytest.skip("No pricing data")
                
                assert pytest.approx(model.metadata.pricing.token_prices.input_price_per_1k) == expected_input_price, \
                    f"Input price mismatch: expected {expected_input_price}, got {model.metadata.pricing.token_prices.input_price_per_1k}"
                
                assert pytest.approx(model.metadata.pricing.token_prices.output_price_per_1k) == expected_output_price, \
                    f"Output price mismatch: expected {expected_output_price}, got {model.metadata.pricing.token_prices.output_price_per_1k}"
            
            assert model.constraints.max_input_tokens == facts["max_input_tokens"], \
                f"Max input tokens mismatch: expected {facts['max_input_tokens']}, got {model.constraints.max_input_tokens}"
            
            assert model.constraints.max_output_tokens == facts["max_output_tokens"], \
                f"Max output tokens mismatch: expected {facts['max_output_tokens']}, got {model.constraints.max_output_tokens}"
            

def test_model_pricing_sanity(subtests, registry: ModelRegistry):
    """Test that model pricing makes sense"""
    
    for model in registry._models.values():
        with subtests.test(model=model.identity.id):
            if not model.metadata.pricing.token_prices:
                pytest.skip("No token prices")
            
            assert model.metadata.pricing.token_prices.input_price_per_1k >= 0, \
                f"Negative input price: {model.metadata.pricing.token_prices.input_price_per_1k}"
            
            assert model.metadata.pricing.token_prices.output_price_per_1k >= 0, \
                f"Negative output price: {model.metadata.pricing.token_prices.output_price_per_1k}"
