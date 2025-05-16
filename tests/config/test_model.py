from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import Field
from pydantic_ai.providers.openai import OpenAIProvider

from no_llm.config.enums import ModelCapability, ModelMode
from no_llm.config.errors import MissingCapabilitiesError
from no_llm.config.model import ModelConfiguration
from no_llm.config.parameters import ConfigurableModelParameters, ModelParameters
from no_llm.errors import InvalidPricingConfigError
from no_llm.providers import EnvVar, Provider


class MockProvider(Provider):
    type: str = "test"
    name: str = "Test Provider"
    api_key: EnvVar[str] = Field(default_factory=lambda: EnvVar[str]("$TEST_KEY"))
    _iterator_index: int = 0

    def to_pydantic(self) -> OpenAIProvider:
        return OpenAIProvider(api_key=str(self.api_key))


def create_test_config() -> dict:
    """Create a test model configuration"""
    return {
        "identity": {
            "id": "test-model",
            "name": "Test Model",
            "version": "1.0.0",
            "description": "Test model",
            "creator": "test",
        },
        "provider_id": "test",
        "mode": "chat",
        "capabilities": ["streaming"],
        "constraints": {
            "context_window": 1024,
            "max_input_tokens": 1000,
            "max_output_tokens": 500,
        },
        "properties": {
            "speed": {"score": 50.0, "label": "test", "description": "test"},
            "quality": {"score": 50.0, "label": "test", "description": "test"},
        },
        "metadata": {
            "privacy_level": ["basic"],
            "pricing": {
                "token_prices": {
                    "input_price_per_1k": 0.03,
                    "output_price_per_1k": 0.06,
                }
            },
            "release_date": datetime.now(timezone.utc).isoformat(),
        },
    }


def create_test_model():
    config = create_test_config()
    model = ModelConfiguration.model_validate(config)
    model.providers = []
    return model


def test_model_config_basic():
    """Test basic model configuration"""
    config = create_test_config()
    model = ModelConfiguration.model_validate(config)

    assert model.identity.id == "test-model"
    # assert model.provider_id == "test"
    assert model.mode == ModelMode.CHAT
    assert ModelCapability.STREAMING in model.capabilities


def test_model_pricing_validation():
    """Test model pricing validation"""
    # Test valid token pricing
    config = create_test_config()
    model = ModelConfiguration.model_validate(config)
    assert model.metadata.pricing.token_prices is not None
    assert model.metadata.pricing.token_prices.input_price_per_1k == 0.03

    # Test invalid pricing (neither token nor character prices)
    config["metadata"]["pricing"] = {"token_prices": None, "character_prices": None}
    with pytest.raises(InvalidPricingConfigError):
        ModelConfiguration.model_validate(config)


def test_model_provider_iteration(monkeypatch):
    """Test iterating through model providers"""
    model = create_test_model()
    monkeypatch.setenv("TEST_KEY", "test-value")

    provider1 = MockProvider(type="test", name="Provider 1")
    provider2 = MockProvider(type="test", name="Provider 2")
    provider3 = MockProvider(type="test", name="Provider 3")

    # Cast list to Sequence[Providers] to satisfy type checker
    model.providers = [provider1, provider2, provider3]  # type: ignore

    iterated_providers = list(model.iter())
    assert len(iterated_providers) == 3
    assert iterated_providers[0].name == "Provider 1"
    assert iterated_providers[1].name == "Provider 2"
    assert iterated_providers[2].name == "Provider 3"


def test_model_provider_iteration_empty():
    """Test iterating with no providers"""
    model = create_test_model()
    model.providers = []

    assert list(model.iter()) == []


def test_model_capability_checks():
    """Test model capability checking methods"""
    model = create_test_model()
    model.capabilities = {ModelCapability.STREAMING, ModelCapability.FUNCTION_CALLING}

    # Test check_capabilities with "any" mode
    assert model.check_capabilities({ModelCapability.STREAMING}, mode="any")
    assert model.check_capabilities({ModelCapability.FUNCTION_CALLING}, mode="any")
    assert model.check_capabilities(
        {ModelCapability.STREAMING, ModelCapability.FUNCTION_CALLING}, mode="any"
    )
    assert not model.check_capabilities({ModelCapability.JSON_MODE}, mode="any")

    # Test check_capabilities with "all" mode
    assert model.check_capabilities({ModelCapability.STREAMING}, mode="all")
    assert model.check_capabilities(
        {ModelCapability.STREAMING, ModelCapability.FUNCTION_CALLING}, mode="all"
    )
    assert not model.check_capabilities(
        {ModelCapability.STREAMING, ModelCapability.JSON_MODE}, mode="all"
    )


def test_model_assert_capabilities():
    """Test model capability assertion method"""
    model = create_test_model()
    model.capabilities = {ModelCapability.STREAMING, ModelCapability.FUNCTION_CALLING}

    # Should not raise for valid capabilities
    model.assert_capabilities({ModelCapability.STREAMING}, mode="any")
    model.assert_capabilities(
        {ModelCapability.STREAMING, ModelCapability.FUNCTION_CALLING}, mode="all"
    )

    # Should raise for missing capabilities
    with pytest.raises(MissingCapabilitiesError):
        model.assert_capabilities({ModelCapability.WEB_SEARCH}, mode="any")

    with pytest.raises(MissingCapabilitiesError):
        model.assert_capabilities(
            {ModelCapability.STREAMING, ModelCapability.WEB_SEARCH}, mode="all"
        )


def test_model_cost_calculation():
    """Test model cost calculation"""
    model = create_test_model()

    # Test with token pricing
    input_cost, output_cost = model.calculate_cost(1000, 500)
    assert input_cost == 0.03  # 1000 tokens * 0.03/1000
    assert output_cost == 0.03  # 500 tokens * 0.06/1000

    # Test with character pricing (should raise NotImplementedError)
    model.metadata.pricing.token_prices = None
    with pytest.raises(NotImplementedError):
        model.calculate_cost(1000, 500)


def test_model_parameter_handling():
    """Test model parameter handling methods"""
    model = create_test_model()

    # Create new model with updated parameters
    new_params = ModelParameters(temperature=0.7, top_p=0.9)
    new_model = model.model_copy(deep=True)
    new_model.set_parameters(new_params)

    assert new_model.parameters.temperature.value == 0.7
    assert new_model.parameters.top_p.value == 0.9
    assert new_model is not model  # Should be a new instance

    # Test set_parameters directly
    params = ModelParameters(temperature=0.4, top_p=0.7)
    model.set_parameters(params)
    assert model.parameters.temperature.value == 0.4
    assert model.parameters.top_p.value == 0.7

    # Test getting parameters
    params = model.parameters
    assert isinstance(params, ConfigurableModelParameters)


def test_model_constraints():
    """Test model constraints"""
    model = create_test_model()

    # Test estimate_exceeds_input_limit
    short_text = "This is a short text"
    # Use a much larger multiplier to ensure we exceed the limit
    long_text = "x" * (
        model.constraints.max_input_tokens * 8
    )  # Using 8 chars per token to be safe

    assert not model.constraints.estimate_exceeds_input_limit(short_text)
    assert model.constraints.estimate_exceeds_input_limit(long_text)


def test_model_provider_iteration_with_env_vars(monkeypatch):
    """Test iterating through providers with varying environment states"""
    model = create_test_model()

    provider1 = MockProvider(
        type="test", name="Provider 1", api_key=EnvVar[str]("$TEST_KEY_1")
    )
    provider2 = MockProvider(
        type="test", name="Provider 2", api_key=EnvVar[str]("$TEST_KEY_2")
    )
    provider3 = MockProvider(
        type="test", name="Provider 3", api_key=EnvVar[str]("$TEST_KEY_3")
    )

    # Cast list to Sequence[Providers] to satisfy type checker
    model.providers = [provider1, provider2, provider3]  # type: ignore

    monkeypatch.delenv("TEST_KEY_1", raising=False)
    monkeypatch.delenv("TEST_KEY_2", raising=False)
    monkeypatch.setenv("TEST_KEY_3", "test-value")

    providers = list(model.iter())
    assert len(providers) == 1
    assert providers[0].name == "Provider 3"
