import pytest

from no_llm.config.enums import ModelCapability
from no_llm.config.errors import (
    FixedParameterError,
    InvalidEnumError,
    InvalidRangeError,
    UnsupportedParameterError,
)
from no_llm.config.parameters import (
    NOT_GIVEN,
    ConfigurableModelParameters,
    EnumValidation,
    ModelParameters,
    ParameterValue,
    ParameterVariant,
    RangeValidation,
    ValidationRule,
)


def test_parameter_value_yaml():
    # Variable value with range
    param = ParameterValue.model_validate(
        {"variant": ParameterVariant.VARIABLE, "value": 0.7, "range": [0.0, 1.0]}
    )
    assert param.variant == ParameterVariant.VARIABLE
    assert param.value == 0.7
    assert param.get() == 0.7
    assert param.is_variable()


def test_model_parameters_yaml():
    """Test loading parameters from YAML-like dict"""
    params = ConfigurableModelParameters()

    # Set parameters one by one
    params.temperature = ParameterValue(variant=ParameterVariant.FIXED, value=0.7)
    params.top_p = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value=0.9,
        validation_rule=RangeValidation(min_value=0.0, max_value=1.0),
    )
    params.frequency_penalty = ParameterValue(
        variant=ParameterVariant.VARIABLE, value=0.5
    )
    params.presence_penalty = ParameterValue(
        variant=ParameterVariant.VARIABLE, value=None
    )
    params.max_tokens = ParameterValue(variant=ParameterVariant.FIXED, value=100)
    params.include_reasoning = ParameterValue(
        variant=ParameterVariant.FIXED, value=True
    )

    assert params.temperature.value == 0.7
    assert params.top_p.value == 0.9
    assert params.frequency_penalty.value == 0.5
    assert params.max_tokens.value == 100
    assert params.include_reasoning.value is True


def test_parameter_validation():
    """Test parameter validation rules"""
    # Test range validation
    with pytest.raises(InvalidRangeError):
        param = ParameterValue(
            variant=ParameterVariant.VARIABLE,
            value=3.0,
            validation_rule=RangeValidation(min_value=0.0, max_value=2.0),
        )
        param.validate_model()


def test_enum_validation():
    # Test enum validation
    validator = EnumValidation(allowed_values=["low", "medium", "high"])

    # Valid value
    validator.validate_value("medium")

    # Invalid value
    with pytest.raises(InvalidEnumError) as exc_info:
        validator.validate_value("invalid")
    assert "Value not in allowed values" in str(exc_info.value)
    assert "['low', 'medium', 'high']" in str(exc_info.value)


def test_capability_requirements():
    # Test parameter with no capability requirement
    param = ParameterValue(variant=ParameterVariant.VARIABLE, value=0.7)
    assert param.required_capability is None
    assert not param.is_unsupported()

    # Test parameter with capability requirement
    param = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value=True,
        required_capability=ModelCapability.REASONING,
    )
    assert param.required_capability == ModelCapability.REASONING
    assert not param.is_unsupported()

    # Test capability check when capability is present
    param = param.check_capability({ModelCapability.REASONING})
    assert not param.is_unsupported()
    assert param.get() is True

    # Test capability check when capability is missing
    param = param.check_capability({ModelCapability.STREAMING})
    assert param.is_unsupported()
    assert param.get() == "UNSUPPORTED"


def test_model_parameters_capabilities():
    """Test parameter capability requirements"""
    params = ConfigurableModelParameters()

    params.temperature = ParameterValue(variant=ParameterVariant.FIXED, value=0.7)
    params.include_reasoning = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value=True,
        required_capability=ModelCapability.REASONING,
    )
    params.reasoning_effort = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value="medium",
        required_capability=ModelCapability.REASONING,
    )

    assert params.include_reasoning.value is True
    assert params.reasoning_effort.value == "medium"
    assert params.temperature.value == 0.7


def test_model_parameters_unsupported_error():
    """Test handling of unsupported parameters"""
    from no_llm.settings import ValidationMode, settings

    settings.validation_mode = ValidationMode.ERROR
    params = ConfigurableModelParameters()

    params.temperature = ParameterValue(variant=ParameterVariant.VARIABLE, value=0.7)
    params.include_reasoning = ParameterValue(
        variant=ParameterVariant.UNSUPPORTED,
        value=None,
        required_capability=ModelCapability.REASONING,
    )

    # Should raise error when trying to get value of unsupported parameter
    with pytest.raises(UnsupportedParameterError):
        _ = params.validate_parameter("include_reasoning", None)
    settings.validation_mode = ValidationMode.CLAMP


def test_fixed_parameter_modification():
    """Test modification of fixed parameters"""
    from no_llm.settings import ValidationMode, settings

    settings.validation_mode = ValidationMode.ERROR
    params = ConfigurableModelParameters(
        temperature=ParameterValue(variant=ParameterVariant.FIXED, value=0.7),
    )

    # Should raise when trying to modify fixed parameter's value
    with pytest.raises(FixedParameterError):
        params.temperature = 0.8
    settings.validation_mode = ValidationMode.CLAMP


def test_multiple_validation():
    """Test multiple validation errors are caught"""
    params = ConfigurableModelParameters()

    # Add parameters with invalid values
    with pytest.raises(InvalidRangeError):
        params.temperature = ParameterValue(
            variant=ParameterVariant.VARIABLE,
            value=3.0,
            validation_rule=RangeValidation(min_value=0.0, max_value=2.0),
        )


def test_parameter_validation_rules():
    """Test that validation rules are properly applied"""
    params = ConfigurableModelParameters()

    # Add parameter with range validation
    params.temperature = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value=0.7,
        validation_rule=RangeValidation(min_value=0.0, max_value=2.0),
    )

    # Should fail when trying to set outside range
    with pytest.raises(InvalidRangeError):
        params.temperature = ParameterValue(
            variant=ParameterVariant.VARIABLE,
            value=3.0,
            validation_rule=RangeValidation(min_value=0.0, max_value=2.0),
        )


def test_model_parameters_direct_yaml():
    """Test that parameters can be loaded directly through model configuration"""
    from no_llm.config.model import ModelConfiguration

    config = {
        "identity": {
            "id": "test-model",
            "name": "Test Model",
            "version": "1.0.0",
            "description": "Test model",
            "creator": "test",  # Added required field
        },
        "provider_id": "test",
        "mode": "chat",
        "capabilities": ["streaming", "reasoning"],
        "constraints": {
            "context_window": 1024,
            "max_input_tokens": 1000,
            "max_output_tokens": 500,
            "token_encoding": "test",
        },
        "properties": {
            "speed": {"score": 50.0, "label": "test", "description": "test"},
            "quality": {"score": 50.0, "label": "test", "description": "test"},
        },
        "parameters": {
            "temperature": 0.7,
            "top_p": {"value": 0.9, "range": [0.0, 1.0]},
            "frequency_penalty": {"value": 0.5},
            "presence_penalty": {"value": 0.5},
            "max_tokens": 100,
            "top_k": "unsupported",
            "include_reasoning": {"value": True, "required_capability": "reasoning"},
        },
        "metadata": {
            "privacy_level": ["basic"],  # Changed to list
            "pricing": {
                "token_prices": {
                    "input_price_per_1k": 0.01,
                    "output_price_per_1k": 0.02,
                }
            },
            "release_date": "2024-01-01T00:00:00Z",
        },
    }

    model = ModelConfiguration.from_config(config)

    # Test that parameters were loaded correctly
    params = model.parameters

    # Test fixed value
    assert params.temperature.is_fixed()
    assert params.temperature.get() == 0.7

    # Test variable with range
    assert params.top_p.is_variable()
    assert params.top_p.get() == 0.9
    assert params.top_p.validation_rule is not None
    assert params.top_p.validation_rule.min_value == 0.0
    assert params.top_p.validation_rule.max_value == 1.0

    # Test variable without range
    assert params.frequency_penalty.is_variable()
    assert params.frequency_penalty.get() == 0.5

    # Test variable with None value
    assert params.presence_penalty.is_variable()
    assert params.presence_penalty.get() == 0.5

    # Test fixed value
    assert params.max_tokens.get() == 100

    # Test unsupported parameter
    assert params.top_k.is_unsupported()
    assert params.top_k.get() == "UNSUPPORTED"

    # Test parameter with capability
    assert params.include_reasoning.required_capability == ModelCapability.REASONING
    assert params.include_reasoning.get() is True

    # Test parameter validation
    values = model.parameters.model_dump()
    assert values["temperature"] == 0.7
    assert values["top_p"] == 0.9
    assert values["frequency_penalty"] == 0.5
    assert values["max_tokens"] == 100
    assert values["include_reasoning"] is True
    assert "top_k" not in values  # Unsupported parameter should be dropped


def test_model_parameters_validation():
    """Test parameter validation through model configuration"""
    from no_llm.config.model import ModelConfiguration

    config = {
        "identity": {
            "id": "test-model",
            "name": "Test Model",
            "version": "1.0.0",
            "description": "Test model",
            "creator": "test",  # Added required field
        },
        "provider_id": "test",
        "mode": "chat",
        "capabilities": ["streaming"],
        "constraints": {
            "context_window": 1024,
            "max_input_tokens": 1000,
            "max_output_tokens": 500,
            "token_encoding": "test",
        },
        "properties": {
            "speed": {"score": 50.0, "label": "test", "description": "test"},
            "quality": {"score": 50.0, "label": "test", "description": "test"},
        },
        "parameters": {
            "temperature": {"variant": "fixed", "value": 0.7},
            "top_p": {"value": 0.9, "range": [0.0, 1.0]},
        },
        "metadata": {
            "privacy_level": ["basic"],  # Changed to list
            "pricing": {
                "token_prices": {
                    "input_price_per_1k": 0.01,
                    "output_price_per_1k": 0.02,
                }
            },
            "release_date": "2024-01-01T00:00:00Z",
        },
    }

    model = ModelConfiguration.from_config(config)

    # Test fixed parameter modification
    with pytest.raises(FixedParameterError) as exc_info:
        model.parameters.validate_parameter("temperature", 0.8)
    assert "Cannot modify fixed parameter 'temperature'" in str(exc_info.value)

    # Test range validation
    with pytest.raises(InvalidRangeError) as exc_info:
        model.parameters.validate_parameter("top_p", 1.5)  # Outside [0.0, 1.0]
    assert "Value 1.5 outside range [0.0, 1.0]" in str(exc_info.value)

    # Test valid modification
    model.parameters.top_p = 0.5
    assert model.parameters.top_p.value == 0.5  # New value within range
    assert model.parameters.temperature.value == 0.7  # Fixed value unchanged


def test_parameter_conversion_flow():
    """Test conversion from ConfigurableModelParameters to ModelParameters"""
    config_params = ConfigurableModelParameters()

    config_params.temperature = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value=0.7,
        validation_rule=RangeValidation(min_value=0.0, max_value=2.0),
    )
    config_params.include_reasoning = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value=True,
        required_capability=ModelCapability.REASONING,
    )

    # Test validation with required capability
    validated = config_params.model_dump()
    model_params = ModelParameters(**validated)
    assert model_params.temperature == 0.7
    assert model_params.include_reasoning is True


def test_parameter_override_validation():
    """Test parameter validation during conversion"""
    config_params = ConfigurableModelParameters()

    config_params.temperature = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value=0.7,
        validation_rule=RangeValidation(min_value=0.0, max_value=2.0),
    )
    config_params.top_p = ParameterValue(variant=ParameterVariant.FIXED, value=0.9)

    # Test valid override
    config_params.validate_parameter("temperature", 1.5)
    config_params.temperature = 1.5
    assert config_params.temperature.value == 1.5
    assert config_params.top_p.value == 0.9


def test_not_given_handling():
    """Test handling of NOT_GIVEN vs None values"""
    config_params = ConfigurableModelParameters()

    config_params.temperature = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value=0.7,
    )
    config_params.max_tokens = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value=None,
    )
    config_params.top_p = ParameterValue(
        variant=ParameterVariant.UNSUPPORTED,
        value=None,
    )

    # Test validation preserves NOT_GIVEN
    config_params.validate_parameter("temperature", 0.8)
    config_params.temperature = 0.8
    assert config_params.temperature.value == 0.8
    assert config_params.max_tokens.value is None
    assert config_params.top_p.variant == ParameterVariant.UNSUPPORTED


def test_model_parameters_merge():
    """Test merging ModelParameters instances"""
    base_params = ModelParameters(temperature=0.7, max_tokens=100)

    override_params = ModelParameters(temperature=0.8, top_p=0.9)

    # Test merging with & - left-hand takes precedence
    merged = override_params & base_params
    assert merged.temperature == 0.8  # From override
    assert merged.max_tokens == 100  # From base
    assert merged.top_p == 0.9  # From override


def test_parameter_dump_and_get():
    """Test parameter dumping and getting methods"""
    config_params = ConfigurableModelParameters.from_config(
        {
            "temperature": {"value": 0.7},
            "top_p": "unsupported",
        }
    )

    # Get parameters (excludes None values and unsupported)
    params = config_params.model_dump()
    model_params = ModelParameters(**params)
    dumped = model_params.dump_parameters(with_defaults=False)
    assert dumped == {"temperature": 0.7}  # Only non-default values


def test_validation_rule_base():
    """Test base ValidationRule class"""
    rule = ValidationRule()
    # Base class validate_value should not raise
    rule.validate_value("any value")
    rule.validate_value(None)


def test_parameter_value_serialization():
    """Test ParameterValue serialization"""
    # Test unsupported parameter
    param = ParameterValue(variant=ParameterVariant.UNSUPPORTED, value=None)
    assert param.serialize_model() == "unsupported"

    # Test fixed value
    param = ParameterValue(variant=ParameterVariant.FIXED, value=0.7)
    assert param.serialize_model() == 0.7

    # Test variable with range validation
    param = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value=0.7,
        validation_rule=RangeValidation(min_value=0.0, max_value=1.0),
    )
    serialized = param.serialize_model()
    assert serialized["value"] == 0.7
    assert serialized["range"] == [0.0, 1.0]

    # Test variable with enum validation
    param = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value="medium",
        validation_rule=EnumValidation(allowed_values=["low", "medium", "high"]),
    )
    serialized = param.serialize_model()
    assert serialized["value"] == "medium"
    assert serialized["values"] == ["low", "medium", "high"]


def test_configurable_parameters_serialization():
    """Test parameter serialization"""
    params = ConfigurableModelParameters()

    params.temperature = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value=0.7,
        validation_rule=RangeValidation(min_value=0.0, max_value=1.0),
    )

    serialized = params.serialize_model()  # Use serialize_model instead of model_dump
    assert isinstance(serialized["temperature"], float)
    assert serialized["temperature"] == 0.7


def test_parameter_value_direct_serialization():
    """Test ParameterValue direct serialization"""
    # Test unsupported parameter
    param = ParameterValue(variant=ParameterVariant.UNSUPPORTED, value=None)
    serialized = param.serialize_model()
    assert serialized == "unsupported"

    # Test fixed value
    param = ParameterValue(variant=ParameterVariant.FIXED, value=0.7)
    serialized = param.serialize_model()
    assert serialized == 0.7

    # Test variable with range validation
    param = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value=0.7,
        validation_rule=RangeValidation(min_value=0.0, max_value=1.0),
    )
    serialized = param.serialize_model()
    assert isinstance(serialized, dict)
    assert serialized["value"] == 0.7
    assert serialized["range"] == [0.0, 1.0]


def test_model_parameters_with_overrides():
    """Test ModelParameters with model-specific overrides"""
    params = ModelParameters(
        temperature=0.7,
        model_override={
            "gpt-4": ModelParameters(temperature=0.8),
            "gpt-3.5": ModelParameters(temperature=0.6),
        },
    )

    # Test dumping without overrides
    base_params = params.dump_parameters(with_defaults=False)
    assert base_params["temperature"] == 0.7

    # Test dumping with specific model override
    gpt4_params = params.dump_parameters(with_defaults=False, model_override="gpt-4")
    assert gpt4_params["temperature"] == 0.8

    # Test dumping with non-existent model override
    other_params = params.dump_parameters(
        with_defaults=False, model_override="other-model"
    )
    assert other_params["temperature"] == 0.7


def test_parameter_value_not_given():
    """Test handling of NOT_GIVEN values in validation"""
    param = ParameterValue(
        variant=ParameterVariant.VARIABLE,
        value="NOT_GIVEN",
        validation_rule=RangeValidation(min_value=0.0, max_value=1.0),
    )

    # NOT_GIVEN should pass validation
    param.validate_model()

    # NOT_GIVEN should be preserved in get()
    assert param.get() == "NOT_GIVEN"


def test_configurable_parameters_validation_modes():
    """Test different validation modes for ConfigurableModelParameters"""
    import no_llm.settings as settings
    from no_llm.settings import ValidationMode

    settings.settings.validation_mode = ValidationMode.CLAMP

    params = ConfigurableModelParameters()
    param = ParameterValue[float](
        variant=ParameterVariant.VARIABLE,
        value=0.7,
        validation_rule=RangeValidation(min_value=0.0, max_value=1.0),
    )
    params.temperature = param

    # Test CLAMP mode
    params.temperature = 1.5  # Should be clamped to 1.0
    assert params.temperature.value == 1.0

    params.temperature = -0.5  # Should be clamped to 0.0
    assert params.temperature.value == 0.0


def test_subclassed_model_parameters():
    """Test parameter validation and dumping with subclassed model parameters"""
    from no_llm.models.openai.o3_mini import O3MiniConfiguration

    # Create an instance of the O3MiniConfiguration
    model_config = O3MiniConfiguration()

    # Verify the model has the custom Parameters class
    assert isinstance(model_config.parameters, O3MiniConfiguration.Parameters)

    # Test that fixed parameters have correct values
    assert model_config.parameters.temperature.variant == ParameterVariant.FIXED
    assert model_config.parameters.temperature.value == 1.0
    assert model_config.parameters.top_p.variant == ParameterVariant.FIXED
    assert model_config.parameters.top_p.value == 1.0
    assert model_config.parameters.frequency_penalty.variant == ParameterVariant.FIXED
    assert model_config.parameters.frequency_penalty.value == 0.0

    # Test that unsupported parameters are marked correctly
    assert model_config.parameters.top_k.variant == ParameterVariant.UNSUPPORTED
    assert model_config.parameters.top_k.value == NOT_GIVEN
    assert (
        model_config.parameters.presence_penalty.variant == ParameterVariant.UNSUPPORTED
    )
    assert model_config.parameters.presence_penalty.value == NOT_GIVEN

    # Test parameter validation - should fail for fixed parameters
    with pytest.raises(FixedParameterError):
        model_config.parameters.validate_parameter("temperature", 0.8)

    with pytest.raises(FixedParameterError):
        model_config.parameters.validate_parameter("top_p", 0.9)

    with pytest.raises(FixedParameterError):
        model_config.parameters.validate_parameter("frequency_penalty", 0.5)

    # Test parameter dumping - should exclude unsupported parameters
    dumped_params = model_config.parameters.model_dump()

    # Should include fixed parameters with their values
    assert "temperature" in dumped_params
    assert dumped_params["temperature"] == 1.0
    assert "top_p" in dumped_params
    assert dumped_params["top_p"] == 1.0
    assert "frequency_penalty" in dumped_params
    assert dumped_params["frequency_penalty"] == 0.0

    # Should exclude unsupported parameters
    assert "top_k" not in dumped_params
    assert "presence_penalty" not in dumped_params

    # Test conversion to ModelParameters
    model_params = ModelParameters(**dumped_params)
    assert model_params.temperature == 1.0
    assert model_params.top_p == 1.0
    assert model_params.frequency_penalty == 0.0

    # Test that ModelParameters dump excludes None/unsupported values
    final_dump = model_params.dump_parameters(with_defaults=False)
    assert "temperature" in final_dump
    assert "top_p" in final_dump
    assert "frequency_penalty" in final_dump
    assert "top_k" not in final_dump
    assert "presence_penalty" not in final_dump


def test_model_loading_with_base_config_parameters(tmp_path):
    """Test parameter validation when loading model from config directory with base_config"""
    from no_llm.registry import ModelRegistry

    # Create config directory structure
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    models_dir = config_dir / "models"
    models_dir.mkdir()

    # Create o3-mini-low.yml config file that uses base_config
    config_content = """
providers:
  - type: openai
    name: OpenAI
    api_key: $OPENAI_API_KEY

identity:
  id: o3-mini-low
  base_config: o3-mini
  name: O3 Mini Low

parameters:
  reasoning_effort: low

extra:
  spotflow_visible: true
"""

    config_file = models_dir / "o3-mini-low.yml"
    config_file.write_text(config_content.strip())

    # Create registry and load from config directory
    registry = ModelRegistry(config_dir)

    # Get the loaded model configuration
    model_config = registry.get_model("o3-mini-low")

    # Verify it has the correct identity
    assert model_config.identity.id == "o3-mini-low"
    assert model_config.identity.name == "O3 Mini Low"

    # Verify it inherits from the base o3-mini configuration
    assert isinstance(
        model_config.parameters, type(registry.get_model("o3-mini").parameters)
    )

    # Test that base config parameters are preserved with correct variants
    assert model_config.parameters.temperature.variant == ParameterVariant.FIXED
    assert model_config.parameters.temperature.value == 1.0
    assert model_config.parameters.top_p.variant == ParameterVariant.FIXED
    assert model_config.parameters.top_p.value == 1.0
    assert model_config.parameters.frequency_penalty.variant == ParameterVariant.FIXED
    assert model_config.parameters.frequency_penalty.value == 0.0

    # Test that unsupported parameters remain unsupported
    assert model_config.parameters.top_k.variant == ParameterVariant.UNSUPPORTED
    assert model_config.parameters.top_k.value == NOT_GIVEN
    assert (
        model_config.parameters.presence_penalty.variant == ParameterVariant.UNSUPPORTED
    )
    assert model_config.parameters.presence_penalty.value == NOT_GIVEN

    # Test parameter validation still works - should fail for fixed parameters
    with pytest.raises(FixedParameterError):
        model_config.parameters.validate_parameter("temperature", 0.8)

    with pytest.raises(FixedParameterError):
        model_config.parameters.validate_parameter("top_p", 0.9)

    with pytest.raises(FixedParameterError):
        model_config.parameters.validate_parameter("frequency_penalty", 0.5)

    # Test parameter dumping excludes unsupported parameters
    dumped_params = model_config.parameters.model_dump()

    # Should include fixed parameters
    assert "temperature" in dumped_params
    assert dumped_params["temperature"] == 1.0
    assert "top_p" in dumped_params
    assert dumped_params["top_p"] == 1.0
    assert "frequency_penalty" in dumped_params
    assert dumped_params["frequency_penalty"] == 0.0

    # Should exclude unsupported parameters
    assert "top_k" not in dumped_params
    assert "presence_penalty" not in dumped_params

    # Should include the custom parameter from the config
    if hasattr(model_config.parameters, "reasoning_effort"):
        assert "reasoning_effort" in dumped_params
        assert dumped_params["reasoning_effort"] == "low"

    # Test conversion to ModelParameters works correctly
    model_params = ModelParameters(**dumped_params)
    assert model_params.temperature == 1.0
    assert model_params.top_p == 1.0
    assert model_params.frequency_penalty == 0.0
