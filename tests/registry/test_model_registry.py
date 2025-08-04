from datetime import datetime, timezone
from pathlib import Path

import pytest

from no_llm.models.config.enums import ModelCapability, ModelMode
from no_llm.models.config.metadata import (
    ModelMetadata,
    ModelPricing,
    PrivacyLevel,
    TokenPrices,
)
from no_llm.models.config.model import ModelConfiguration, ModelConstraints, ModelIdentity
from no_llm.models.config.parameters import ConfigurableModelParameters
from no_llm.models.config.properties import ModelProperties, QualityProperties, SpeedProperties
from no_llm.errors import (
    ConfigurationLoadError,
    ModelNotFoundError,
)
from no_llm.providers import OpenAIProvider, Provider
from no_llm.models.registry import ModelRegistry
from no_llm.models.registry import SetFilter
from no_llm._utils import find_yaml_file


def create_test_model(model_id: str = "test-model") -> ModelConfiguration:
    """Create a test model configuration"""
    return ModelConfiguration(
        identity=ModelIdentity(
            id=model_id,
            name="Test Model",
            version="1.0.0",
            description="Test model",
            creator="test",
        ),
        providers=[OpenAIProvider()],
        mode=ModelMode.CHAT,
        capabilities={ModelCapability.STREAMING},
        constraints=ModelConstraints(
            max_input_tokens=1000,
            max_output_tokens=500,
        ),
        parameters=ConfigurableModelParameters(),
        properties=ModelProperties(
            speed=SpeedProperties(score=50.0, label="test", description="test"),
            quality=QualityProperties(score=50.0, label="test", description="test"),
        ),
        metadata=ModelMetadata(
            privacy_level=[PrivacyLevel.BASIC],
            pricing=ModelPricing(
                token_prices=TokenPrices(
                    input_price_per_1k=0.01, output_price_per_1k=0.02
                )
            ),
            release_date=datetime.now(timezone.utc),
            data_cutoff_date=None,
        ),
    )


@pytest.fixture
def config_dir(tmp_path) -> Path:
    """Create a temporary config directory"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create provider directory
    providers_dir = config_dir / "providers"
    providers_dir.mkdir()

    # Create models directory
    models_dir = config_dir / "models"
    models_dir.mkdir()

    return config_dir

@pytest.fixture
def builtin_registry() -> ModelRegistry:
    """Create a test registry with basic setup"""
    registry = ModelRegistry()
    registry._models.clear()
    registry._builtin_models.clear()
    return registry


@pytest.fixture
def base_registry() -> ModelRegistry:
    """Create a test registry with basic setup"""
    registry = ModelRegistry()
    registry._models.clear()
    registry._builtin_models.clear()
    return registry
    # Register test provider type
    # registry.register_provider_type('test', MockProvider)
    # Create and register provider instance
    # provider = MockProvider()
    # registry.register_provider('test', provider)


@pytest.fixture
def registry(config_dir) -> ModelRegistry:
    """Create a registry with test configuration"""
    return ModelRegistry(config_dir)
    # registry.register_provider_type('test', MockProvider)


def test_registry_model_registration(base_registry: ModelRegistry):
    """Test model registration with providers"""
    # Create and register a test model
    model = create_test_model("model1")
    base_registry.register(model)

    # Test model was registered
    assert model.identity.id in base_registry._models

    # Test provider was set
    assert model.providers is not None
    assert isinstance(model.providers[0], OpenAIProvider)


def test_registry_model_listing(base_registry: ModelRegistry):
    """Test model listing and filtering"""
    # Clear any existing models
    base_registry._models.clear()

    # Register test models
    model1 = create_test_model("model1")
    model2 = create_test_model("model2")
    model2.capabilities.add(ModelCapability.VISION)

    base_registry.register(model1)
    base_registry.register(model2)

    # Test listing all models
    models = list(base_registry.list(only_valid=False, only_active=False))
    assert len(models) == 2

    # Test filtering by capability
    vision_models = list(
        base_registry.list(capabilities={ModelCapability.VISION}, only_valid=False, only_active=False)
    )
    assert len(vision_models) == 1
    assert vision_models[0].identity.id == "model2"

    # Test filtering by provider
    provider_models = list(base_registry.list(provider="openai", only_valid=False, only_active=False))
    assert len(provider_models) == 2


def test_registry_model_removal(base_registry: ModelRegistry):
    """Test model and provider removal"""
    # Register test model
    model = create_test_model()
    base_registry.register(model)

    # Test model removal
    base_registry.remove(model.identity.id)
    with pytest.raises(ModelNotFoundError):
        base_registry.get(model.identity.id)


def test_find_yaml_file(tmp_path: Path):
    """Test YAML file extension handling"""
    registry = ModelRegistry()
    base_path = tmp_path / "configs"
    base_path.mkdir()

    # Test .yml extension
    yml_file = base_path / "test.yml"
    yml_file.write_text("test: data")

    found = find_yaml_file(base_path, "test")
    assert found == yml_file

    # Test .yaml extension
    yaml_file = base_path / "other.yaml"
    yaml_file.write_text("test: data")

    found = find_yaml_file(base_path, "other")
    assert found == yaml_file

    # Test non-existent file returns default .yml
    not_found = find_yaml_file(base_path, "nonexistent")
    assert not_found == base_path / "nonexistent.yml"


def test_load_model_config_errors(tmp_path: Path):
    """Test model configuration loading errors"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    models_dir = config_dir / "models"
    models_dir.mkdir()

    registry = ModelRegistry(config_dir)

    # Test invalid YAML
    invalid_file = models_dir / "invalid.yml"
    invalid_file.write_text("invalid: : yaml")

    with pytest.raises(ConfigurationLoadError):
        registry._load_model_config("invalid")

    # Test invalid model configuration
    invalid_model = models_dir / "bad-model.yml"
    invalid_model.write_text("""
identity:
  id: bad-model
  # Missing required fields
""")

    with pytest.raises(ConfigurationLoadError):
        registry._load_model_config("bad-model")


def test_registry_model_privacy_filtering(base_registry: ModelRegistry):
    """Test filtering models by privacy level"""
    # Clear any existing models
    base_registry._models.clear()

    # Create models with different privacy levels
    model1 = create_test_model("model1")
    model1.metadata.privacy_level = [PrivacyLevel.BASIC]

    model2 = create_test_model("model2")
    model2.metadata.privacy_level = [PrivacyLevel.HIPAA]

    model3 = create_test_model("model3")
    model3.metadata.privacy_level = [PrivacyLevel.BASIC, PrivacyLevel.HIPAA]

    # Register all models
    base_registry.register(model1)
    base_registry.register(model2)
    base_registry.register(model3)

    # Test simple set filtering (defaults to "any" mode)
    basic_models = list(base_registry.list(privacy_levels={PrivacyLevel.BASIC}))
    assert len(basic_models) == 2  # model1 and model3
    assert "model1" in [m.identity.id for m in basic_models]
    assert "model3" in [m.identity.id for m in basic_models]

    # Test explicit SetFilter with "any" mode
    hipaa_models = list(
        base_registry.list(
            privacy_levels=SetFilter({PrivacyLevel.HIPAA}, mode="any")
        )
    )
    assert len(hipaa_models) == 2  # model2 and model3
    assert "model2" in [m.identity.id for m in hipaa_models]
    assert "model3" in [m.identity.id for m in hipaa_models]

    # Test SetFilter with "all" mode
    all_mode_models = list(
        base_registry.list(
            privacy_levels=SetFilter(
                {PrivacyLevel.BASIC, PrivacyLevel.HIPAA}, mode="all"
            )
        )
    )
    assert len(all_mode_models) == 1  # only model3 has both levels
    assert all_mode_models[0].identity.id == "model3"

    # Test combining with capabilities
    basic_streaming_models = list(
        base_registry.list(
            privacy_levels={PrivacyLevel.BASIC},  # Simple set usage
            capabilities=SetFilter(
                {ModelCapability.STREAMING}, mode="any"
            ),  # Explicit SetFilter usage
        )
    )
    assert len(basic_streaming_models) == 2  # model1 and model3


def test_builtin_models_registration(base_registry: ModelRegistry   ):
    """Test that all built-in model configurations are loaded correctly"""
    # Initialize registry without custom config dir
    registry = ModelRegistry(config_dir=None)
    assert len(list(registry.list())) > 0
    registry.reload()
    assert len(list(registry.list())) > 0


def test_custom_config_overrides_builtin(tmp_path: Path):
    """Test that custom YAML configurations can override built-in models"""
    # Create a config directory with models subdirectory
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    models_dir = config_dir / "models"
    models_dir.mkdir(exist_ok=True)

    base_registry = ModelRegistry(config_dir=None)
    builtin_model = base_registry.get("claude-3-haiku")

    # Create minimal override config
    custom_config = models_dir / "claude-3-haiku.yml"
    override_yaml = """
identity:
  id: claude-3-haiku
  description: Customized version of Claude 3 Haiku
"""
    custom_config.write_text(
        override_yaml.strip()
    )  # Remove leading/trailing whitespace

    assert custom_config.exists()

    custom_registry = ModelRegistry(config_dir)
    custom_model = custom_registry.get("claude-3-haiku")

    assert custom_model.identity.id == "claude-3-haiku"
    assert custom_model.identity.description == "Customized version of Claude 3 Haiku"
    assert custom_model.identity.description != builtin_model.identity.description
    assert custom_model.identity.name == builtin_model.identity.name
    assert custom_model.identity.version == builtin_model.identity.version
    assert custom_model.mode == builtin_model.mode
    assert custom_model.capabilities == builtin_model.capabilities


def test_registry_model_loading_errors(tmp_path: Path):
    """Test error handling during model configuration loading"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    models_dir = config_dir / "models"
    models_dir.mkdir()

    # Create invalid YAML file
    invalid_file = models_dir / "invalid.yml"
    invalid_file.write_text("invalid: yaml: content")

    registry = ModelRegistry(config_dir)

    # Should log error but not crash
    registry.reload()

    # Create file with invalid model configuration
    bad_config = models_dir / "bad_config.yml"
    bad_config.write_text("""
identity:
  id: bad-model
  # Missing required fields
""")

    # Should handle validation error gracefully
    registry.reload()

    # Test loading non-existent model
    with pytest.raises(ModelNotFoundError):
        registry.get("non-existent-model")


def test_registry_model_listing_filters():
    """Test model listing with various filter combinations"""
    # Create a registry without built-in models for cleaner testing
    registry = ModelRegistry()
    registry._models.clear()  # Clear built-in models for this test

    # Create test models
    model1 = create_test_model("model1")
    model1.capabilities = {ModelCapability.STREAMING}
    model1.metadata.privacy_level = [PrivacyLevel.BASIC]

    model2 = create_test_model("model2")
    model2.capabilities = {ModelCapability.JSON_MODE}
    model2.metadata.privacy_level = [PrivacyLevel.HIPAA]

    registry.register(model1)
    registry.register(model2)

    # Test filtering with empty filters
    models = list(registry.list())
    assert len(models) == 2

    # Test filtering with non-matching capability
    models = list(
        registry.list(
            capabilities=SetFilter({ModelCapability.FUNCTION_CALLING}, mode="any")
        )
    )
    assert len(models) == 0

    # Test filtering with non-matching privacy level
    models = list(
        registry.list(privacy_levels=SetFilter({PrivacyLevel.SOC2}, mode="any"))
    )
    assert len(models) == 0

    # Test filtering with matching filters
    models = list(
        registry.list(
            capabilities=SetFilter({ModelCapability.STREAMING}, mode="any"),
            privacy_levels=SetFilter({PrivacyLevel.BASIC}, mode="any"),
        )
    )
    assert len(models) == 1
    assert models[0].identity.id == "model1"


@pytest.mark.skip(
    reason="Skipping configuration reloading test, since ROI not enough for now"
)
def test_registry_configuration_reloading(tmp_path):
    """Test configuration reloading behavior"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    models_dir = config_dir / "models"
    models_dir.mkdir()

    # Create registry without built-in models
    registry = ModelRegistry(config_dir)

    # Add a model configuration
    model_file = models_dir / "test.yml"
    model_file.write_text("""
identity:
  id: test-model
  name: Test Model
  version: 1.0.0
  description: Test model
  creator: test
mode: chat
capabilities: [streaming]
constraints:
  max_input_tokens: 1000
  max_output_tokens: 500
metadata:
  privacy_level: [basic]
  pricing:
    token_prices:
      input_price_per_1k: 0.01
      output_price_per_1k: 0.02
  release_date: "2024-01-01T00:00:00Z"
""")

    # Test explicit reload
    registry.reload()
    assert len(list(registry.list())) == 2

    # Get the custom model
    custom_model = registry.get("test-model")
    assert custom_model.identity.name == "Test Model"

    # Modify the model
    model_file.write_text("""
identity:
  id: test-model
  name: Updated Model
  version: 1.0.0
  description: Test model
  creator: test
mode: chat
capabilities: [streaming]
constraints:
  max_input_tokens: 1000
  max_output_tokens: 500
metadata:
  privacy_level: [basic]
  pricing:
    token_prices:
      input_price_per_1k: 0.01
      output_price_per_1k: 0.02
  release_date: "2024-01-01T00:00:00Z"
""")

    # Test reload with changes
    registry.reload()
    custom_model = registry.get("test-model")
    assert custom_model.identity.name == "Updated Model"
    assert len(list(registry.list())) == 2


def test_model_set_active_functionality():
    """Test setting model active status"""
    registry = ModelRegistry()
    model = create_test_model("test-model")
    registry.register(model)
    
    # Model should be active by default
    assert model.is_active is True
    
    # Set model to inactive
    registry.set_active("test-model", False)
    retrieved_model = registry.get("test-model")
    assert retrieved_model.is_active is False
    
    # Set model back to active
    registry.set_active("test-model", True)
    retrieved_model = registry.get("test-model")
    assert retrieved_model.is_active is True


def test_model_set_active_nonexistent_model():
    """Test setting active status on non-existent model raises error"""
    registry = ModelRegistry()
    
    with pytest.raises(ModelNotFoundError) as exc_info:
        registry.set_active("nonexistent", True)
    
    assert exc_info.value.model_id == "nonexistent"


def test_model_listing_with_active_filter(base_registry: ModelRegistry):
    """Test model listing respects is_active filter"""
    registry = base_registry
    
    # Add multiple models
    model1 = create_test_model("model1")
    model2 = create_test_model("model2")
    registry.register(model1)
    registry.register(model2)
    
    # Deactivate one model
    registry.set_active("model1", False)
    
    # Check filtering works
    active_models = list(registry.list(only_valid=False, only_active=True))
    all_models = list(registry.list(only_valid=False, only_active=False))
    
    assert len(active_models) == 1
    assert len(all_models) == 2  # Total unchanged
    
    # Verify the deactivated model is not in active list
    active_model_ids = {m.identity.id for m in active_models}
    assert "model1" not in active_model_ids
    
    # But it should be in the all models list
    all_model_ids = {m.identity.id for m in all_models}
    assert "model1" in all_model_ids


def test_model_combined_active_and_valid_filtering():
    """Test combined filtering by both is_active and is_valid"""
    registry = ModelRegistry()
    
    # Add a test model
    model = create_test_model("test-model")
    registry.register(model)
    
    # Test all combinations
    models_active_valid = list(registry.list(only_valid=True, only_active=True))
    models_active_any = list(registry.list(only_valid=False, only_active=True))
    models_any_valid = list(registry.list(only_valid=True, only_active=False))
    models_any_any = list(registry.list(only_valid=False, only_active=False))
    
    # Logical relationships should hold
    assert len(models_active_valid) <= len(models_active_any)
    assert len(models_active_valid) <= len(models_any_valid)
    assert len(models_active_any) <= len(models_any_any)
    assert len(models_any_valid) <= len(models_any_any)


def test_model_filtering_with_other_criteria():
    """Test that is_active and is_valid work with existing filtering criteria"""
    registry = ModelRegistry()
    registry._models.clear()
    
    # Create models with different capabilities
    model1 = create_test_model("model1")
    model1.capabilities = {ModelCapability.STREAMING}
    
    model2 = create_test_model("model2")  
    model2.capabilities = {ModelCapability.STRUCTURED_OUTPUT}
    
    registry.register(model1)
    registry.register(model2)
    
    # Test filtering by capabilities with active filter
    streaming_models = list(registry.list(
        capabilities={ModelCapability.STREAMING},
        only_active=True,
        only_valid=False
    ))
    assert len(streaming_models) == 1
    assert streaming_models[0].identity.id == "model1"
    
    # Deactivate model1 and test again
    registry.set_active("model1", False)
    
    streaming_models_active = list(registry.list(
        capabilities={ModelCapability.STREAMING},
        only_active=True,
        only_valid=False
    ))
    assert len(streaming_models_active) == 0
    
    streaming_models_all = list(registry.list(
        capabilities={ModelCapability.STREAMING},
        only_active=False,
        only_valid=False
    ))
    assert len(streaming_models_all) == 1
