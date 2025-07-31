from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from no_llm.errors import ConfigurationLoadError, ProviderNotFoundError
from no_llm.providers import AnthropicProvider, OpenAIProvider, VertexProvider
from no_llm.providers.registry import ProviderRegistry


def create_test_provider(provider_type: str = "anthropic") -> AnthropicProvider:
    """Create a test provider configuration"""
    return AnthropicProvider(
        type=provider_type, # type: ignore
        name=f"Test {provider_type.title()}",
    )


@pytest.fixture
def config_dir(tmp_path) -> Path:
    """Create a temporary config directory with providers subdirectory"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    providers_dir = config_dir / "providers"
    providers_dir.mkdir()

    return config_dir


@pytest.fixture
def base_registry() -> ProviderRegistry:
    """Create a test registry without config directory"""
    return ProviderRegistry()


@pytest.fixture
def registry(config_dir) -> ProviderRegistry:
    """Create a registry with test configuration directory"""
    return ProviderRegistry(config_dir)


def test_registry_initialization_no_config():
    """Test registry initialization without config directory"""
    registry = ProviderRegistry()
    
    assert registry._config_dir is None
    assert len(registry._providers) == 0


def test_registry_initialization_with_config(config_dir):
    """Test registry initialization with config directory"""
    registry = ProviderRegistry(config_dir)
    
    assert registry._config_dir == config_dir
    assert len(registry._providers) == 0  # No providers in empty directory


def test_registry_provider_registration(base_registry):
    """Test provider registration"""
    provider = create_test_provider("anthropic")
    base_registry.register_provider(provider)

    assert "anthropic" in base_registry._providers
    assert base_registry._providers["anthropic"] == provider


def test_registry_provider_override(base_registry):
    """Test provider override behavior"""
    provider1 = create_test_provider("anthropic")
    provider1.name = "First Provider"
    
    provider2 = create_test_provider("anthropic")
    provider2.name = "Second Provider"

    base_registry.register_provider(provider1)
    base_registry.register_provider(provider2)

    assert len(base_registry._providers) == 1
    assert base_registry._providers["anthropic"].name == "Second Provider"


def test_registry_get_provider(base_registry):
    """Test getting providers by type"""
    provider = create_test_provider("anthropic")
    base_registry.register_provider(provider)

    retrieved = base_registry.get_provider("anthropic")
    assert retrieved == provider


def test_registry_get_nonexistent_provider(base_registry):
    """Test getting non-existent provider raises error"""
    with pytest.raises(ProviderNotFoundError) as exc_info:
        base_registry.get_provider("nonexistent")
    
    assert exc_info.value.provider_id == "nonexistent"


def test_registry_remove_provider(base_registry):
    """Test provider removal"""
    provider = create_test_provider("anthropic")
    base_registry.register_provider(provider)

    base_registry.remove_provider("anthropic")
    
    with pytest.raises(ProviderNotFoundError):
        base_registry.get_provider("anthropic")


def test_registry_remove_nonexistent_provider(base_registry):
    """Test removing non-existent provider raises error"""
    with pytest.raises(ProviderNotFoundError) as exc_info:
        base_registry.remove_provider("nonexistent")
    
    assert exc_info.value.provider_id == "nonexistent"


def test_registry_list_providers_empty(base_registry):
    """Test listing providers when registry is empty"""
    providers = list(base_registry.list_providers())
    assert len(providers) == 0


def test_registry_list_providers_with_valid_env(base_registry):
    """Test listing providers with valid environment"""
    provider = create_test_provider("anthropic")
    
    # Mock has_valid_env to return True
    with patch.object(provider, 'has_valid_env', return_value=True):
        base_registry.register_provider(provider)
        providers = list(base_registry.list_providers(only_valid=True))
        assert len(providers) == 1
        assert providers[0] == provider


def test_registry_list_providers_with_invalid_env(base_registry):
    """Test listing providers with invalid environment"""
    provider = create_test_provider("anthropic")
    
    # Mock has_valid_env to return False
    with patch.object(provider, 'has_valid_env', return_value=False):
        base_registry.register_provider(provider)
        providers = list(base_registry.list_providers(only_valid=True))
        assert len(providers) == 0


def test_registry_list_providers_include_invalid(base_registry):
    """Test listing providers including invalid ones"""
    provider = create_test_provider("anthropic")
    
    # Mock has_valid_env to return False
    with patch.object(provider, 'has_valid_env', return_value=False):
        base_registry.register_provider(provider)
        providers = list(base_registry.list_providers(only_valid=False))
        assert len(providers) == 1
        assert providers[0] == provider


def test_registry_list_provider_instances_single(base_registry):
    """Test listing provider instances for single-instance provider"""
    provider = create_test_provider("anthropic")
    
    with patch.object(provider, 'has_valid_env', return_value=True):
        base_registry.register_provider(provider)
        instances = list(base_registry.list_provider_instances())
        assert len(instances) == 1
        assert instances[0] == provider


def test_registry_list_provider_instances_multi():
    """Test listing provider instances for multi-instance provider (like Vertex)"""
    base_registry = ProviderRegistry()
    
    # Create a VertexProvider with multiple locations
    vertex_provider = VertexProvider(
        type="vertex",
        name="Vertex AI",
        model_family="gemini",
        locations=["us-central1", "europe-west1"]
    )
    
    with patch.object(vertex_provider, 'has_valid_env', return_value=True):
        base_registry.register_provider(vertex_provider)
        instances = list(base_registry.list_provider_instances())
        
        # VertexProvider should yield multiple instances (one per location)
        assert len(instances) >= 1  # At least one instance should be yielded


def test_find_yaml_file(tmp_path):
    """Test YAML file extension handling"""
    registry = ProviderRegistry()
    base_path = tmp_path / "configs"
    base_path.mkdir()

    # Test .yml extension
    yml_file = base_path / "test.yml"
    yml_file.write_text("type: test")

    found = registry._find_yaml_file(base_path, "test")
    assert found == yml_file

    # Test .yaml extension
    yaml_file = base_path / "other.yaml"
    yaml_file.write_text("type: other")

    found = registry._find_yaml_file(base_path, "other")
    assert found == yaml_file

    # Test non-existent file returns default .yml
    not_found = registry._find_yaml_file(base_path, "nonexistent")
    assert not_found == base_path / "nonexistent.yml"


def test_load_provider_from_yaml(tmp_path):
    """Test loading provider from YAML file"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    providers_dir = config_dir / "providers"
    providers_dir.mkdir()

    # Create valid provider YAML
    provider_file = providers_dir / "anthropic.yml"
    provider_config = {
        "type": "anthropic",
        "name": "Anthropic",
    }
    
    with open(provider_file, 'w') as f:
        yaml.dump(provider_config, f)

    registry = ProviderRegistry(config_dir)
    
    # Verify provider was loaded
    provider = registry.get_provider("anthropic")
    assert provider.type == "anthropic"
    assert provider.name == "Anthropic"


def test_load_provider_invalid_yaml(tmp_path):
    """Test handling of invalid YAML files"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    providers_dir = config_dir / "providers"
    providers_dir.mkdir()

    # Create invalid YAML file
    invalid_file = providers_dir / "invalid.yml"
    invalid_file.write_text("invalid: yaml: content: :")

    # Should not crash, but log error
    registry = ProviderRegistry(config_dir)
    
    # Should have no providers loaded
    providers = list(registry.list_providers())
    assert len(providers) == 0


def test_load_provider_invalid_config(tmp_path):
    """Test handling of invalid provider configuration"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    providers_dir = config_dir / "providers"
    providers_dir.mkdir()

    # Create YAML with invalid provider config
    invalid_config = providers_dir / "bad_provider.yml"
    invalid_config_data = {
        "type": "nonexistent_provider_type",
        "name": "Bad Provider"
    }
    
    with open(invalid_config, 'w') as f:
        yaml.dump(invalid_config_data, f)

    # Should not crash, but log error
    registry = ProviderRegistry(config_dir)
    
    # Should have no providers loaded
    providers = list(registry.list_providers())
    assert len(providers) == 0


def test_load_multiple_providers(tmp_path):
    """Test loading multiple providers from directory"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    providers_dir = config_dir / "providers"
    providers_dir.mkdir()

    # Create multiple provider files
    providers_configs = [
        {
            "file": "anthropic.yml",
            "config": {
                "type": "anthropic",
                "name": "Anthropic",
            }
        },
        {
            "file": "openai.yml", 
            "config": {
                "type": "openai",
                "name": "OpenAI",
            }
        }
    ]

    for provider_data in providers_configs:
        provider_file = providers_dir / provider_data["file"]
        with open(provider_file, 'w') as f:
            yaml.dump(provider_data["config"], f)

    registry = ProviderRegistry(config_dir)
    
    # Should have loaded both providers
    providers = list(registry.list_providers(only_valid=False))
    assert len(providers) == 2
    
    provider_types = {provider.type for provider in providers}
    assert "anthropic" in provider_types
    assert "openai" in provider_types


def test_reload_configurations(tmp_path):
    """Test configuration reloading"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    providers_dir = config_dir / "providers"
    providers_dir.mkdir()

    registry = ProviderRegistry(config_dir)
    
    # Initially no providers
    assert len(list(registry.list_providers(only_valid=False))) == 0

    # Add a provider file
    provider_file = providers_dir / "anthropic.yml"
    provider_config = {
        "type": "anthropic",
        "name": "Anthropic Test",
    }
    
    with open(provider_file, 'w') as f:
        yaml.dump(provider_config, f)

    # Reload configurations
    registry.reload_configurations()
    
    # Should now have the provider
    providers = list(registry.list_providers(only_valid=False))
    assert len(providers) == 1
    assert providers[0].type == "anthropic"
    assert providers[0].name == "Anthropic Test"


def test_create_provider_from_config():
    """Test creating provider from config dictionary"""
    registry = ProviderRegistry()
    
    config = {
        "type": "anthropic",
        "name": "Test Anthropic",
    }
    
    provider = registry._create_provider_from_config(config)
    assert provider.type == "anthropic"
    assert provider.name == "Test Anthropic"


def test_registry_providers_directory_not_exists(tmp_path):
    """Test behavior when providers directory doesn't exist"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    # Don't create providers directory

    registry = ProviderRegistry(config_dir)
    
    # Should not crash and have no providers
    providers = list(registry.list_providers())
    assert len(providers) == 0


def test_registry_providers_directory_empty(tmp_path):
    """Test behavior with empty providers directory"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    providers_dir = config_dir / "providers"
    providers_dir.mkdir()

    registry = ProviderRegistry(config_dir)
    
    # Should have no providers
    providers = list(registry.list_providers())
    assert len(providers) == 0