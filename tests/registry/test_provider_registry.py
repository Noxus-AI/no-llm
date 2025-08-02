from pathlib import Path

import pytest
import yaml

from no_llm.errors import ProviderNotFoundError
from no_llm.providers.registry import ProviderRegistry
from no_llm._utils import find_yaml_file


def create_test_provider_config(provider_type: str = "anthropic", provider_id: str | None = None) -> dict:
    """Create a test provider configuration dict"""
    return {
        "type": provider_type,
        "id": provider_id or provider_type,
        "name": f"Test {provider_type.title()}",
    }


@pytest.fixture
def config_dir(tmp_path) -> Path:
    """Create a temporary config directory with providers subdirectory"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    providers_dir = config_dir / "providers"
    providers_dir.mkdir()

    return config_dir


@pytest.fixture(scope="function")
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
    assert len(registry._providers) > 0  # Should have builtin providers


def test_registry_initialization_with_config(config_dir):
    """Test registry initialization with config directory"""
    registry = ProviderRegistry(config_dir)
    
    assert registry._config_dir == config_dir
    assert len(registry._providers) > 0  # Should have builtin providers


def test_registry_provider_registration(base_registry: ProviderRegistry):
    """Test provider registration"""
    config = create_test_provider_config("anthropic", "test-anthropic")
    provider = base_registry._create_provider_from_config(config)
    base_registry.register(provider)

    assert "test-anthropic" in base_registry._providers
    assert base_registry._providers["test-anthropic"] == provider


def test_registry_provider_override(base_registry: ProviderRegistry):
    """Test provider override behavior"""
    initial_count = len(base_registry._providers)
    
    config1 = create_test_provider_config("anthropic", "test-provider")
    config1["name"] = "First Provider"
    provider1 = base_registry._create_provider_from_config(config1)
    
    config2 = create_test_provider_config("anthropic", "test-provider")  
    config2["name"] = "Second Provider"
    provider2 = base_registry._create_provider_from_config(config2)

    base_registry.register(provider1)
    base_registry.register(provider2)

    # Should have same count since we're overriding
    assert len(base_registry._providers) == initial_count + 1
    assert base_registry._providers["test-provider"].name == "Second Provider"


def test_registry_get_provider(base_registry: ProviderRegistry):
    """Test getting providers by ID"""
    config = create_test_provider_config("anthropic", "test-anthropic")
    provider = base_registry._create_provider_from_config(config)
    base_registry.register(provider)

    retrieved = base_registry.get("test-anthropic")
    assert retrieved == provider


def test_registry_get_providers_by_type_multiple(base_registry: ProviderRegistry):
    """Test getting multiple providers of same type"""
    # Start with builtin anthropic and openai
    initial_anthropic = len(list(base_registry.list_by_type("anthropic", only_valid=False, only_active=False)))
    initial_openai = len(list(base_registry.list_by_type("openai", only_valid=False, only_active=False)))
    
    config1 = create_test_provider_config("anthropic", "anthropic-1")
    config2 = create_test_provider_config("anthropic", "anthropic-2")
    config3 = create_test_provider_config("openai", "openai-1")
    
    provider1 = base_registry._create_provider_from_config(config1)
    provider2 = base_registry._create_provider_from_config(config2)
    provider3 = base_registry._create_provider_from_config(config3)
    
    base_registry.register(provider1)
    base_registry.register(provider2)
    base_registry.register(provider3)

    anthropic_providers = list(base_registry.list_by_type("anthropic", only_valid=False, only_active=False))
    assert len(anthropic_providers) == initial_anthropic + 2
    
    openai_providers = list(base_registry.list_by_type("openai", only_valid=False, only_active=False))
    assert len(openai_providers) == initial_openai + 1


def test_registry_get_providers_by_type_empty(base_registry: ProviderRegistry):
    """Test getting providers by type when none exist"""
    providers = list(base_registry.list_by_type("nonexistent"))
    assert len(providers) == 0


def test_registry_get_nonexistent_provider(base_registry: ProviderRegistry):
    """Test getting non-existent provider raises error"""
    with pytest.raises(ProviderNotFoundError) as exc_info:
        base_registry.get("nonexistent")
    
    assert exc_info.value.provider_id == "nonexistent"


def test_registry_remove_provider(base_registry: ProviderRegistry):
    """Test provider removal"""
    config = create_test_provider_config("anthropic", "test-anthropic")
    provider = base_registry._create_provider_from_config(config)
    base_registry.register(provider)

    base_registry.remove("test-anthropic")
    
    with pytest.raises(ProviderNotFoundError):
        base_registry.get("test-anthropic")


def test_registry_remove_nonexistent_provider(base_registry: ProviderRegistry):
    """Test removing non-existent provider raises error"""
    with pytest.raises(ProviderNotFoundError) as exc_info:
        base_registry.remove("nonexistent")
    
    assert exc_info.value.provider_id == "nonexistent"


def test_registry_list_providers_with_builtins_only(base_registry: ProviderRegistry):
    """Test listing providers when registry only has builtins"""
    providers = list(base_registry.list(only_valid=False))
    assert len(providers) >= 13  # Should have all builtin providers
    
    # Check that we have expected builtin types (all from AnyProvider union)
    provider_types = {p.type for p in providers}
    expected_types = {"anthropic", "openai", "vertex", "groq", "mistral", "azure", "perplexity", "deepseek", "together", "openrouter", "grok", "fireworks", "bedrock"}
    assert expected_types.issubset(provider_types)


def test_registry_list_providers_all(base_registry: ProviderRegistry):
    """Test listing all providers regardless of environment validity"""
    initial_count = len(list(base_registry.list(only_valid=False)))
    
    config = create_test_provider_config("anthropic", "test-anthropic")
    provider = base_registry._create_provider_from_config(config)
    base_registry.register(provider)
    
    providers = list(base_registry.list(only_valid=False))
    assert len(providers) == initial_count + 1
    assert provider in providers


def test_find_yaml_file(tmp_path):
    """Test YAML file extension handling"""
    registry = ProviderRegistry()
    base_path = tmp_path / "configs"
    base_path.mkdir()

    yml_file = base_path / "test.yml"
    yml_file.write_text("type: test")

    found = find_yaml_file(base_path, "test")
    assert found == yml_file

    yaml_file = base_path / "other.yaml"
    yaml_file.write_text("type: other")

    found = find_yaml_file(base_path, "other")
    assert found == yaml_file

    not_found = find_yaml_file(base_path, "nonexistent")
    assert not_found == base_path / "nonexistent.yml"


def test_load_provider_from_yaml(tmp_path):
    """Test loading provider from YAML file"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    providers_dir = config_dir / "providers"
    providers_dir.mkdir()

    provider_file = providers_dir / "anthropic.yml"
    provider_config = {
        "type": "anthropic",
        "id": "anthropic",
        "name": "Anthropic",
    }
    
    with open(provider_file, 'w') as f:
        yaml.dump(provider_config, f)

    registry = ProviderRegistry(config_dir)
    
    provider = registry.get("anthropic")
    assert provider.type == "anthropic"
    assert provider.name == "Anthropic"


def test_load_provider_invalid_yaml(tmp_path):
    """Test handling of invalid YAML files"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    providers_dir = config_dir / "providers"
    providers_dir.mkdir()

    invalid_file = providers_dir / "invalid.yml"
    invalid_file.write_text("invalid: yaml: content: :")

    registry = ProviderRegistry(config_dir)
    
    providers = list(registry.list())
    assert len(providers) > 0  # Should have builtin providers


def test_load_provider_invalid_config(tmp_path):
    """Test handling of invalid provider configuration"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    providers_dir = config_dir / "providers"
    providers_dir.mkdir()

    invalid_config = providers_dir / "bad_provider.yml"
    invalid_config_data = {
        "type": "nonexistent_provider_type",
        "name": "Bad Provider"
    }
    
    with open(invalid_config, 'w') as f:
        yaml.dump(invalid_config_data, f)

    registry = ProviderRegistry(config_dir)
    
    providers = list(registry.list())
    assert len(providers) > 0  # Should have builtin providers


def test_load_multiple_providers(tmp_path):
    """Test loading multiple providers from directory"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    providers_dir = config_dir / "providers"
    providers_dir.mkdir()

    providers_configs = [
        {
            "file": "anthropic.yml",
            "config": {
                "type": "anthropic",
                "id": "anthropic",
                "name": "Anthropic",
            }
        },
        {
            "file": "openai.yml", 
            "config": {
                "type": "openai",
                "id": "openai",
                "name": "OpenAI",
            }
        }
    ]

    for provider_data in providers_configs:
        provider_file = providers_dir / provider_data["file"]
        with open(provider_file, 'w') as f:
            yaml.dump(provider_data["config"], f)

    registry = ProviderRegistry(config_dir)
    
    providers = list(registry.list(only_valid=False))
    assert len(providers) >= 13  # Should have all builtin providers
    
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
    
    assert len(list(registry.list(only_valid=True))) == 0

    provider_file = providers_dir / "anthropic.yml"
    provider_config = {
        "type": "anthropic",
        "id": "anthropic",
        "name": "Anthropic Test",
    }
    
    with open(provider_file, 'w') as f:
        yaml.dump(provider_config, f)

    registry.reload()
    
    providers = list(registry.list(only_valid=False))
    assert len(providers) >= 13  # Should have all builtin providers
    
    # Find the anthropic provider and verify it was overridden
    anthropic_provider = registry.get("anthropic")
    assert anthropic_provider.type == "anthropic"
    assert anthropic_provider.name == "Anthropic Test"


def test_create_provider_from_config():
    """Test creating provider from config dictionary"""
    registry = ProviderRegistry()
    
    config = {
        "type": "anthropic",
        "id": "test-anthropic",
        "name": "Test Anthropic",
    }
    
    provider = registry._create_provider_from_config(config)
    assert provider.type == "anthropic"
    assert provider.name == "Test Anthropic"


def test_registry_providers_directory_not_exists(tmp_path):
    """Test behavior when providers directory doesn't exist"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    registry = ProviderRegistry(config_dir)
    
    providers = list(registry.list())
    assert len(providers) > 0  # Should have builtin providers


def test_registry_providers_directory_empty(tmp_path):
    """Test behavior with empty providers directory"""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    providers_dir = config_dir / "providers"
    providers_dir.mkdir()

    registry = ProviderRegistry(config_dir)
    
    providers = list(registry.list(only_valid=False))
    assert len(providers) >= 13  # Should have all builtin providers


def test_registry_comprehensive_builtin_behavior():
    """Test the complete builtin provider behavior"""
    registry = ProviderRegistry()
    
    # Should have all providers from AnyProvider union
    all_providers = list(registry.list(only_valid=False))
    provider_types = {p.type for p in all_providers}
    
    # All expected types from AnyProvider should be present
    expected_types = {
        "anthropic", "openai", "vertex", "groq", "mistral", 
        "azure", "perplexity", "deepseek", "together", 
        "openrouter", "grok", "fireworks", "bedrock"
    }
    assert expected_types.issubset(provider_types)
    
    # Each type should have exactly one builtin provider
    for provider_type in expected_types:
        providers_of_type = list(registry.list_by_type(provider_type, only_valid=False, only_active=False))
        assert len(providers_of_type) == 1
        assert providers_of_type[0].type == provider_type
        
    # Verify we can access all by ID (same as type for builtins)
    for provider_type in expected_types:
        provider = registry.get(provider_type)
        assert provider.type == provider_type


def test_provider_set_active_functionality(base_registry: ProviderRegistry):
    """Test setting provider active status"""
    provider_id = "anthropic"
    
    # Provider should be active by default
    provider = base_registry.get(provider_id)
    assert provider.is_active is True
    
    # Set provider to inactive
    base_registry.set_active(provider_id, False)
    provider = base_registry.get(provider_id)
    assert provider.is_active is False
    
    # Set provider back to active
    base_registry.set_active(provider_id, True)
    provider = base_registry.get(provider_id)
    assert provider.is_active is True


def test_provider_set_active_nonexistent_provider(base_registry: ProviderRegistry):
    """Test setting active status on non-existent provider raises error"""
    with pytest.raises(ProviderNotFoundError) as exc_info:
        base_registry.set_active("nonexistent", True)
    
    assert exc_info.value.provider_id == "nonexistent"


def test_provider_listing_with_active_filter(base_registry: ProviderRegistry):
    """Test provider listing respects is_active filter"""
    # Count active providers initially
    active_providers = list(base_registry.list(only_valid=False, only_active=True))
    all_providers = list(base_registry.list(only_valid=False, only_active=False))
    initial_active_count = len(active_providers)
    initial_total_count = len(all_providers)
    
    # Deactivate one provider
    provider_id = "anthropic"
    base_registry.set_active(provider_id, False)
    
    # Check filtering works
    active_providers = list(base_registry.list(only_valid=False, only_active=True))
    all_providers = list(base_registry.list(only_valid=False, only_active=False))
    
    assert len(active_providers) == initial_active_count - 1
    assert len(all_providers) == initial_total_count  # Total unchanged
    
    # Verify the deactivated provider is not in active list
    active_provider_ids = {p.id for p in active_providers}
    assert provider_id not in active_provider_ids
    
    # But it should be in the all providers list
    all_provider_ids = {p.id for p in all_providers}
    assert provider_id in all_provider_ids


def test_provider_get_by_type_with_active_filter(base_registry: ProviderRegistry):
    """Test get_providers_by_type respects is_active filter"""
    provider_type = "anthropic"
    
    # Should find active anthropic providers
    active_providers = list(base_registry.list_by_type(
        provider_type, only_valid=False, only_active=True
    ))
    all_providers = list(base_registry.list_by_type(
        provider_type, only_valid=False, only_active=False
    ))
    
    initial_active_count = len(active_providers)
    initial_total_count = len(all_providers)
    
    # Deactivate the anthropic provider
    base_registry.set_active("anthropic", False)
    
    # Check filtering works
    active_providers = list(base_registry.list_by_type(
        provider_type, only_valid=False, only_active=True
    ))
    all_providers = list(base_registry.list_by_type(
        provider_type, only_valid=False, only_active=False
    ))
    
    assert len(active_providers) == initial_active_count - 1
    assert len(all_providers) == initial_total_count



def test_provider_combined_active_and_valid_filtering(base_registry: ProviderRegistry):
    """Test combined filtering by both is_active and is_valid"""
    # Test all combinations
    providers_active_valid = list(base_registry.list(only_valid=True, only_active=True))
    providers_active_any = list(base_registry.list(only_valid=False, only_active=True))
    providers_any_valid = list(base_registry.list(only_valid=True, only_active=False))
    providers_any_any = list(base_registry.list(only_valid=False, only_active=False))
    
    # Logical relationships should hold
    assert len(providers_active_valid) <= len(providers_active_any)
    assert len(providers_active_valid) <= len(providers_any_valid)
    assert len(providers_active_any) <= len(providers_any_any)
    assert len(providers_any_valid) <= len(providers_any_any)