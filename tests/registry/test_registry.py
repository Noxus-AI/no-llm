from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from no_llm.registry import Registry
from no_llm.models.config import ModelConfiguration
from no_llm.providers.config import ProviderConfiguration


class TestRegistry:
    @pytest.fixture
    def registry(self):
        return Registry()

    def test_init(self, registry):
        assert registry.models is not None
        assert registry.providers is not None

    def test_init_with_config_dir(self, tmp_path):
        registry = Registry(config_dir=tmp_path)
        assert registry.models is not None
        assert registry.providers is not None

    def test_models_list_method_exists(self, registry):
        assert hasattr(registry.models, 'list')
        assert callable(registry.models.list)

    def test_providers_list_method_exists(self, registry):
        assert hasattr(registry.providers, 'list')
        assert callable(registry.providers.list)

    def test_providers_list_by_type_method_exists(self, registry):
        assert hasattr(registry.providers, 'list_by_type')
        assert callable(registry.providers.list_by_type)

    @patch('no_llm.models.registry.ModelRegistry.get')
    @patch('no_llm.providers.registry.ProviderRegistry.list_by_type')
    def test_get_compatible_providers(self, mock_list_by_type, mock_get_model, registry: Registry):
        # Mock model with compatible providers
        mock_provider_class = Mock()
        mock_provider_class.return_value.type = "anthropic"
        
        mock_model = Mock(spec=ModelConfiguration)
        mock_model._compatible_providers = {mock_provider_class}
        mock_model.identity = Mock()
        mock_model.identity.id = "test-model"
        mock_get_model.return_value = mock_model

        # Mock provider response
        mock_provider = Mock(spec=ProviderConfiguration)
        mock_list_by_type.return_value = [mock_provider]

        # Test
        result = list(registry.get_compatible_providers("test-model"))
        
        mock_get_model.assert_called_once_with("test-model")
        mock_list_by_type.assert_called_once_with("anthropic", only_valid=True, only_active=True)
        assert result == [mock_provider]

    @patch('no_llm.providers.registry.ProviderRegistry.get')
    @patch('no_llm.models.registry.ModelRegistry.list')
    def test_get_models_for_provider(self, mock_list_models, mock_get_provider, registry: Registry):
        # Mock provider
        mock_provider = Mock(spec=ProviderConfiguration)
        mock_provider.type = "anthropic"
        mock_get_provider.return_value = mock_provider

        # Mock models with compatible provider types
        mock_provider_class = type(mock_provider)
        
        compatible_model = Mock(spec=ModelConfiguration)
        compatible_model._compatible_providers = {mock_provider_class}
        compatible_model.identity = Mock()
        compatible_model.identity.id = "compatible-model"
        
        incompatible_model = Mock(spec=ModelConfiguration)
        incompatible_model._compatible_providers = {Mock}  # Different provider type
        incompatible_model.identity = Mock()
        incompatible_model.identity.id = "incompatible-model"
        
        mock_list_models.return_value = [compatible_model, incompatible_model]

        # Test
        result = list(registry.get_models_for_provider("test-provider"))
        
        mock_get_provider.assert_called_once_with("test-provider")
        mock_list_models.assert_called_once_with(only_valid=True, only_active=True)
        assert result == [compatible_model]

    @patch('no_llm.models.registry.ModelRegistry.reload')
    @patch('no_llm.providers.registry.ProviderRegistry.reload')
    def test_reload_all(self, mock_providers_reload, mock_models_reload, registry: Registry):
        registry.reload()
        
        mock_models_reload.assert_called_once()
        mock_providers_reload.assert_called_once()

    def test_get_compatible_providers_with_filters(self, registry: Registry):
        with patch.object(registry.models, 'get') as mock_get_model, \
             patch.object(registry.providers, 'list_by_type') as mock_list_by_type:
            
            mock_provider_class = Mock()
            mock_provider_class.return_value.type = "anthropic"
            
            mock_model = Mock(spec=ModelConfiguration)
            mock_model._compatible_providers = {mock_provider_class}
            mock_model.identity = Mock()
            mock_model.identity.id = "test-model"
            mock_get_model.return_value = mock_model
            
            mock_list_by_type.return_value = []
            
            list(registry.get_compatible_providers("test-model", only_valid=False, only_active=False))
            
            mock_list_by_type.assert_called_once_with("anthropic", only_valid=False, only_active=False)

    def test_get_models_for_provider_with_filters(self, registry: Registry):
        with patch.object(registry.providers, 'get') as mock_get_provider, \
             patch.object(registry.models, 'list') as mock_list_models:
            
            mock_provider = Mock(spec=ProviderConfiguration)
            mock_provider.type = "anthropic"
            mock_get_provider.return_value = mock_provider
            
            mock_list_models.return_value = []
            
            list(registry.get_models_for_provider("test-provider", only_valid=False, only_active=False))
            
            mock_list_models.assert_called_once_with(only_valid=False, only_active=False)

    def test_multiple_provider_types_compatibility(self, registry: Registry):
        with patch.object(registry.models, 'get') as mock_get_model, \
             patch.object(registry.providers, 'list_by_type') as mock_list_by_type:
            
            # Model supports multiple provider types
            mock_provider_class1 = Mock()
            mock_provider_class1.return_value.type = "anthropic"
            mock_provider_class2 = Mock()
            mock_provider_class2.return_value.type = "openai"
            
            mock_model = Mock(spec=ModelConfiguration)
            mock_model._compatible_providers = {mock_provider_class1, mock_provider_class2}
            mock_model.identity = Mock()
            mock_model.identity.id = "multi-provider-model"
            mock_get_model.return_value = mock_model
            
            mock_provider1 = Mock(spec=ProviderConfiguration)
            mock_provider2 = Mock(spec=ProviderConfiguration)
            
            def side_effect(provider_type, **kwargs):
                if provider_type == "anthropic":
                    return [mock_provider1]
                elif provider_type == "openai":
                    return [mock_provider2]
                return []
            
            mock_list_by_type.side_effect = side_effect
            
            result = list(registry.get_compatible_providers("test-model"))
            
            assert len(result) == 2
            assert mock_provider1 in result
            assert mock_provider2 in result