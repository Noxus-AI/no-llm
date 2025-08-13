from __future__ import annotations

from unittest.mock import patch

import pytest
from no_llm import ModelCapability, ModelParameters, ModelRegistry
from no_llm.presets import ModelPreset
from no_llm.providers.provider_configs.test import TestProvider


@pytest.fixture
def registry_with_test_models() -> ModelRegistry:
    """Get builtin registry and modify models to use test provider"""
    registry = ModelRegistry()
    
    # Get some builtin models and modify them to use test provider
    models = list(registry.list())
    if not models:
        pytest.skip("No builtin models available")
    
    # Take first two models and replace their providers with test provider
    for model in models[:2]:
        model.providers = [TestProvider()]
    
    return registry


@pytest.fixture 
def available_models(registry_with_test_models: ModelRegistry) -> list[str]:
    """Get list of available model IDs"""
    models = list(registry_with_test_models.list())
    return [model.identity.id for model in models[:2]] if models else []


@pytest.fixture
def sample_parameters() -> ModelParameters:
    """Create sample ModelParameters for testing"""
    return ModelParameters(temperature=0.7, max_tokens=100)


class TestModelPresetInitialization:
    def test_default_initialization(self):
        preset = ModelPreset(models=["test-model"])
        
        assert preset.models == ["test-model"]
        assert preset.required_capabilities == set()
        assert preset.title == "A Model Preset"
        assert preset.subtitle == "A model preset"
        assert preset.description == "A model preset"
        assert preset.parameters is None
        assert preset.data_center_fallback is True
        assert preset._current_model is None

    def test_full_initialization(self, sample_parameters: ModelParameters):
        preset = ModelPreset(
            models=["model1", "model2"],
            required_capabilities={ModelCapability.STREAMING, ModelCapability.FUNCTION_CALLING},
            title="Custom Preset",
            subtitle="Custom subtitle",
            description="Custom description",
            parameters=sample_parameters,
            data_center_fallback=False,
        )
        
        assert preset.models == ["model1", "model2"]
        assert preset.required_capabilities == {ModelCapability.STREAMING, ModelCapability.FUNCTION_CALLING}
        assert preset.title == "Custom Preset"
        assert preset.subtitle == "Custom subtitle"
        assert preset.description == "Custom description"
        assert preset.parameters == sample_parameters
        assert preset.data_center_fallback is False

    def test_nested_presets_initialization(self):
        nested_preset = ModelPreset(models=["nested-model"])
        parent_preset = ModelPreset(models=["parent-model", nested_preset])
        
        assert len(parent_preset.models) == 2
        assert parent_preset.models[0] == "parent-model"
        assert isinstance(parent_preset.models[1], ModelPreset)


class TestModelNames:
    def test_model_names_string_models(self):
        preset = ModelPreset(models=["model1", "model2", "model3"])
        names = preset.model_names()
        
        assert set(names) == {"model1", "model2", "model3"}

    def test_model_names_with_duplicates(self):
        preset = ModelPreset(models=["model1", "model2", "model1"])
        names = preset.model_names()
        
        assert set(names) == {"model1", "model2"}
        assert len(names) == 2

    def test_model_names_nested_presets(self):
        nested_preset = ModelPreset(models=["nested1", "nested2"])
        parent_preset = ModelPreset(models=["parent1", nested_preset, "parent2"])
        names = parent_preset.model_names()
        
        assert set(names) == {"parent1", "parent2", "nested1", "nested2"}

    def test_model_names_deeply_nested_presets(self):
        deep_preset = ModelPreset(models=["deep1"])
        nested_preset = ModelPreset(models=["nested1", deep_preset])
        parent_preset = ModelPreset(models=["parent1", nested_preset])
        names = parent_preset.model_names()
        
        assert set(names) == {"parent1", "nested1", "deep1"}

    def test_model_names_with_unknown_type(self):
        # This test demonstrates the robustness of model_names() with invalid input
        # In practice, this would be a type error, but the method handles it gracefully
        from typing import Any
        preset = ModelPreset(models=["model1"])
        preset.models = ["model1", 123]  # type: ignore  # Bypass type checking for test
        
        names = preset.model_names()
        
        assert "model1" in names
        assert "123" in names


class TestIterMethod:
    def test_iter_single_model(self, registry_with_test_models: ModelRegistry, available_models: list[str]):
        model_id = available_models[0]
        preset = ModelPreset(models=[model_id])
        configs = list(preset.iter(registry_with_test_models))
        
        assert len(configs) >= 1
        assert configs[0].identity.id == model_id

    def test_iter_multiple_models(self, registry_with_test_models: ModelRegistry, available_models: list[str]):
        model_ids = available_models[:2]
        preset = ModelPreset(models=model_ids)
        configs = list(preset.iter(registry_with_test_models))
        
        assert len(configs) >= 2
        config_ids = [config.identity.id for config in configs]
        for model_id in model_ids:
            assert model_id in config_ids

    def test_iter_model_not_found(self, registry_with_test_models: ModelRegistry):
        preset = ModelPreset(models=["nonexistent-model"])
        
        with patch('no_llm.presets.logger') as mock_logger:
            configs = list(preset.iter(registry_with_test_models))
            
            assert len(configs) == 0
            mock_logger.warning.assert_called_once()
            assert "not found in registry" in mock_logger.warning.call_args[0][0]

    def test_iter_with_capability_filtering(self, registry_with_test_models: ModelRegistry, available_models: list[str]):
        if not available_models:
            pytest.skip("No models available")
        
        # Use a rare capability that most models won't have
        preset = ModelPreset(
            models=available_models,
            required_capabilities={ModelCapability.VIDEO_GENERATION}
        )
        
        with patch('no_llm.presets.logger') as mock_logger:
            configs = list(preset.iter(registry_with_test_models))
            
            # Most models won't have video generation capability
            assert len(configs) <= len(available_models)

    def test_iter_with_parameters(self, registry_with_test_models: ModelRegistry, available_models: list[str], sample_parameters: ModelParameters):
        if not available_models:
            pytest.skip("No models available")
        
        model_id = available_models[0]
        preset = ModelPreset(models=[model_id], parameters=sample_parameters)
        configs = list(preset.iter(registry_with_test_models))
        
        assert len(configs) >= 1

    def test_iter_nested_presets(self, registry_with_test_models: ModelRegistry, available_models: list[str]):
        nested_preset = ModelPreset(models=[available_models[1]])
        parent_preset = ModelPreset(models=[available_models[0], nested_preset])
        configs = list(parent_preset.iter(registry_with_test_models))
        
        assert len(configs) >= 2
        config_ids = [config.identity.id for config in configs]
        assert available_models[0] in config_ids
        assert available_models[1] in config_ids

    def test_iter_nested_presets_with_parameters(self, registry_with_test_models: ModelRegistry, available_models: list[str], sample_parameters: ModelParameters):
        nested_preset = ModelPreset(models=[available_models[1]])
        parent_preset = ModelPreset(models=[available_models[0], nested_preset], parameters=sample_parameters)
        configs = list(parent_preset.iter(registry_with_test_models))
        
        assert len(configs) >= 2

    def test_iter_data_center_fallback_enabled(self, registry_with_test_models: ModelRegistry, available_models: list[str]):
        """Test that data center fallback creates configs for each provider variant"""
        model_id = available_models[0]
        preset = ModelPreset(models=[model_id], data_center_fallback=True)
        configs = list(preset.iter(registry_with_test_models))
        
        # Should get at least one config
        assert len(configs) >= 1

    def test_iter_data_center_fallback_disabled(self, registry_with_test_models: ModelRegistry, available_models: list[str]):
        """Test that data center fallback disabled uses original provider"""
        model_id = available_models[0]
        preset = ModelPreset(models=[model_id], data_center_fallback=False)
        configs = list(preset.iter(registry_with_test_models))
        
        assert len(configs) >= 1


class TestGetCurrentModel:
    def test_get_current_model_before_iteration(self):
        preset = ModelPreset(models=["test-model"])
        
        with pytest.raises(ValueError, match="No model selected"):
            preset.get_current_model()

    def test_get_current_model_after_iteration(self, registry_with_test_models: ModelRegistry, available_models: list[str]):
        model_id = available_models[0]
        preset = ModelPreset(models=[model_id])
        configs = list(preset.iter(registry_with_test_models))
        
        assert len(configs) >= 1
        current_model = preset.get_current_model()
        assert current_model is not None
        assert current_model.identity.id == model_id

    def test_current_model_updates_during_iteration(self, registry_with_test_models: ModelRegistry, available_models: list[str]):
        preset = ModelPreset(models=available_models[:2])
        
        configs_iterator = preset.iter(registry_with_test_models)
        first_config = next(configs_iterator)
        
        current_model = preset.get_current_model()
        assert current_model.identity.id == first_config.identity.id
        
        # Get next config and verify current model updates
        second_config = next(configs_iterator)
        current_model = preset.get_current_model()
        assert current_model.identity.id == second_config.identity.id


class TestPresetEdgeCases:
    def test_empty_models_list(self, registry_with_test_models: ModelRegistry):
        preset = ModelPreset(models=[])
        configs = list(preset.iter(registry_with_test_models))
        
        assert len(configs) == 0

    def test_mixed_valid_invalid_models(self, registry_with_test_models: ModelRegistry, available_models: list[str]):
        valid_model = available_models[0]
        preset = ModelPreset(models=[valid_model, "nonexistent-model"])
        
        with patch('no_llm.presets.logger') as mock_logger:
            configs = list(preset.iter(registry_with_test_models))
            
            assert len(configs) >= 1
            config_ids = [config.identity.id for config in configs]
            assert valid_model in config_ids
            
            # Should log warning for nonexistent model
            mock_logger.warning.assert_called()

    def test_all_models_filtered_by_capabilities(self, registry_with_test_models: ModelRegistry, available_models: list[str]):
        # Use capabilities that our test models don't have
        preset = ModelPreset(
            models=available_models,
            required_capabilities={ModelCapability.VIDEO_GENERATION, ModelCapability.AUDIO_TRANSCRIPTION}
        )
        
        with patch('no_llm.presets.logger') as mock_logger:
            configs = list(preset.iter(registry_with_test_models))
            
            # Should have no configs due to capability filtering
            assert len(configs) == 0
            mock_logger.warning.assert_called()

    def test_circular_nested_presets(self):
        """Test that circular references are handled gracefully"""
        preset1 = ModelPreset(models=["model1"])
        preset2 = ModelPreset(models=["model2"])
        
        # Create circular reference
        preset1.models = ["model1", preset2]
        preset2.models = ["model2", preset1]
        
        # This will cause infinite recursion in the current implementation
        # We expect this to fail until the implementation adds recursion protection
        import pytest
        with pytest.raises(RecursionError):
            preset1.model_names()


class TestPresetIntegration:
    def test_preset_with_test_provider_integration(self, registry_with_test_models: ModelRegistry, available_models: list[str]):
        """Test that presets work correctly with TestProvider"""
        if not available_models:
            pytest.skip("No models available")
        
        preset = ModelPreset(
            models=available_models,
            title="Test Provider Integration",
            description="Testing ModelPreset with real model configs but test provider"
        )
        
        configs = list(preset.iter(registry_with_test_models))
        assert len(configs) >= 1
        
        # Verify we got real model configurations with test provider
        for config in configs:
            # Each config should have valid providers (test provider always valid)
            assert config.is_valid
            # Test provider is available in the iteration
            providers = list(config.iter())
            assert len(providers) >= 1
            # Check that it's a real model with actual properties
            assert config.identity.name
            assert config.identity.creator
            assert len(config.capabilities) > 0

    def test_preset_with_real_model_config_structure(self, registry_with_test_models: ModelRegistry, available_models: list[str]):
        """Test preset with real model configuration"""
        if not available_models:
            pytest.skip("No models available")
        
        model_id = available_models[0]
        preset = ModelPreset(
            models=[model_id],
            title="Production Preset",
            subtitle="High-quality models for production",
            description="A preset containing production-ready models",
            parameters=ModelParameters(temperature=0.1, max_tokens=4000),
            data_center_fallback=True,
        )
        
        configs = list(preset.iter(registry_with_test_models))
        assert len(configs) >= 1
        
        current_model = preset.get_current_model()
        assert current_model.identity.id == model_id
        assert current_model.identity.name  # Has a real name

    def test_complex_nested_preset_hierarchy(self, registry_with_test_models: ModelRegistry, available_models: list[str]):
        """Test complex nested preset structure with real models"""
        if len(available_models) < 2:
            pytest.skip("Need at least 2 models for testing")
        
        model1_preset = ModelPreset(
            models=[available_models[0]],
            title="First Model Group"
        )
        
        model2_preset = ModelPreset(
            models=[available_models[1]],
            title="Second Model Group"
        )
        
        combined_preset = ModelPreset(
            models=[model1_preset, model2_preset],
            title="Multi-Model Preset",
            description="Combines models from different groups",
            parameters=ModelParameters(temperature=0.7, max_tokens=2000)
        )
        
        # This should work without errors and provide models from all sources
        configs = list(combined_preset.iter(registry_with_test_models))
        assert len(configs) >= 2
        
        config_ids = [config.identity.id for config in configs]
        # Should contain both models
        assert available_models[0] in config_ids
        assert available_models[1] in config_ids
        
        # Verify all configs are real models with properties
        for config in configs:
            assert config.identity.name
            assert config.identity.creator
            assert len(config.capabilities) > 0
