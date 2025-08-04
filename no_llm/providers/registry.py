from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from loguru import logger
from pydantic import TypeAdapter, ValidationError

from no_llm._utils import _get_annotated_union_members
from no_llm.errors import ProviderNotFoundError
from no_llm.providers import AnyProvider
from no_llm.providers.config import ProviderConfiguration

if TYPE_CHECKING:
    from collections.abc import Iterator


class ProviderRegistry:
    def __init__(self, config_dir: str | Path | None = None):
        self._providers: dict[str, ProviderConfiguration] = {}
        self._config_dir = Path(config_dir) if config_dir else None

        logger.debug("Initializing ProviderRegistry")

        # Register builtin providers with default configurations
        self._register_builtin_providers()

        if config_dir:
            logger.debug(f"Using config directory: {config_dir}")
            self._load_configurations()

    def _register_builtin_providers(self) -> None:
        """Register builtin providers with default configurations"""
        logger.debug("Registering builtin providers")

        # Get all provider classes from the AnyProvider union
        # AnyProvider is Annotated[Union[...], Discriminator(...)]
        provider_classes: list[type[AnyProvider]] = _get_annotated_union_members(AnyProvider)

        for provider_class in provider_classes:
            try:
                provider = provider_class()

                self.register(provider)
                logger.debug(f"Registered builtin provider: {provider.id} ({provider.type})")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not register builtin provider {provider_class.__name__}: {e}")
                continue

    def _create_provider_from_config(self, config: dict) -> ProviderConfiguration:
        """Create a provider instance from YAML configuration"""
        try:
            return TypeAdapter(AnyProvider).validate_python(config)
        except ValidationError as e:
            logger.error(f"Failed to create provider from config: {e}")
            raise

    def register_providers_from_directory(self, providers_dir: Path | str) -> None:
        providers_dir = Path(providers_dir)
        if not providers_dir.exists():
            logger.warning(f"Providers directory not found: {providers_dir}")
            return

        logger.debug(f"Loading providers from {providers_dir}")
        logger.debug(f"Providers directory contents: {list(providers_dir.iterdir())}")

        yaml_files = []
        for ext in ["*.yml", "*.yaml"]:
            yaml_files.extend(list(providers_dir.glob(ext)))

        logger.debug(f"Found {len(yaml_files)} YAML files: {[f.name for f in yaml_files]}")

        for provider_file in yaml_files:
            provider_id = provider_file.stem
            try:
                logger.debug(f"Loading provider config from file: {provider_file}")
                with open(provider_file) as f:
                    config = yaml.safe_load(f)
                logger.debug(f"Loaded YAML config: {config}")

                provider = self._create_provider_from_config(config)
                self.register(provider)
                logger.debug(f"Registered provider from file: {provider_id} -> {provider.id} ({provider.type})")
            except Exception as e:  # noqa: BLE001
                logger.opt(exception=e).error(f"Error loading provider {provider_id}")

    def _load_configurations(self) -> None:
        if not self._config_dir:
            logger.warning("No config directory set")
            return

        providers_dir = self._config_dir / "providers"
        logger.debug(f"Providers directory path: {providers_dir}")
        logger.debug(f"Providers directory exists: {providers_dir.exists()}")
        if providers_dir.exists():
            logger.debug(f"Providers directory contents: {list(providers_dir.iterdir())}")
        self.register_providers_from_directory(providers_dir)

    def register(self, provider: ProviderConfiguration) -> None:
        """Register a provider instance"""
        if provider.id in self._providers:
            logger.debug(f"Overriding existing provider: {provider.id}")

        self._providers[provider.id] = provider
        logger.debug(f"Registered provider: {provider.id} ({provider.name}) type={provider.type}")

    def get(self, provider_id: str) -> ProviderConfiguration:
        """Get a provider by ID"""
        if provider_id not in self._providers:
            logger.error(f"Provider {provider_id} not found")
            raise ProviderNotFoundError(provider_id)
        return self._providers[provider_id]

    def list_by_type(
        self, provider_type: str, *, only_valid: bool = False, only_active: bool = False
    ) -> Iterator[ProviderConfiguration]:
        """Get all providers of a specific type

        Args:
            provider_type: The type of provider to filter by
            only_valid: If True, only return providers with valid environment setup
            only_active: If True, only return providers that are active
        """
        logger.debug(f"Getting providers by type: {provider_type} (only_valid={only_valid}, only_active={only_active})")
        for provider in self._providers.values():
            if provider.type != provider_type:
                continue
            if only_active and not provider.is_active:
                logger.debug(f"Skipping provider {provider.id} - inactive")
                continue
            if only_valid and not provider.is_valid:
                logger.debug(f"Skipping provider {provider.id} - invalid environment")
                continue
            yield provider

    def list(self, *, only_valid: bool = False, only_active: bool = False) -> Iterator[ProviderConfiguration]:
        """List all registered providers

        Args:
            only_valid: If True, only return providers with valid environment setup
            only_active: If True, only return providers that are active
        """
        logger.debug(f"Listing providers (only_valid={only_valid}, only_active={only_active})")

        for provider in self._providers.values():
            if only_active and not provider.is_active:
                logger.debug(f"Skipping provider {provider.id} - inactive")
                continue
            if only_valid and not provider.is_valid:
                logger.debug(f"Skipping provider {provider.id} - invalid environment")
                continue
            yield provider

    def set_active(self, provider_id: str, is_active: bool) -> None:
        """Set the active status of a provider"""
        if provider_id not in self._providers:
            logger.error(f"Cannot set active status: provider {provider_id} not found")
            raise ProviderNotFoundError(provider_id)

        self._providers[provider_id].is_active = is_active
        logger.debug(f"Set provider {provider_id} active status to: {is_active}")

    def remove(self, provider_id: str) -> None:
        """Remove a provider by ID"""
        if provider_id not in self._providers:
            logger.error(f"Cannot remove: provider {provider_id} not found")
            raise ProviderNotFoundError(provider_id)
        del self._providers[provider_id]
        logger.debug(f"Removed provider: {provider_id}")

    def reload(self) -> None:
        """Reload all provider configurations"""
        logger.debug("Reloading all configurations")
        self._providers.clear()
        self._register_builtin_providers()
        self._load_configurations()
