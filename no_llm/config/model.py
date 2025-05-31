from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai.settings import ModelSettings

from no_llm.config.benchmarks import BenchmarkScores
from no_llm.config.enums import ModelCapability, ModelMode
from no_llm.config.errors import MissingCapabilitiesError
from no_llm.config.integrations import IntegrationAliases
from no_llm.config.metadata import ModelMetadata
from no_llm.config.parameters import (
    NOT_GIVEN,
    ConfigurableModelParameters,
    ModelParameters,
)
from no_llm.config.properties import ModelProperties
from no_llm.providers import Provider, Providers


class ModelIdentity(BaseModel):
    id: str = Field(description="Unique identifier for the model")
    name: str = Field(description="Display name")
    version: str = Field(description="Model version")
    description: str = Field(description="Detailed description")
    creator: str = Field(description="Creator of the model")
    model_api_name: str | None = Field(default=None, description="Model API name")
    base_config: str | None = Field(default=None, description="Base config")


class ModelConstraints(BaseModel):
    max_input_tokens: int = Field(gt=0, description="Maximum input size")
    max_output_tokens: int = Field(gt=0, description="Maximum output size")

    def estimate_exceeds_input_limit(self, text: str) -> bool:
        chars_per_token = 4
        estimated_tokens = len(text) // chars_per_token
        return estimated_tokens > self.max_input_tokens


class ModelConfiguration(BaseModel):
    identity: ModelIdentity
    providers: Sequence[Providers] = Field(default_factory=list, description="Provider configuration", min_length=1)
    mode: ModelMode
    capabilities: set[ModelCapability]
    constraints: ModelConstraints
    properties: ModelProperties | None = Field(default=None, description="Model properties")
    parameters: ConfigurableModelParameters = Field(
        default_factory=ConfigurableModelParameters,
        description="Model parameters with their constraints",
    )
    metadata: ModelMetadata
    benchmarks: BenchmarkScores | None = Field(default=None, description="Model benchmark scores")
    integration_aliases: IntegrationAliases | None = Field(default=None, description="Integration aliases")
    extra: dict[str, Any] = Field(default_factory=dict, description="Extra model configuration")
    model_config = {"json_encoders": {set[ModelCapability]: lambda x: sorted(x, key=lambda c: c.value)}}

    def iter(self) -> Iterator[Provider]:
        for provider in self.providers:
            yield from provider.iter()

    def check_capabilities(self, capabilities: set[ModelCapability], mode: Literal["any", "all"] = "any") -> bool:
        if mode == "any":
            return bool(capabilities.intersection(self.capabilities))
        return capabilities.issubset(self.capabilities)

    def assert_capabilities(self, capabilities: set[ModelCapability], mode: Literal["any", "all"] = "any") -> None:
        if not self.check_capabilities(capabilities, mode):
            raise MissingCapabilitiesError(self.identity.name, list(capabilities), list(self.capabilities))

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> tuple[float, float]:
        if self.metadata.pricing.token_prices is None:
            msg = "Token pricing not available for this model. Character level pricing is not supported yet."
            raise NotImplementedError(msg)

        input_cost = input_tokens * self.metadata.pricing.token_prices.input_price_per_1k / 1000
        output_cost = output_tokens * self.metadata.pricing.token_prices.output_price_per_1k / 1000
        return input_cost, output_cost

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ModelConfiguration:
        parameters_cfg = config.pop("parameters", {})
        param_cls = cls.model_fields["parameters"].annotation
        if param_cls is None:
            msg = "Parameters class is not defined"
            raise ValueError(msg)
        parameters = param_cls.from_config(parameters_cfg)
        return cls(**config, parameters=parameters)

    def set_parameters(self, parameters: ModelParameters) -> None:
        """Set parameters from a dictionary"""
        copied_parameters = parameters.model_copy(deep=True)
        if (
            copied_parameters.model_override
            and copied_parameters.model_override != NOT_GIVEN
            and self.identity.id in copied_parameters.model_override
        ):
            copied_parameters = copied_parameters & copied_parameters.model_override[self.identity.id]

        for key, value in copied_parameters.model_dump(exclude_defaults=True).items():
            if key in self.parameters.model_fields:
                self.parameters._validate_and_update_parameter(  # noqa: SLF001
                    key, value, capabilities=self.capabilities
                )
                # setattr(self.parameters, key, value)

    def to_pydantic_settings(self) -> ModelSettings:
        # TODO: translate to each specific model
        return ModelSettings(**self.parameters.model_dump())
