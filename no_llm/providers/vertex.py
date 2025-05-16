from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal, assert_never, cast

from anthropic import AsyncAnthropicVertex
from mistralai_gcp import MistralGoogleCloud
from pydantic import Field, PrivateAttr
from pydantic_ai.providers.anthropic import (
    AnthropicProvider as PydanticAnthropicProvider,
)
from pydantic_ai.providers.google_vertex import (
    GoogleVertexProvider as PydanticGoogleVertexProvider,
)
from pydantic_ai.providers.google_vertex import VertexAiRegion
from pydantic_ai.providers.mistral import MistralProvider as PydanticMistralProvider

from no_llm.providers.base import Provider
from no_llm.providers.env_var import EnvVar

if TYPE_CHECKING:
    from collections.abc import Iterator


def pydantic_mistral_gcp_patch():
    from mistralai_gcp import (
        CompletionChunk as MistralCompletionChunk,
    )
    from mistralai_gcp import (
        Content as MistralContent,
    )
    from mistralai_gcp import (
        ContentChunk as MistralContentChunk,
    )
    from mistralai_gcp import (
        FunctionCall as MistralFunctionCall,
    )
    from mistralai_gcp import (
        OptionalNullable as MistralOptionalNullable,
    )
    from mistralai_gcp import (
        TextChunk as MistralTextChunk,
    )
    from mistralai_gcp import (
        ToolChoiceEnum as MistralToolChoiceEnum,
    )
    from mistralai_gcp.models import (
        ChatCompletionResponse as MistralChatCompletionResponse,
    )
    from mistralai_gcp.models import (
        CompletionEvent as MistralCompletionEvent,
    )
    from mistralai_gcp.models import (
        Messages as MistralMessages,
    )
    from mistralai_gcp.models import (
        Tool as MistralTool,
    )
    from mistralai_gcp.models import (
        ToolCall as MistralToolCall,
    )
    from mistralai_gcp.models.assistantmessage import (
        AssistantMessage as MistralAssistantMessage,
    )
    from mistralai_gcp.models.function import Function as MistralFunction
    from mistralai_gcp.models.systemmessage import SystemMessage as MistralSystemMessage
    from mistralai_gcp.models.toolmessage import ToolMessage as MistralToolMessage
    from mistralai_gcp.models.usermessage import UserMessage as MistralUserMessage
    from mistralai_gcp.types.basemodel import Unset as MistralUnset
    from mistralai_gcp.utils.eventstreaming import (
        EventStreamAsync as MistralEventStreamAsync,
    )

    # from mistralai_gcp.models.imageurl import (
    #     ImageURL as MistralImageURL,
    #     ImageURLChunk as MistralImageURLChunk,
    # )

    sys.modules["pydantic_ai.models.mistral"].MistralUserMessage = MistralUserMessage  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralSystemMessage = MistralSystemMessage  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralAssistantMessage = MistralAssistantMessage  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralFunction = MistralFunction  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralToolMessage = MistralToolMessage  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralChatCompletionResponse = MistralChatCompletionResponse  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralCompletionEvent = MistralCompletionEvent  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralMessages = MistralMessages  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralTool = MistralTool  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralToolCall = MistralToolCall  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralUnset = MistralUnset  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralEventStreamAsync = MistralEventStreamAsync  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralOptionalNullable = MistralOptionalNullable  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralTextChunk = MistralTextChunk  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralToolChoiceEnum = MistralToolChoiceEnum  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralCompletionChunk = MistralCompletionChunk  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralContent = MistralContent  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralContentChunk = MistralContentChunk  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralFunctionCall = MistralFunctionCall  # type: ignore
    # sys.modules['pydantic_ai.models.mistral'].MistralImageURL = MistralImageURL  # type: ignore
    # sys.modules['pydantic_ai.models.mistral'].MistralImageURLChunk = MistralImageURLChunk  # type: ignore


class VertexProvider(Provider):
    """Google Vertex AI provider configuration"""

    type: Literal["vertex"] = "vertex"
    name: str = "Vertex AI"
    project_id: EnvVar[str] = Field(default_factory=lambda: EnvVar[str]("$VERTEX_PROJECT_ID"))
    locations: list[str] = Field(default=["us-central1", "europe-west1"])
    model_family: Literal["gemini", "claude", "mistral", "llama"] = Field(
        description="The family of models to use",
    )
    _value: str | None = PrivateAttr(default=None)

    def iter(self) -> Iterator[Provider]:
        if not self.has_valid_env():
            return

        for location in self.locations:
            provider = self.model_copy()
            provider._value = location  # noqa: SLF001
            yield provider

    @property
    def current(self) -> str:
        """Get current value, defaulting to first location if not set"""
        return self._value or self.locations[0]

    def reset_variants(self) -> None:
        self._value = None

    def to_pydantic(
        self,
    ) -> PydanticGoogleVertexProvider | PydanticAnthropicProvider | PydanticMistralProvider:
        if self.model_family == "gemini":
            return PydanticGoogleVertexProvider(
                project_id=str(self.project_id),
                region=cast(VertexAiRegion, self.current),
            )
        elif self.model_family == "claude":
            return PydanticAnthropicProvider(
                anthropic_client=AsyncAnthropicVertex(  # type: ignore
                    project_id=str(self.project_id),
                    region=cast(VertexAiRegion, self.current),
                ),
            )
        elif self.model_family == "mistral":
            pydantic_mistral_gcp_patch()
            return PydanticMistralProvider(
                mistral_client=MistralGoogleCloud(  # type: ignore
                    project_id=str(self.project_id),
                    region=cast(VertexAiRegion, self.current),
                ),
            )
        elif self.model_family == "llama":
            msg = "LLama is not supported in Vertex AI"
            raise NotImplementedError(msg)
        else:
            assert_never(self.model_family)
