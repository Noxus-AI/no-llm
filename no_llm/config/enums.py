from enum import Enum


class ModelMode(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image"
    AUDIO_TRANSCRIPTION = "audio_in"
    AUDIO_SPEECH = "audio_out"


class ModelCapability(str, Enum):
    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    PARALLEL_FUNCTION_CALLING = "parallel_function_calling"
    VISION = "vision"
    SYSTEM_PROMPT = "system_prompt"
    TOOLS = "tools"
    JSON_MODE = "json_mode"
    STRUCTURED_OUTPUT = "structured_output"
    REASONING = "reasoning"
    WEB_SEARCH = "web_search"
    AUDIO_TRANSCRIPTION = "audio_in"
    AUDIO_SPEECH = "audio_out"
    VIDEO_TRANSCRIPTION = "video_in"
    VIDEO_GENERATION = "video_out"
