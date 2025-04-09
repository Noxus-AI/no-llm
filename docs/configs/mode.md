# Model Modes

Model modes define the primary function of a model in stdllm. Each mode represents a specific type of task the model is designed to perform.

!!! warning "Limited Mode Support"
    Currently, stdllm primarily supports `CHAT` mode and chat-based models. Support for other modes (completion, embedding, image generation, audio) is under development and will be available in future releases.

## Available Modes

| Mode | Description | Common Use Cases |
|------|-------------|-----------------|
| `CHAT` | Interactive chat completion | - Conversational AI<br>- Virtual assistants<br>- Interactive Q&A |
| `COMPLETION` | Text completion | - Text generation<br>- Content creation<br>- Code completion |
| `EMBEDDING` | Vector embedding generation | - Semantic search<br>- Document similarity<br>- Text clustering |
| `IMAGE_GENERATION` | Image creation from text | - Art generation<br>- Design mockups<br>- Visual content creation |
| `AUDIO_TRANSCRIPTION` | Speech-to-text conversion | - Meeting transcription<br>- Subtitle generation<br>- Voice notes to text |
| `AUDIO_SPEECH` | Text-to-speech synthesis | - Audio content creation<br>- Accessibility features<br>- Voice assistants |

## Listing Models by Mode

You can list all models supporting a specific mode using the registry:

```python
from stdllm.config.enums import ModelMode
from stdllm.registry import ModelRegistry

registry = ModelRegistry()

# List all chat models
chat_models = list(registry.list_models(mode=ModelMode.CHAT))

# List chat models from a specific provider
openai_chat_models = list(registry.list_models(
    mode=ModelMode.CHAT,
    provider="openai"
))
```