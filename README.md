# LLM-Utils

Utility library, scripts, tokenization implementation and comparisons, and a barebones gradio UI for chatting to various LLM endpoints.

Includes
- `llm_utils` module with utilities for tokenization, API calling e.g. OpenAI, Together.AI and OpenAI python library use
- Tokenization that's more correct than that provided by the majority of OSS libraries, together with notebooks exploring differences and verifying correctness. The whole issue of prompt formatting is currently a mess and can impact model inference performance.
- A minimal Gradio UI in `ui` for OpenAI library compatible endpoints and those that provide a REST API meaning that many of the open source and community models can be used with e.g. vLLM
