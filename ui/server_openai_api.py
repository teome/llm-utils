import os
from typing import List, Optional
import requests
from fire import Fire
import openai
import gradio as gr

# from ..endpoint_utils import retry_on_failure

DEFAULT_MODEL = "gpt-4"

try:
    api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ["OPENAI_API_KEY"]

if api_key is None:
    raise RuntimeError("OPENAI_API_KEY not set")

global client
client: Optional[openai.OpenAI] = None

def chat_completions_create(
    messages: List[openai.types.chat.ChatCompletionMessageParam],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 150,
    temperature: float = 0.8,
    client: Optional[openai.OpenAI] = client,
    stream: bool = True,
    max_retries: int = 2,
    timeout: int = 60,
    **kwargs):
    """Create chat completions"""
    if client is None:
        # Create an instance of the OpenAI class, assumes OPENAI_API_KEY is set
        client = openai.OpenAI(api_key=api_key, max_retries=max_retries, timeout=timeout)

    # @retry_on_failure(
    #     max_retries=max_retries,
    #     delay=delay,
    #     backoff=backoff,
    #     exceptions=(
    #         requests.exceptions.RequestException,
    #         openai.RateLimitError,
    #         openai.InternalServerError,
    #         openai.APIConnectionError),
    # )
    def _call():
        # Call the chat completions API
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **kwargs
        )
        return response
    return _call()


def predict(message, history, model=DEFAULT_MODEL, system_prompt="", max_tokens=1024, temperature=0.8):
    history_openai_format = []
    if system_prompt:
        history_openai_format.append({"role": "system", "content": system_prompt})
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})


    response = chat_completions_create(
        messages=history_openai_format,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial_message = partial_message + chunk.choices[0].delta.content
            yield partial_message


def main(model=DEFAULT_MODEL, max_retries=2, timeout=60, base_url=None):

    global client  # pylint: disable=global-statement
    client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout)

    with gr.Blocks() as demo:
        model  = gr.Dropdown([
            "gpt-3", "gpt-3.5-turbo",
            "gpt-4", "gpt-4-1106-preview", "gpt-4-0613", "gpt-4-0314"], label="Model")
        system_prompt = gr.Textbox("You are helpful AI.", label="System Prompt")
        temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.8, label="Temperature")
        max_tokens = gr.Number(value=1024, label="Max Tokens")


        gr.ChatInterface(
            predict,
            additional_inputs=[model, system_prompt, max_tokens, temperature],
        )

    demo.queue().launch()


if __name__ == "__main__":
    Fire(main)