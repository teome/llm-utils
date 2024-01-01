from functools import partial
import json
import os
from typing import List, Optional, Sequence
from warnings import warn
from fire import Fire
import gradio as gr
import openai
import sseclient

from llm_utils.endpoint_utils import rest_api_request


DEFAULT_MODEL = "gpt-4"
MODEL_LIST = [
    "gpt-3", "gpt-3.5-turbo",
    "gpt-4", "gpt-4-1106-preview", "gpt-4-0613", "gpt-4-0314",
    "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-v0.1",
    "togethercomputer/llama-2-70b-chat",
    "zero-one-ai/Yi-34B-Chat",
    "NousResearch/Nous-Hermes-2-Yi-34B",
]

SYSTEM_PROMPT = """\
You are a helpful, inciteful, accurate and up to date pair programmer. \
When beginning a response, you outline each step and the reasoning behind it, for the problem or task to be solved. Having done this, you should propose a solution. \
The solution can be in the form of code, which should be in codeblocks, or in the form of a plan or system description as appropriate.
"""


def _check_merge_system_prompt(model: str) -> bool:
    """Check if model is one that requires system prompt to be merged with history"""
    models_without_system_role = [
        "mistral", # TODO: there are others, but for now only care about this one
    ]

    for model_name in models_without_system_role:
        if model_name in model:
            return True
    return False


def prepare_openai_chat_history(
    message: str,
    history: List[Sequence[str]],
    system_prompt: Optional[str],
    merge_system_prompt: bool = False) -> List[openai.types.chat.ChatCompletionMessageParam]:
    """Prepare chat history for openai chat completions"""

    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})

    if system_prompt and not merge_system_prompt:
        history_openai_format.insert(0, {"role": "system", "content": system_prompt})
    elif system_prompt:
        history_openai_format[0]["content"] = f'{system_prompt}\n\n{history_openai_format[0]["content"]}'

    return history_openai_format


def chat_completions_create(
    messages: List[openai.types.chat.ChatCompletionMessageParam],
    client: openai.OpenAI,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 150,
    temperature: float = 0.8,
    stream: bool = True,
    **kwargs):
    """Create chat completions"""

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


def predict(
    message,
    history,
    model=DEFAULT_MODEL,
    system_prompt=SYSTEM_PROMPT,
    max_tokens=1024,
    temperature=0.8,
    frequency_penalty=0.0,
    client=None,
):
    history_openai_format = prepare_openai_chat_history(message, history, system_prompt)

    response = chat_completions_create(
        messages=history_openai_format,
        client=client,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        stream=True,
    )
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial_message = partial_message + chunk.choices[0].delta.content
            yield partial_message


def predict_inference(
    message,
    history,
    model=DEFAULT_MODEL,
    system_prompt=SYSTEM_PROMPT,
    max_tokens=1024,
    temperature=0.8,
    frequency_penalty=0.0,
    tokenizer=None,
    base_url=None,
    api_key=None,
    timeout=60,
    stream=True,
    ):
    """Predict using inference style endpoint via HTTP request"""

    # have to have them after other options due to the way gradio works to provide args not kwargs
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")
    if base_url is None:
        raise ValueError("base_url must be provided")
    if api_key is None:
        raise ValueError("api_key must not be None")

    history_openai_format = prepare_openai_chat_history(
        message, history, system_prompt, merge_system_prompt=_check_merge_system_prompt(model))
    prompt = str(tokenizer.apply_chat_template(history_openai_format, tokenize=False))

    # Note: bit of a hack to use the same variable for openai and together/others.
    # Openai has `frequency_penalty` and `presence_penalty` but togethercomputer has `repetition_penalty`.
    # Just use freq and rep via the same variable. But `frequency_penalty` is 0-2 and `repetition_penalty`
    # is 1-2 so need to offset and scale.
    # Not sure it's exactly the same or that together isn't using presence_penalty, but it's close enough for now.
    repetition_penalty = frequency_penalty / 2 + 1

    if "together" in str(base_url) or "vllm" in str(base_url):
        # TODO: check for vllm format, this is just a guess for now
        json_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "stream_tokens": stream,
        }
    else:
        raise NotImplementedError("Only togethercomputer inference style endpoints are supported")

    response = rest_api_request(
        url=base_url,
        api_key=api_key,
        json_data=json_data,
        stream=stream,
        timeout=timeout,
    )

    partial_message = ""
    client = sseclient.SSEClient(response)
    for event in client.events():
        if event.data != "[DONE]":
            partial_result = json.loads(event.data)
            partial_message += partial_result["choices"][0]["text"]
            yield partial_message


def main(model=DEFAULT_MODEL, max_retries=2, timeout=60, base_url=None, api_key=None):

    if api_key is None:
        api_key_env = None
        if base_url is None or "openai" in base_url:
            api_key_env = "OPENAI_API_KEY"
        elif "together" in base_url:
            api_key_env = "TOGETHER_API_KEY"
        elif "vllm" in base_url or "localhost" in base_url:
            api_key_env = "EMPTY"

        if api_key_env is None:
            warn('No known api key environment variable found for base_url, using empty string')
            api_key = ""
        else:
            try:
                api_key = os.environ[api_key_env]
            except KeyError:
                from dotenv import load_dotenv  # pylint: disable=import-outside-toplevel
                load_dotenv()
                api_key = os.environ[api_key_env]


    # setup either the client for openai compatible api or the predict function for inference style api
    if base_url is None or "chat" in base_url:
        client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout)
        predict_fn = partial(predict, client=client)
    else:
        # Assume we're accessing an inference style endpoint and need to creat the prompt ourselves
        from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel
        tokenizer = AutoTokenizer.from_pretrained(model)
        predict_fn = partial(predict_inference, tokenizer=tokenizer, base_url=base_url, api_key=api_key)


    with gr.Blocks() as demo:
        model  = gr.Dropdown(MODEL_LIST,
            value=model,
            allow_custom_value=True,
            label="Model")
        system_prompt = gr.Textbox(SYSTEM_PROMPT, label="System Prompt")
        max_tokens = gr.Number(value=1024, label="Max Tokens")
        temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.8, label="Temperature")
        frequency_penalty = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=0.0, label="Frequency Penalty (repetition for together)")
        # TODO: add top_p

        gr.ChatInterface(
            predict_fn,
            additional_inputs=[
                model, system_prompt, max_tokens, temperature, frequency_penalty,],
        )

    demo.queue().launch()


if __name__ == "__main__":
    Fire(main)
