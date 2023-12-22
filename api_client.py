import json
import os
from typing import Optional
from fire import Fire
import sseclient

from endpoint_utils import rest_api_request, get_response, iterate_streaming_response



def main(
        url: str = "https://api.together.xyz/inference",
        api_key: Optional[str] = None,
        prompt: str = "Explain entropy in terms of information theory.",
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.8,
        max_tokens=1024,
        repetition_penalty=1.2,
        stream: bool = True,
):

    if api_key is None:
        if "together" in url:
            api_key = os.getenv("TOGETHER_API_KEY")
        elif "openai" in url:
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("No api key provide or available from defaults (together, openai)")

    if "together" in url or "vllm" in url:
        json_data = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "stream_tokens": stream,
        }
        if "mistral" in model and "Instruct" in model:
            json_data["prompt"] = f"[INST] {prompt} [/INST]"
            json_data["stop"] = ["</s>", "[/INST]"]
        else:
            json_data["prompt"] = prompt
    elif "openai" in url:
        json_data = {
            "messages": [{"role": "user", "content": prompt},],
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "frequency_penalty": repetition_penalty,
            "stream": stream,
        }

    response = rest_api_request(
        url=url,
        api_key=api_key,
        json_data=json_data,
        stream=stream,
    )

    if "together" in url or "vllm" in url:
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data == "[DONE]":
                break

            partial_result = json.loads(event.data)
            token = partial_result["choices"][0]["text"]
            print(token, end="", flush=True)
    elif "openai" in url:
        raise NotImplementedError("OpenAI streaming not implemented")


if __name__ == '__main__':
    Fire(main)
