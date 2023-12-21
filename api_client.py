import os
from typing import Optional
from fire import Fire

from endpoint_utils import rest_api_request, get_response, iterate_streaming_response



def main(
        url: str = "https://api.together.xyz/inference",
        api_key: Optional[str] = None,
        prompt: str = "Explain entropy in terms of information theory.",
        model="mistralai/Mixtral-8x7B-v0.1",
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
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
        }
    elif "openai" in url:
        json_data = {
            "messages": [{"role": "user", "content": prompt},],
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
        }

    response = rest_api_request(
        url=url,
        api_key=api_key,
        json_data=json_data,
        stream=stream,
    )

    def extract_message(response):
        if "together" in url or "vllm" in url:
            return response["output"]["choices"][0]["text"]
        elif "openai" in url:
            return response["choices"][0]["message"]["content"]
        else:
            raise NotImplementedError("No handling available for this url")


    if stream:
        for i, response in enumerate(iterate_streaming_response(response)):
            print(f"\n\nResponse {i}:")
            print(extract_message(response), flush=True)
    else:
        response = get_response(response)
        print(extract_message(response))


if __name__ == '__main__':
    Fire(main)
