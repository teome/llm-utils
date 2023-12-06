"""Utilitu functions for interacting with the OpenAI API >=v1"""
from functools import wraps
import time
from warnings import warn
from openai import OpenAI
# from openai.resources.chat.completions import Completions
import requests


def retry_on_failure(max_retries=3, delay=1, backoff=2, exceptions=(requests.exceptions.RequestException,)):
    """
    Decorator for retrying a function if it raises an exception.

    max_retries -- Max number of retries
    delay -- Initial delay between retries
    backoff -- Multiplier applied to delay between retries
    exceptions -- A tuple of exceptions to catch. Defaults to catching network-related exceptions.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_retries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as ex:
                    warn(f"P{str(ex)}, Retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator


def call_chat_completions(
    messages,
    model="gpt-3.5-turbo",
    client=None,
    max_tokens=50,
    temperature=0.8,
    return_response=False,
    timeout=None,
    max_retries=3,
    delay=1,
    backoff=2,
    **kwargs
):
    """Call the chat completions API and return the generated completions."""
    if client is None:
        # Create an instance of the OpenAI class, assumes OPENAI_API_KEY is set
        client = OpenAI()

    @retry_on_failure(
        max_retries=max_retries,
        delay=delay,
        backoff=backoff,
        exceptions=(requests.exceptions.RequestException,),
    )
    def create_chat_completions():
        # Call the chat completions API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            **kwargs
        )
        return response

    response = None
    response = create_chat_completions()

    # Return the generated completions
    if return_response:
        return create_chat_completions().choices[0].message.content, response
    return create_chat_completions().choices[0].message.content
