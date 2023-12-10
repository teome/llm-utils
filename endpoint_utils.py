"""Utility functions for interacting with the OpenAI API >=v1 and other endpoints"""
from functools import wraps
import time
from warnings import warn
from openai import OpenAI
from openai.types.chat import ChatCompletion
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


def openai_extract_chat_completion_message(response: ChatCompletion) -> dict:
    """Ensure handling of None values to allow for use back into a conversation
    
    Simply calls model_dump on a pydantic object with exclude_unset=True.
    
    This avoids issues with e.g. function call args that should be empty dict, not 
    Non if unset"""
    return response.choices[0].message.model_dump(exclude_unset=True)


def openai_chat_completions_create(
    messages,
    model="gpt-3.5-turbo",
    client=None,
    api_key=None,
    base_url=None,
    max_tokens=50,
    temperature=0.8,
    return_response=False,
    timeout=None,
    max_retries=3,
    delay=1,
    backoff=2,
    **kwargs
):
    """
    Call the chat completions API and return the generated completions.

    Args:
        messages (list): List of message objects representing the conversation.
        model (str, optional): The model to use for chat completions. Defaults to "gpt-3.5-turbo".
        client (object, optional): An instance of the OpenAI class. Defaults to None.
        api_key (str, optional): The OpenAI API key. Defaults to None.
        base_url (str, optional): The OpenAI API base URL. Defaults to None.
        max_tokens (int, optional): The maximum number of tokens in the generated completions. Defaults to 50.
        temperature (float, optional): Controls the randomness of the generated completions. Defaults to 0.8.
        return_response (bool, optional): Whether to return the full API response. Defaults to False.
        timeout (int, optional): The maximum time in seconds to wait for the API response. Defaults to None.
        max_retries (int, optional): The maximum number of retries in case of API failures. Defaults to 3.
        delay (int, optional): The delay in seconds between retries. Defaults to 1.
        backoff (int, optional): The backoff factor for exponential backoff between retries. Defaults to 2.
        **kwargs: Additional keyword arguments to pass to the API.

    Returns:
        str or tuple: The generated completions. If `return_response` is True, returns a tuple containing the completions and the API response.

    """
    if client is None:
        # Create an instance of the OpenAI class, assumes OPENAI_API_KEY is set
        client = OpenAI(api_key=api_key, base_url=base_url)

    @retry_on_failure(
        max_retries=max_retries,
        delay=delay,
        backoff=backoff,
        exceptions=(requests.exceptions.RequestException,),
    )
    def chat_completions_create():
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
    response = chat_completions_create()

    # Return the generated completions
    if return_response:
        return response.choices[0].message.content, response
    return response.choices[0].message.content
