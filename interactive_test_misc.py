# %%
from typing import List

from transformers import AutoTokenizer

import conversation
from endpoint_utils import (
    OpenAI, openai_chat_completions_create, openai_extract_chat_completion_message,
    prepare_http_request_json, openai_http_api_request)

# %%
initial_messages = [
    {"role": "system", "content": "You are Samantha, a helpful, friendly, funny and highly knowledgeable assistant."},
    {"role": "user", "content": "What is the history of the Metalheadz label?"}
]
conv = conversation.get_conv_template("chatml")
conv.append_messages(messages=initial_messages)
conv.display_conversation()
print(conv.format_message(
    {"role": "user", "content": "What is the history of the Metalheadz label?"},
    color_scheme={"user": "blue", "assistant": "red"}))
print(conv.format_message(
    {"role": "user", "content": "What is the history of the Metalheadz label?"},
    color_scheme={"user": "blue", "assistant": "red"},
    detailed=False))

#%%

from dotenv import load_dotenv
load_dotenv()

# %%
MODEL = "gpt-3.5-turbo"
# %%

# Create an instance of the OpenAI class, assumes OPENAI_API_KEY is set
client = OpenAI()
# Call the chat completions API
message, response = openai_chat_completions_create(
    messages=conv.messages,
    model=MODEL,
    client=client,
    return_response=True,
    max_tokens=256,
    temperature=0.8,
    max_retries=1,
)

# %%
print(message)
print(response.model_dump_json(indent=2))

# %%
# conv.append_message(response.choices[0].message.model_dump(exclude_unset=True))
conv.append_message(openai_extract_chat_completion_message(response))


conv.display_conversation()
# %%

conv.append_message({"role": "user", "content": "Who were the founders?"})
prompt = conv.get_prompt()
print(prompt)
# %%
message, response = openai_chat_completions_create(
    messages=conv.messages,
    model=MODEL,
    client=client,
    return_response=True,
    max_tokens=256,
    temperature=0.8,
    max_retries=1,
)

# %%
conv.append_message(openai_extract_chat_completion_message(response))
print(response.model_dump_json(indent=2))
print()
conv.display_conversation(detailed=True)

print("\n\nString prompt format:\n")
print(conv.get_prompt(add_generator_prompt=True))

# %%
# %%[markdowm]
# ## Test the completions api with more direct calling via json
# The same functionality can be used in most other endpoints, not much baked in that's OpenAI specific
# %%
conv.messages = initial_messages
json_data = prepare_http_request_json(messages=conv.messages, model=MODEL,)

# %%
request = openai_http_api_request(url="https://api.openai.com/v1/chat/completions", json_data=json_data, timeout=60)
# %%
print(request.json())
req_json = request.json()["choices"][0]
print(req_json)
# %%

from openai.types.chat import ChatCompletion
completion = ChatCompletion(**request.json())  # validate with pydantic types
print("pydantic validated ChatCompletion\n", completion)

# %%
completion.choices[0].message.model_dump(exclude_unset=True)
# %%
print(openai_extract_chat_completion_message(completion))
conv.append_message(completion.choices[0].message.model_dump(exclude_unset=True))
conv.display_conversation()

# %%
# alternatively just use the dict from json directly if other APIs
conv.messages.pop()
conv.append_message(request.json()["choices"][0]["message"])
conv.display_conversation()
