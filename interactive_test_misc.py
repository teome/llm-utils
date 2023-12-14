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




###################################################
###################################################
# %%[markdown]
# ## Experimenting with prompt formatting with tokenizer more directly for llama and mistral models

# %%
import tokenize_llama_utils
from tokenize_llama_utils import Tokenizer, LlamaPrompt

messages: List[tokenize_llama_utils.Message] = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"},]


# %%[markdown]
# ### Llama first

# %%

tokenizer = Tokenizer(model_path="models/meta-llama/Llama-2-7b-chat/tokenizer.model")
print(tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id)
print(tokenizer.decode([tokenizer.bos_id, tokenizer.eos_id,]))

prompt_tokens = LlamaPrompt.chat_completion([messages], tokenizer)
print('MetaAI implementation sentencpiece tokenizer')
print(prompt_tokens)
print(tokenizer.decode(prompt_tokens[0]))

# %%
# Quick test to see equivalence. Notet that using the sentencepiece tokenizer directly we
# get a different result. Seems it't not encoding the special tags as expected (or as HF tokenizer does),
# so DON'T DO THIS
text = "<s> [INST] What is your favourite condiment? [/INST] " \
"Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!  </s>" \
"<s> [INST] Do you have mayonnaise recipes? [/INST] "

prompt_tokens = tokenizer.encode(text, bos=False, eos=False)
print(prompt_tokens)
print('HF implementation tokenizer and chat template from raw string')
print(f'Input text:\n{text}\n')
print(print(f'Output text:\n{tokenizer.decode(prompt_tokens)}'))

# %%
# for comparison, look at the same but with HF tokenizer -- Note we have to use the HF tokenizer to get chat template

hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
print(hf_tokenizer.bos_token_id, hf_tokenizer.eos_token_id, hf_tokenizer.pad_token_id)
print('HF implementation tokenizer and chat template from messages list of dicts')

print(hf_tokenizer.decode([hf_tokenizer.bos_token_id, hf_tokenizer.eos_token_id,]))
hf_prompt_tokens = hf_tokenizer.apply_chat_template(messages)
print(hf_prompt_tokens)
print(hf_tokenizer.decode(hf_prompt_tokens, skip_special_tokens=True))



###################################################
# %%[markdown]

# ### Direct comparison on tokenization and decoding
# %%

print('Compare tokenized prompts and decoded strings token by token\n')

prompt_tokens = LlamaPrompt.chat_completion([messages], tokenizer)[0]
print('MetaAI implementation sentencpiece tokenizer from messages list of dicts')
print(prompt_tokens)
decoded_prompt = tokenizer.decode(prompt_tokens)
print(decoded_prompt)

hf_prompt_tokens = hf_tokenizer.apply_chat_template(messages)
print('\n\nHF tokenizer with chat template from messages list of dicts')
print(hf_prompt_tokens)
hf_decoded_prompt = hf_tokenizer.decode(hf_prompt_tokens, skip_special_tokens=True)
print(hf_decoded_prompt)

print('\n\nTest equivalence of prompt tokens and decoded strings')
if prompt_tokens == hf_prompt_tokens:
    print('Prompt tokens are the same')
else:
    print('Prompt tokens are different')
    assert prompt_tokens == hf_prompt_tokens, 'Prompt tokens are different'

if decoded_prompt == hf_decoded_prompt:
    print('Decoded prompts are the same')
else:
    print('\n\nDecoded prompts are different')
    print(f'MetaAI decoded prompt:\n{decoded_prompt}')
    print(f'HF decoded prompt:\n{hf_decoded_prompt}')
    assert decoded_prompt == hf_decoded_prompt, 'Decoded prompts are different'



# %%[markdown] ##################################################################

# ### Mistral version of the Llama tokenization without BOS at the start of every user message
# %%
from tokenize_mistral_utils import MistralPrompt

# %%
mistral_tokenizer = Tokenizer(model_path="models/mistralai/Mixtral-8x7B-Instruct-v0.1/tokenizer.model")
# %%
print('Mistral implementation sentencpiece tokenizer from messages list of dicts')
mistral_prompt_tokens = MistralPrompt.chat_completion([messages], tokenizer)[0]
print(mistral_prompt_tokens)
mistral_decoded_prompt = tokenizer.decode(mistral_prompt_tokens)
print(mistral_decoded_prompt)

# %%
hf_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
hf_missral_prompt_tokens = hf_tokenizer.apply_chat_template(messages)

# %%
# %%[markdown]
# ### Check Mistral prompt when using a system prompt in the messages list, with and withuot the system tags
# %%
messages_with_system = [
    {"role": "system", "content": "You are Samantha, a helpful, friendly, funny and highly knowledgeable assistant."},
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"},]
print('Mistral prompt with system prompt and no system tags')
print(tokenizer.decode(MistralPrompt.chat_completion([messages_with_system], tokenizer)[0]))
print('Mistral prompt with system prompt and no system tags')
print(tokenizer.decode(MistralPrompt.chat_completion([messages_with_system], tokenizer, include_system_tags=True)[0]))
# %%
