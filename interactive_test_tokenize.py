# %%[markdown]
# # Experimentation and testing of prompt formatting with tokenizer for llama and mistral models

# %%

from typing import List

from transformers import AutoTokenizer

import tokenize_llama_utils
from tokenize_llama_utils import Tokenizer, LlamaPrompt
import tokenize_mistral_utils

# %%

# messages: List[tokenize_llama_utils.Message] = [
#     {"role": "user", "content": "What is your favourite condiment?"},
#     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#     {"role": "user", "content": "Do you have mayonnaise recipes?"},]

messages = [
    {"role": "user", "content": "2+2"},
    {"role": "assistant", "content": "4!"},
    {"role": "user", "content": "+2"},
    {"role": "assistant", "content": "6!"},
    {"role": "user", "content": "+4"},
]


# %%[markdown] ##################################################################
# ## Mistral
# %%

from transformers import AutoTokenizer
from typing import List, Dict

def build_prompt(
    messages: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
):
    prompt = ""
    for i, msg in enumerate(messages):
        is_user = {"user": True, "assistant": False}[msg["role"]]
        assert (i % 2 == 0) == is_user
        content = msg["content"]
        assert content == content.strip()
        if is_user:
            prompt += f"[INST] {content} [/INST]"
        else:
            prompt += f" {content}</s>"
    tokens_ids = tokenizer.encode(prompt)
    token_str = tokenizer.convert_ids_to_tokens(tokens_ids)
    return tokens_ids, token_str

hf_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tok = Tokenizer(model_path="models/mistralai/Mistral-7B-Instruct-v0.2/tokenizer.model")

# %%
print(hf_tok.__class__)
print(tok.__class__)

tokens_ids, token_str = build_prompt(messages, hf_tok)
print(tokens_ids)
print(token_str)
print(hf_tok.decode(tokens_ids, skip_special_tokens=True))


# %%

# prompt = ""
# for i, msg in enumerate(messages):
#     is_user = {"user": True, "assistant": False}[msg["role"]]
#     assert (i % 2 == 0) == is_user
#     content = msg["content"]
#     assert content == content.strip()
#     if is_user:
#         prompt += f"[INST] {content} [/INST]"
#     else:
#         prompt += f" {content}</s>"

# token_ids = tok.encode(prompt, bos=True, eos=False)
# token_str = tok.decode(tokens_ids)
# print(token_ids)
# print(token_str)
# print(tok.sp_model.id_to_piece(token_ids))



# TODO: modify code to use this method, this is working
# Note that this is only for constructing the assistant messages manually like this if they've
# been converted to text. If dealing with the tokens of the assistant message, it will already have </s>
# TODO update tests to aggree with this
prompt_toks = [tok.bos_id]
for i, msg in enumerate(messages):
    is_user = {"user": True, "assistant": False}[msg["role"]]
    assert (i % 2 == 0) == is_user
    content = msg["content"]
    assert content == content.strip()

    if is_user:
        prompt_toks += tok.encode(f"[INST] {content} [/INST]", bos=False, eos=False)
    else:
        prompt_toks += tok.encode(f"{content}", bos=False, eos=True)

token_ids = prompt_toks
token_str = tok.decode(tokens_ids)
print(token_ids)
print(tok.sp_model.id_to_piece(token_ids))
print(token_str)



# %%

# Expected values
tokens_ids, token_str = build_prompt(messages, tok)
print(tokens_ids)
expected_tok_ids = [1, 733, 16289, 28793, 28705, 28750, 28806, 28750, 733, 28748, 16289, 28793, 28705, 28781, 28808, 2, 733, 16289, 28793, 648, 28750, 733, 28748, 16289, 28793, 28705, 28784, 28808, 2, 733, 16289, 28793, 648, 28781, 733, 28748, 16289, 28793]
print(token_str)
expected_tok_ids_to_tokens = ['<s>', '▁[', 'INST', ']', '▁', '2', '+', '2', '▁[', '/', 'INST', ']', '▁', '4', '!', '</s>', '▁[', 'INST', ']', '▁+', '2', '▁[', '/', 'INST', ']', '▁', '6', '!', '</s>', '▁[', 'INST', ']', '▁+', '4', '▁[', '/', 'INST', ']']










# %%[markdown] ##################################################################
# ## Llama

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
hf_mistral_prompt_tokens = hf_tokenizer.apply_chat_template(messages)
hf_decoded_prompt = hf_tokenizer.decode(hf_mistral_prompt_tokens, skip_special_tokens=False)
print(mistral_prompt_tokens)
print(hf_prompt_tokens)
print(mistral_decoded_prompt)
print(hf_decoded_prompt)

assert mistral_prompt_tokens == hf_mistral_prompt_tokens, 'Prompt tokens are different'
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
hf_tokenizer.apply_chat_template(messages_with_system)

# %%[markdown] ##################################################################
# ## Mistral alternative approach thanks to talking to the devs themselves
#
# They suggested how to implement and test the tokenization and avoid multi-interaction string issues

