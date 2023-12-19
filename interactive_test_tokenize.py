# %%[markdown]
# # Experimentation and testing of prompt formatting with tokenizer for Mistral and Llama models

# %%

from typing import List, Dict

from transformers import AutoTokenizer

import tokenize_llama_utils
from tokenize_llama_utils import LlamaPrompt
import tokenize_mistral_utils
from tokenize_mistral_utils import MistralPrompt

# %%
messages = [
    {"role": "user", "content": "2+2"},
    {"role": "assistant", "content": "4!"},
    {"role": "user", "content": "+2"},
    {"role": "assistant", "content": "6!"},
    {"role": "user", "content": "+4"},
]

expected_tok_ids = [1, 733, 16289, 28793, 28705, 28750, 28806, 28750, 733, 28748, 16289, 28793, 28705, 28781, 28808, 2, 733, 16289, 28793, 648, 28750, 733, 28748, 16289, 28793, 28705, 28784, 28808, 2, 733, 16289, 28793, 648, 28781, 733, 28748, 16289, 28793]
expected_tok_ids_to_tokens = ['<s>', '▁[', 'INST', ']', '▁', '2', '+', '2', '▁[', '/', 'INST', ']', '▁', '4', '!', '</s>', '▁[', 'INST', ']', '▁+', '2', '▁[', '/', 'INST', ']', '▁', '6', '!', '</s>', '▁[', 'INST', ']', '▁+', '4', '▁[', '/', 'INST', ']']

# %%[markdown] ##################################################################
# ## Mistral
# This is Mistral team provided reference code for tokenizattion
# Other methods must be equivalent to this
# %%

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
    print(f'Prompt:\n{prompt}')
    tokens_ids = tokenizer.encode(prompt)
    token_str = tokenizer.convert_ids_to_tokens(tokens_ids)
    return tokens_ids, token_str

hf_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tok = tokenize_mistral_utils.Tokenizer(model_path="models/mistralai/Mistral-7B-Instruct-v0.2/tokenizer.model")

print('HF tokenizer:', type(hf_tok))
print('Mistral reference (sentencepiece) tokenizer:', type(tok), type(tok._model))

# %%

mistral_ref_token_ids, mistral_ref_token_str = build_prompt(messages, hf_tok)
print(mistral_ref_token_ids)
print(mistral_ref_token_str)
print(hf_tok.decode(mistral_ref_token_ids, skip_special_tokens=True))
print(hf_tok.decode(mistral_ref_token_ids, skip_special_tokens=False))


# %%[markdown]
# Same but using the sentencepiece tokenizer directly and Mistral's class and same string formatting.
# This doesn't work correctly because of the handling of special tags in sentencepiece...
# '</s>' gets split and other tags are split into multiple tokens
#
# Lesson is just work more closely with tokens if possible. Can to some extent with HF tokenizer but still
# need to be careful and some of the chat_templates are still a little different.
# %%
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
print(f'Prompt:\n{prompt}')

mistral_sp_ids = tok.encode(prompt, bos=True, eos=False)
mistral_sp_str = tok.decode(mistral_sp_ids)
print(mistral_sp_ids)
print(mistral_sp_str)
print(tok._model.id_to_piece(mistral_sp_ids))


# %%[markdown]
# Both of the following methods work, the first more similar to the original Llama code.
#
# Despite this, the second method is probably better as it's more explicit and makes
# fewer assumptions about the input data.
#
# In both cases this is working because it's seperately encoding either messages or pairs
# of messages, not the whole mult-turn dialog at once.

# %%
prompt = ""
dialog_tokens: List[int] = [tok.bos_id]
for prompt, answer in zip(messages[::2], messages[1::2]):
    dialog_tokens += tok.encode(
        f"[INST] {(prompt['content']).strip()} [/INST] {(answer['content']).strip()}",
        bos=False,
        eos=True,
    )
if messages[-1]["role"] == "user":
    dialog_tokens += tok.encode(
        f"[INST] {(messages[-1]['content']).strip()} [/INST]",
        bos=False,
        eos=False,
    )


token_ids = dialog_tokens
token_str = tok.decode(token_ids)
print(token_ids)
print(tok._model.id_to_piece(token_ids))
print(token_str)

mistral_llama_sp_ids = token_ids.copy()
mistral_llama_sp_str = tok._model.id_to_piece(token_ids)

# %%[markdown]
# Seperate tokenization of user and assistant messages and using tokenizer directly and Mistral's class.
#
# This feels like the closest thing to what we would actually have with a real user and assistant.
# That said, it's not what llama did
# %%
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
token_str = tok.decode(token_ids)
print(token_ids)
print(tok._model.id_to_piece(token_ids))
print(token_str)

mistral_working_sp_ids = token_ids.copy()
mistral_working_sp_str = tok._model.id_to_piece(token_ids)


# %%[markdown]
# ---
#
# ### Verifying equivalence of the two methods and implementation in MistralPrompt class
# The approach from the Mistral team using HF tokenizer and second of these reference methods
# using sentencepiece are implemented in the tokenizer_mistral_utils.py 'MistralPrompt' class
#
# Verify equivalence of the two methods
# %%
mistral_prompt = tokenize_mistral_utils.MistralPrompt

token_ids_hf = mistral_prompt.encode_instruct_hf(messages, hf_tok)
token_str_hf = hf_tok.convert_ids_to_tokens(token_ids_hf)
print('HF tokenizer and string prompt')
print(token_ids_hf)
print(token_str_hf)
print(hf_tok.decode(token_ids_hf, skip_special_tokens=True))

token_ids_sp = mistral_prompt.encode_instruct_mistral(messages, tok)
token_str_sp = tok._model.id_to_piece(token_ids_sp)
print('Sentencepiece tokenizer and per-message tokenization')
print(token_ids_sp)
print(token_str_sp)
print(tok.decode(token_ids_sp))

assert token_ids_hf == token_ids_sp, 'Prompt tokens are different'
assert token_str_hf == token_str_sp, 'Prompt string encodings are different'
assert token_ids_hf == mistral_ref_token_ids, 'Prompt tokens are different'
assert token_str_hf == mistral_ref_token_str, 'Prompt string encodings are different'
print('\n\n***All tests passed***\n\n')

# %%[markdown]
# ### Test system prompt encoding with the system tags
#
# Mistral don't use these tags but the functioality may be useful and the tags can be
# overridden to be anything using the class members `B_SYS`, `E_SYS`
# %%
messages_with_system = [{'role': 'system', 'content': 'You are a helpful AI.'}] + messages

token_ids_hf = MistralPrompt.encode_instruct_hf(messages_with_system, hf_tok, include_system_tags=True)
token_str_hf = hf_tok.convert_ids_to_tokens(token_ids_hf)
print('HF tokenizer and string prompt')
print(token_ids_hf)
print(token_str_hf)
print(hf_tok.decode(token_ids_hf, skip_special_tokens=True))

token_ids_sp = MistralPrompt.encode_instruct_mistral(messages_with_system, tok, include_system_tags=True)
token_str_sp = tok._model.id_to_piece(token_ids_sp)
print('Sentencepiece tokenizer and per-message tokenization')
print(token_ids_sp)
print(token_str_sp)
print(tok.decode(token_ids_sp))

assert token_ids_hf == token_ids_sp, 'Prompt tokens are different'
print('\n\n***All tests passed***\n\n')
#%%


# %%[markdown]
# ### Comparison with our Conversation class
# Make sure we're getting the same tokens and strings from the Conversation class formatting
# %%
import conversation
conv_mistral = conversation.get_conv_template('mistral')
print(conv_mistral)
conv_mistral.append_messages(messages)
print(conv_mistral.get_prompt())
conv_mistral_token_ids = hf_tok.encode(conv_mistral.get_prompt())
conv_mistral_token_str = hf_tok.convert_ids_to_tokens(conv_mistral_token_ids)
print(mistral_ref_token_str)
print(conv_mistral_token_str)
print(hf_tok.decode(mistral_ref_token_ids, skip_special_tokens=True))
print(hf_tok.decode(conv_mistral_token_ids, skip_special_tokens=True))
# %%



#################################################################################
# %%[markdown] ##################################################################

# ## Llama LLamaPrompt class testing

# %%
tokenizer = tokenize_llama_utils.Tokenizer(model_path="models/meta-llama/Llama-2-7b-chat/tokenizer.model")
print(tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id)
print(tokenizer.decode([tokenizer.bos_id, tokenizer.eos_id,]))

llama_prompt = tokenize_llama_utils.LlamaPrompt
prompt_tokens = llama_prompt.encode_instruct(messages, tokenizer)
print('MetaAI implementation sentencpiece tokenizer')
print(prompt_tokens)
print(tokenizer.decode(prompt_tokens))

# %%[markdown]
# Quick test to show that sentencepiece shouldn't be given special tags (bos/eos) in strings.
# This is what HF seems to do to construct a complete string with multiple messages, but their
# implementation/tokenizer handles it correctly
# %%
text = "<s>[INST] 2+2 [/INST] 4! </s><s>[INST] +2 [/INST] 6! </s><s>[INST] +4 [/INST] "
prompt_tokens = tokenizer.encode(text, bos=False, eos=False)
print(prompt_tokens)
print('HF implementation tokenizer and chat template from raw string')
print(f'Input text:\n{text}\n')
print(f'Output text:\n{tokenizer.decode(prompt_tokens)}')

# %%[markdown]
# Now, use the HF tokenization and chat_template, which gives the same result as sentencepiece with
# encoding for each message or pair of messages
# %%
hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
print(hf_tokenizer.bos_token_id, hf_tokenizer.eos_token_id, hf_tokenizer.pad_token_id)
print('HF implementation tokenizer and chat template from messages list of dicts')

print(hf_tokenizer.decode([hf_tokenizer.bos_token_id, hf_tokenizer.eos_token_id,]))
hf_prompt_tokens = hf_tokenizer.apply_chat_template(messages)
print(hf_prompt_tokens)
print(hf_tokenizer.decode(hf_prompt_tokens, skip_special_tokens=True))

# %%[markdown]
# ### Direct comparison on tokenization and decoding
# %%
print('Compare tokenized prompts and decoded strings token by token\n')

prompt_tokens = LlamaPrompt.encode_instruct(messages, tokenizer)
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


# %%








#################################################################################
# %%
hf_tokens = hf_tok.tokenize('hello world, abc', add_special_tokens=True, )
hf_ids = hf_tok.convert_tokens_to_ids(hf_tokens)
hf_tokens2 = hf_tok.convert_ids_to_tokens(hf_ids)
print(hf_tokens, hf_ids, hf_tokens2)

#  %%
mistral_ref_token_ids, mistral_ref_token_str = build_prompt(messages, hf_tok)
print(mistral_ref_token_ids)
print(mistral_ref_token_str)
print(hf_tok.decode(mistral_ref_token_ids, skip_special_tokens=True))
print(hf_tok.decode(mistral_ref_token_ids, skip_special_tokens=False))

