from typing import List
import pytest

import tokenize_llama_utils
from tokenize_llama_utils import Tokenizer, LlamaPrompt

@pytest.fixture
def tokenizer():
    return Tokenizer(model_path="models/meta-llama/Llama-2-7b-chat/tokenizer.model")


from typing import List, Dict
import pytest
from transformers import AutoTokenizer

from tokenize_llama_utils import LlamaPrompt, Tokenizer

@pytest.fixture
def tokenizer_sentencepiece():
    return Tokenizer(model_path="models/meta-llama/Llama-2-7b-chat/tokenizer.model")


@pytest.fixture
def tokenizer_hf():
    return AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

messages = [
    {"role": "user", "content": "2+2"},
    {"role": "assistant", "content": "4!"},
    {"role": "user", "content": "+2"},
    {"role": "assistant", "content": "6!"},
    {"role": "user", "content": "+4"},
]
# TODO: update this with new expected tokens.
# Currently not an issue as there's no difference between the two tokenizers so we can just compare decoded strings
# expected_tok_ids = [1, 733, 16289, 28793, 28705, 28750, 28806, 28750, 733, 28748, 16289, 28793, 28705, 28781, 28808, 2, 733, 16289, 28793, 648, 28750, 733, 28748, 16289, 28793, 28705, 28784, 28808, 2, 733, 16289, 28793, 648, 28781, 733, 28748, 16289, 28793]
# expected_tok_ids_to_tokens = ['<s>', '▁[', 'INST', ']', '▁', '2', '+', '2', '▁[', '/', 'INST', ']', '▁', '4', '!', '</s>', '▁[', 'INST', ']', '▁+', '2', '▁[', '/', 'INST', ']', '▁', '6', '!', '</s>', '▁[', 'INST', ']', '▁+', '4', '▁[', '/', 'INST', ']']


def test_encode_instruct(tokenizer_hf, tokenizer_sentencepiece):
    encoded_hf = LlamaPrompt.encode_instruct_hf(messages, tokenizer_hf)
    encoded_sentencepiece = LlamaPrompt.encode_instruct_llama(messages, tokenizer_sentencepiece)
    assert encoded_hf == encoded_sentencepiece
    assert tokenizer_hf.convert_ids_to_tokens(encoded_hf) == tokenizer_sentencepiece.sp_model.id_to_piece(encoded_sentencepiece)
    assert tokenizer_hf.decode(encoded_hf, skip_special_tokens=True) == tokenizer_sentencepiece.decode(encoded_sentencepiece)


def test_encode_instruct_system_tags(tokenizer_hf, tokenizer_sentencepiece):
    messages_with_system = [{"role": "system", "content": "Hello"}] + messages
    encoded_hf = LlamaPrompt.encode_instruct_hf(messages_with_system, tokenizer_hf)
    encoded_sentencepiece = LlamaPrompt.encode_instruct_llama(messages_with_system, tokenizer_sentencepiece)
    assert encoded_hf == encoded_sentencepiece
    assert tokenizer_hf.convert_ids_to_tokens(encoded_hf) == tokenizer_sentencepiece.sp_model.id_to_piece(encoded_sentencepiece)
    assert tokenizer_hf.decode(encoded_hf, skip_special_tokens=True) == tokenizer_sentencepiece.decode(encoded_sentencepiece)
    assert encoded_hf == encoded_sentencepiece








# def test_tokenizer_encode(tokenizer):
#     s = "Hello, world!"
#     bos = True
#     eos = True
#     encoded = tokenizer.encode(s, bos, eos)
#     assert isinstance(encoded, list)
#     assert len(encoded) > 0

# def test_tokenizer_decode(tokenizer):
#     t = [1, 2, 3, 4]
#     decoded = tokenizer.decode(t)
#     assert isinstance(decoded, str)
#     assert len(decoded) > 0

# def test_llama_prompt_chat_completion(tokenizer):
#     dialogs = [
#         [
#             {"role": "user", "content": "Hello"},
#             {"role": "assistant", "content": "Hi there!"},
#             {"role": "user", "content": "How are you?"},
#         ],
#         [
#             {"role": "system", "content": "You are a helpful AI."},
#             {"role": "user", "content": "How are you?"},
#         ],
#     ]

#     for dialog in dialogs:
#         prompt_tokens = LlamaPrompt.chat_completion(dialog, tokenizer)
#         assert isinstance(prompt_tokens, list)
#         assert all(isinstance(tokens, list) for tokens in prompt_tokens)


# def test_llama_tokenization_hf_chat_template(tokenizer):
#     # copied from interactive_test_misc.py
#     from transformers import AutoTokenizer

#     hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

#     messages: List[tokenize_llama_utils.Message] = [
#         {"role": "user", "content": "What is your favourite condiment?"},
#         {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#         {"role": "user", "content": "Do you have mayonnaise recipes?"},]

#     print('Compare tokenized prompts and decoded strings token by token\n')

#     prompt_tokens = LlamaPrompt.chat_completion(messages, tokenizer)
#     print('MetaAI implementation sentencpiece tokenizer from messages list of dicts')
#     print(prompt_tokens)
#     decoded_prompt = tokenizer.decode(prompt_tokens)
#     print(decoded_prompt)

#     hf_prompt_tokens = hf_tokenizer.apply_chat_template(messages)
#     print('\n\nHF tokenizer with chat template from messages list of dicts')
#     print(hf_prompt_tokens)
#     hf_decoded_prompt = hf_tokenizer.decode(hf_prompt_tokens, skip_special_tokens=True)
#     print(hf_decoded_prompt)

#     print('\n\nTest equivalence of prompt tokens and decoded strings')
#     if prompt_tokens == hf_prompt_tokens:
#         print('Prompt tokens are the same')
#     else:
#         print('Prompt tokens are different')
#         assert prompt_tokens == hf_prompt_tokens, 'Prompt tokens are different'

#     if decoded_prompt == hf_decoded_prompt:
#         print('Decoded prompts are the same')
#     else:
#         print('\n\nDecoded prompts are different')
#         print(f'MetaAI decoded prompt:\n{decoded_prompt}')
#         print(f'HF decoded prompt:\n{hf_decoded_prompt}')
#         assert decoded_prompt == hf_decoded_prompt, 'Decoded prompts are different'
