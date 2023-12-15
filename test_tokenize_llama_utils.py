from typing import List
import pytest

import tokenize_llama_utils
from tokenize_llama_utils import Tokenizer, LlamaPrompt

@pytest.fixture
def tokenizer():
    return Tokenizer(model_path="models/meta-llama/Llama-2-7b-chat/tokenizer.model")

def test_tokenizer_encode(tokenizer):
    s = "Hello, world!"
    bos = True
    eos = True
    encoded = tokenizer.encode(s, bos, eos)
    assert isinstance(encoded, list)
    assert len(encoded) > 0

def test_tokenizer_decode(tokenizer):
    t = [1, 2, 3, 4]
    decoded = tokenizer.decode(t)
    assert isinstance(decoded, str)
    assert len(decoded) > 0

def test_llama_prompt_chat_completion(tokenizer):
    dialogs = [
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ],
        [
            {"role": "system", "content": "You are a helpful AI."},
            {"role": "user", "content": "How are you?"},
        ],
    ]

    for dialog in dialogs:
        prompt_tokens = LlamaPrompt.chat_completion(dialog, tokenizer)
        assert isinstance(prompt_tokens, list)
        assert all(isinstance(tokens, list) for tokens in prompt_tokens)


def test_llama_tokenization_hf_chat_template(tokenizer):
    # copied from interactive_test_misc.py
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    messages: List[tokenize_llama_utils.Message] = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"},]

    print('Compare tokenized prompts and decoded strings token by token\n')

    prompt_tokens = LlamaPrompt.chat_completion(messages, tokenizer)
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
