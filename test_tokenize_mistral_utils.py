from typing import List, Dict
import pytest

from tokenize_mistral_utils import MistralPrompt, Tokenizer

@pytest.fixture
def tokenizer():
    return Tokenizer(model_path="models/mistralai/Mixtral-8x7B-Instruct-v0.1/tokenizer.model")

@pytest.fixture
def hf_tokenizer():
    from transformers import AutoTokenizer
    # return AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    return AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

def test_enc_user_assistant(tokenizer, hf_tokenizer):
    messages: List[Dict] = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"},]

    prompt_tokens = MistralPrompt.chat_completion([messages], tokenizer)[0]
    hf_prompt_tokens = hf_tokenizer.apply_chat_template(messages)
    assert prompt_tokens == hf_prompt_tokens, "Encoded rompt tokens are different"

def test_enc_dec_user_assistant(tokenizer):
    messages: List[Dict] = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"},]

    expected_prompt = "[INST] What is your favourite condiment? [/INST] Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!  [INST] Do you have mayonnaise recipes? [/INST]"
    prompt_tokens = MistralPrompt.chat_completion([messages], tokenizer)[0]
    decoded_prompt = tokenizer.decode(prompt_tokens)
    assert decoded_prompt == expected_prompt, "Decoded prompt strings are different"