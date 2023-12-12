import pytest
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

    prompt_tokens = LlamaPrompt.chat_completion(dialogs, tokenizer)
    assert isinstance(prompt_tokens, list)
    assert len(prompt_tokens) == 2
    assert all(isinstance(tokens, list) for tokens in prompt_tokens)
