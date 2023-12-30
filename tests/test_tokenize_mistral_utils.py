import pytest
from transformers import AutoTokenizer

from llm_utils.tokenize_mistral_utils import MistralPrompt, Tokenizer

@pytest.fixture
def tokenizer_sentencepiece():
    return Tokenizer(model_path="models/mistralai/Mixtral-8x7B-Instruct-v0.1/tokenizer.model")


@pytest.fixture
def tokenizer_hf():
    return AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

messages = [
    {"role": "user", "content": "2+2"},
    {"role": "assistant", "content": "4!"},
    {"role": "user", "content": "+2"},
    {"role": "assistant", "content": "6!"},
    {"role": "user", "content": "+4"},
]
expected_tok_ids = [1, 733, 16289, 28793, 28705, 28750, 28806, 28750, 733, 28748, 16289, 28793, 28705, 28781, 28808, 2, 733, 16289, 28793, 648, 28750, 733, 28748, 16289, 28793, 28705, 28784, 28808, 2, 733, 16289, 28793, 648, 28781, 733, 28748, 16289, 28793]
expected_tok_ids_to_tokens = ['<s>', '▁[', 'INST', ']', '▁', '2', '+', '2', '▁[', '/', 'INST', ']', '▁', '4', '!', '</s>', '▁[', 'INST', ']', '▁+', '2', '▁[', '/', 'INST', ']', '▁', '6', '!', '</s>', '▁[', 'INST', ']', '▁+', '4', '▁[', '/', 'INST', ']']

def test_encode_instruct_hf(tokenizer_hf):
    encoded = MistralPrompt.encode_instruct_hf(messages, tokenizer_hf)
    assert encoded == expected_tok_ids
    assert tokenizer_hf.convert_ids_to_tokens(encoded) == expected_tok_ids_to_tokens

def test_encode_instruct_sentencepiece(tokenizer_sentencepiece):
    encoded = MistralPrompt.encode_instruct_mistral(messages, tokenizer_sentencepiece)
    assert encoded == expected_tok_ids
    assert tokenizer_sentencepiece._model.id_to_piece(encoded) == expected_tok_ids_to_tokens

def test_encode_instruct_system_tags(tokenizer_hf, tokenizer_sentencepiece):
    messages_with_system = [{"role": "system", "content": "Hello"}] + messages
    encoded_hf = MistralPrompt.encode_instruct_hf(messages_with_system, tokenizer_hf, include_system_tags=True)
    encoded_sentencepiece = MistralPrompt.encode_instruct_mistral(messages_with_system, tokenizer_sentencepiece, include_system_tags=True)
    assert encoded_hf == encoded_sentencepiece
