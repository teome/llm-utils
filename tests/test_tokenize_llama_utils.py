from typing import List
import pytest
from transformers import AutoTokenizer

from llm_utils.tokenize_llama_utils import LlamaPrompt, Tokenizer

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
