# Tokenizer class from
# https://github.com/mistralai/mistral-src/blob/main/mistral/tokenizer.py
# MistralPrompt class modified from
# https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L284
# Note that Mistral's tokenization is similar but different in important ways
# from Llama. There's no explicit <<SYS>> tag but the option of using it here
# is included for compatibility with Llama. Otherwise, just put system in the
# user message. There's also whitespace gotchas to watch out for.

from pathlib import Path
from typing import Any, Dict, List, Union

from sentencepiece import SentencePieceProcessor

from tokenize_llama_utils import LlamaPrompt, Message


class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def n_words(self) -> int:
        return self._model.vocab_size()

    @property
    def bos_id(self) -> int:
        return self._model.bos_id()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
        assert isinstance(s, str)
        t = self._model.encode(s)
        if bos:
            t = [self.bos_id, *t]
        if eos:
            t = [*t, self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self._model.decode(t)


class MistralPrompt(LlamaPrompt):
    @classmethod
    def encode_instruct(
        cls,
        messages: List[Message],
        tokenizer: Union[Tokenizer, Any],
        include_system_tags: bool = False,
    ) -> List[int]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Can be called as a static method, but the tokenizer must be passed in that case

        Args:
            messages (List[Message]): List of conversational messages, where each message is a (typed) dict of
                `{'role': role, 'content': content}`.
            tokenizer (Union[Tokenizer, Any]): Tokenizer to use for encoding the prompt. Can be either the Tokenizer
                class from this module (based on SentencePiece) or an AutoTokenizer from HuggingFace for the model.
            include_system_tags (bool): Whether to include the system tags in the prompt. Default False given that
                the reference prompts from Mistral don't use them.

        Returns:
            List[int]: List of encoded prompt tokens.

        Raises:
            AssertionError: If the last message in the messages is not from the user.
            AssertionError: If the message roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates tokens based on the instruct template from Mistral. Each list of messages is can start
            with a system message, followed by a user message, and then alternating user and assistant messages. The
            system message is optional, but if it is included, it must be the first message in the list. The last
            message must be from the user. The messages are then encoded into tokens using the instruct template.

            It is only slightly different from the Llama prompt in that the BOS tokens are only added to the start of
            the first message, whereas the EOS are added to the end of each assistant response message. Why did
            they do this?

            Depending on the tokenizer class passed in, subtly different logic has to be used to deal with encoding
            differences. Handling of the resulting tokens by the model is the same regardless of using HF or
            reference PyTorch implementation
        """
        # could just reference them via the class, but redefine here for clarity and to keep the rest of the code the same
        B_INST = cls.B_INST
        E_INST = cls.E_INST
        SPECIAL_TAGS = cls.SPECIAL_TAGS
        UNSAFE_ERROR = cls.UNSAFE_ERROR

        b_sys = cls.B_SYS if include_system_tags else ""
        e_sys = cls.E_SYS if include_system_tags else "\n\n"

        if isinstance(tokenizer, Tokenizer):
            return cls.encode_mistral(messages, tokenizer, include_system_tags=include_system_tags)
        return cls.encode_hf(messages, tokenizer, include_system_tags=include_system_tags)

    @classmethod
    def merge_system_user_messages(cls, messages: List[Message], include_system_tags: bool = False):
        """
        Merge system and user messages into a single message.

        Args:
            messages (List[Message]): List of messages to merge.
            include_system_tags (bool): Whether to include the system tags in the prompt. Default False given that
                the reference prompts from Mistral don't use them. Tags are stored as class members
                `B_SYS`, `E_SYS` but could be overridden. Although Mistral doesn't specifiy any, we use the Llama
                defaults `<<SYS>>` and `<<SYS>>` for compatibility.

        Returns:
            List[Message]: List of merged messages.
        """
        assert messages[0]["role"] == "system", "First message must be from system, otherwise there's nothing to do"
        b_sys = cls.B_SYS if include_system_tags else ""
        e_sys = cls.E_SYS if include_system_tags else "\n\n"

        if messages[0]["role"] == "system":
            messages = [
                Message({
                    "role": messages[1]["role"],
                    "content": b_sys
                    + messages[0]["content"]
                    + e_sys
                    + messages[1]["content"],
                })
            ] + messages[2:]
        return messages

    @classmethod
    def encode_instruct_mistral(cls, messages: List[Message], tokenizer: Tokenizer, include_system_tags: bool = False):
        """
        Encode a dialog into a prompt for Mistral.

        Args:
            dialog (List[Message]): Dialog to encode.
            tokenizer (Tokenizer): Tokenizer to use for encoding. Expected to be the Mistral tokenizer which itself
                is based on SentencePiece.
            include_system_tags (bool): Whether to include the system tags in the prompt. Default False given that
                the reference prompts from Mistral don't use them. Tags are stored as class members
                `B_SYS`, `E_SYS` but could be overridden. Although Mistral doesn't specifiy any, we use the Llama
                defaults `<<SYS>>` and `<<SYS>>` for compatibility.


        Returns:
            List[int]: Encoded prompt tokens.
        """
        if messages[0]["role"] == "system":
            messages = cls.merge_system_user_messages(messages, include_system_tags=include_system_tags)

        prompt_toks = [tokenizer.bos_id]
        for i, msg in enumerate(messages):
            is_user = {"user": True, "assistant": False}[msg["role"]]
            assert (i % 2 == 0) == is_user
            content = msg["content"]
            assert content == content.strip()

            if is_user:
                prompt_toks += tokenizer.encode(f"[INST] {content} [/INST]", bos=False, eos=False)
            else:
                prompt_toks += tokenizer.encode(f"{content}", bos=False, eos=True)

        return prompt_toks

    @classmethod
    def encode_instruct_hf(cls, messages: List[Message], tokenizer, include_system_tags: bool = False):
        """
        Encode a list of role/content messages to tokens using a HuggingFace tokenizer instantiated
        for the desired Mistral (or compatible) model.

        Args:
            messages (List[Message]): Dialog to encode.
            tokenizer: Tokenizer to use for encoding. Tokenizer must be instance of HF AutoTokenizer.
            include_system_tags (bool): Whether to include the system tags in the prompt. Default False given that
                the reference prompts from Mistral don't use them. Tags are stored as class members
                `B_SYS`, `E_SYS` but could be overridden. Although Mistral doesn't specifiy any, we use the Llama
                defaults `<<SYS>>` and `<<SYS>>` for compatibility.

        Returns:
            List[int]: Encoded prompt tokens.
        """
        if messages[0]["role"] == "system":
            messages = cls.merge_system_user_messages(messages, include_system_tags=include_system_tags)

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
        return tokens_ids

# Code given by Mistral team during discussion of the formatting issues with OSS libs
def _mistral_team_ref_build_prompt(
    messages: List[Dict[str, str]],
    tokenizer,
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