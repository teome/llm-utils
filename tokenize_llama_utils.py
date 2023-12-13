"""Functions to tokenize strings, in particular model formats to avoid issues with raw strings and special characters."""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
from logging import getLogger
from typing import Callable, Dict, List, Literal, Optional, TypedDict

from sentencepiece import SentencePieceProcessor

# largely copied and adapted from MetaAI llama repo and Mistral docs.
# Avoid the need for transformers and the llama repo given that it's just sentencepiece
# and this avoids any issues with bad formatting of the prompts that's common
# https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L284



# def tokenize_messages(messages: List[str], tokenize_func: Optional[Callable] = None, tokenizer: Optional[Callable] = None) -> List[int]:
#     if tokenize_func is None and tokenizer is None:
#         raise ValueError("Either tokenize_func or tokenizer must be provided.")
#         tokenize_func = tokenize_string



logger = getLogger()


class Tokenizer:
    """tokenizing and encoding/decoding text using SentencePiece."""
    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a SentencePiece model.

        Args:
            model_path (str): The path to the SentencePiece model file.
        """
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}") # pylint: disable=1203:logging-fstring-interpolation

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )  # pylint: disable=1203:logging-fstring-interpolation
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.sp_model.decode(t)



Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]


class LlamaPrompt:
    B_INST = "[INST]"
    E_INST = "[/INST]"
    B_SYS = "<<SYS>>\n"
    E_SYS = "\n<</SYS>>\n\n"
    SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
    UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

    @classmethod
    def chat_completion(
        cls,
        dialogs: List[Dialog],
        tokenizer: Callable,
    ) -> List[int]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Can be called as a static method, but the tokenizer must be passed in that case

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            tokenizer (Optional[Callable]): Tokenizer to use for encoding the prompt. If None, the class must be
                instantiated with a tokenizer or tokenizer_path.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates chat assistant prompts from a list of dialogs. Each dialog is a list of messages,
                that can start with a system message, followed by a user message, and alternating between user and
                assistant messages. The last message in a dialog must be from the user
        """
        # could just reference them via the class, but redefine here for clarity and to keep the rest of the code the same
        B_INST = cls.B_INST
        E_INST = cls.E_INST
        B_SYS = cls.B_SYS
        E_SYS = cls.E_SYS
        SPECIAL_TAGS = cls.SPECIAL_TAGS
        UNSAFE_ERROR = cls.UNSAFE_ERROR

        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)
        return prompt_tokens


