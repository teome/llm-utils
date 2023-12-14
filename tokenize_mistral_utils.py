from typing import Callable, List
from tokenize_llama_utils import Tokenizer, LlamaPrompt, Role, Message, Dialog

class MistralPrompt(LlamaPrompt):
    @classmethod
    def chat_completion(
        cls,
        dialogs: List[Dialog],
        tokenizer: Callable,
        include_system_tags: bool = False,
    ) -> List[int]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Can be called as a static method, but the tokenizer must be passed in that case

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            tokenizer (Optional[Callable]): Tokenizer to use for encoding the prompt. If None, the class must be
                instantiated with a tokenizer or tokenizer_path.
            include_system_tags (bool): Whether to include the system tags in the prompt. Default False given that
                the reference prompts from Mistral don't use them.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates chat assistant prompts from a list of dialogs. Each dialog is a list of messages,
                that can start with a system message, followed by a user message, and alternating between user and
                assistant messages. The last message in a dialog must be from the user
            It is only slightly different from the Llama prompt in that the BOS tokens are only added to the start of
                the first message, whereas the EOS are added to the end of each assistant response message. Why did
                they do this?
        """
        # could just reference them via the class, but redefine here for clarity and to keep the rest of the code the same
        B_INST = cls.B_INST
        E_INST = cls.E_INST
        SPECIAL_TAGS = cls.SPECIAL_TAGS
        UNSAFE_ERROR = cls.UNSAFE_ERROR

        b_sys = cls.B_SYS if include_system_tags else ""
        e_sys = cls.E_SYS if include_system_tags else "\n\n"

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
                        "content": b_sys
                        + dialog[0]["content"]
                        + e_sys
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = [tokenizer.bos_id]
            dialog_tokens += sum(
                [
                    tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=False,
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
                bos=False,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)
        return prompt_tokens


