"""
Conversation prompt templates.

Heavily inspired and partly copied from fastchat/vLLM approach
https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py

TODO consider just using the library if more of its functionality is needed

Mistral-Instruct format
<s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
As reference, here is the format used to tokenize instructions during fine-tuning:
 [START_SYMBOL_ID] +
 tok("[INST]") + tok(USER_MESSAGE_1) + tok("[/INST]") +
 tok(BOT_MESSAGE_1) + [END_SYMBOL_ID] +
 …
 tok("[INST]") + tok(USER_MESSAGE_N) + tok("[/INST]") +
 tok(BOT_MESSAGE_N) + [END_SYMBOL_ID]
NOTE The function tok should never generate the EOS token, however FastChat (used in vLLM)
sends the full prompt as a string which might lead to incorrect tokenization of the EOS token
and prompt injection. Users are encouraged to send tokens instead as described above.
NOTE Maybe just what everyone else does and encode the special tokens in the prompt string. HF and vLLM do this

NB need to add bos and eos tokens from tokenizer, not encoded in the prompt string then tokenized
<s>[INST] <<SYS>> <system prompt> <</SYS>> <user prompt> [/INST] <response> </s>
TODO only add </s> as a token directly after a assistant response, not after a user prompt,
AND use the token not a string appended (according to docs and see huggingface templating which
also uses the token not adding a string)
https://huggingface.co/docs/transformers/main/chat_templating#special-variables
https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212

{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
    {% elif message['role'] == 'system' %}
        {{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' '  + message['content'] + ' ' + eos_token }}
    {% endif %}
{% endfor %}
"""

from dataclasses import dataclass, field
from typing import List, Any, Dict, Union, Tuple


class _BColors:
    """Color codes for terminal output formatting"""

    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    DEFAULT = "\033[39m"
    WARNING = "\033[39m"
    HEADER = "\033[95m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ENDC = "\033[0m"


_terminal_colors = {
    "red": _BColors.RED,
    "yellow": _BColors.YELLOW,
    "green": _BColors.GREEN,
    "blue": _BColors.BLUE,
    "cyan": _BColors.CYAN,
    "white": _BColors.WHITE,
    "default": _BColors.DEFAULT,
    "warning": _BColors.WARNING,
    "fail": _BColors.RED,
    "header": _BColors.HEADER,
    "bold": _BColors.BOLD,
    "underline": _BColors.UNDERLINE,
    "endc": _BColors.ENDC,
}


@dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # # The template of the system prompt
    # system_template: str = "{system_message}"
    # # The system message
    # system_message: str = ""
    # The names of roles
    roles: Tuple[str,...] = ("system", "user", "assistant")
    roles_templates: Dict[str, str] = field(default_factory=dict)
    # The role to use for adding a final empty content prompt to the end of the conversation
    # to get the generator to begin generating. Most likely to be the assistant role
    generator_str: str = ""
    # Whether to include the system prompt in the user prompt (e.g. llama-2 chat etc.)
    system_in_user: bool = False
    # All messages. Each item is at least {"role": role, "content": content, ...}.
    # More fields can be added for different purposes.
    messages: List[Dict[str, str]] = ()
    # The number of few shot examples
    offset: int = 0
    # # The separator style and configurations
    # sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    # sep: str = "\n"
    # sep2: str = None
    # Some models (e.g. Mistral, deviate from the standard format and have just a start token
    # for all following messages, and the user message doesn't have one. This is different from
    # the model it derived from llama-2 which has a start token for each user message. This is a
    # hack to get around that confusing implementation. Why can't people just settle on a format!)
    start_str: str = ""
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self, add_generator_prompt=True) -> str:
        """Get the prompt for the conversation."""
        ret = "" + self.start_str
        system_content = ""
        for i, message in enumerate(self.messages):
            if message["role"] == "system" and self.system_in_user:
                if self.messages[i+1]["role"] != "user":
                    raise ValueError(
                        "The system prompt must be followed by a user prompt when `system_in_user=True` for prompt.")
                system_content = self.roles_templates["system"].format(message["content"])
                continue

            content = ""
            if system_content != "":
                content = system_content
                system_content = ""
            content += message["content"]

            ret += self.roles_templates[message["role"]].format(content.strip())


        # Add a final empty content prompt to the end of the conversation.
        # Skip if the last message is already empty, regaredless of the role
        if add_generator_prompt and self.messages[-1]["content"] != "":
            ret += self.generator_str

        return ret

    def append_message(self, message):
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise ValueError("Message must be a dict with at least `role` and `content`.")
        self.messages.append(message)

    def append_messages(self, messages: List[Dict[str, str]]):
        self.messages.extend(messages)

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""

        # not much to do as we're already using the same dict format, but use the offset
        return self.messages[self.offset:]

    def copy(self):
        return Conversation(
            name=self.name,
            # system_template=self.system_template,
            # system_message=self.system_message,
            roles=self.roles,
            roles_templates=self.roles_templates.copy(),
            generator_str=self.generator_str,
            system_in_user=self.system_in_user,
            messages=[message.copy() for message in self.messages],
            offset=self.offset,
            # sep_style=self.sep_style,
            # sep=self.sep,
            # sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def format_message(self, message, detailed=True, color_scheme=None):
        """Format the message with color based on the role."""
        if color_scheme is None:
            color_scheme = {}
        terminal_color = _terminal_colors[color_scheme.get(message["role"], "white")]

        if detailed:
            non_role_strs = ' |'.join([f"{k}: {v}" for k, v in message.items() if k != 'role'])
            return f"{terminal_color}role: {message['role']}{_terminal_colors['endc']} | {non_role_strs}"
        return f"{terminal_color}{message['role']}:{_terminal_colors['endc']} {message['content']}"

    def display_conversation(self, detailed=False, color_scheme=None):
        """Display the conversation."""
        if color_scheme is None:
            color_scheme = {
                "system": "yellow",
                "user": "default",
                "assistant": "blue",
                "function": "cyan",
            }

        for message in self.messages:
            formatted_message = self.format_message(message, detailed=detailed, color_scheme=color_scheme)
            print(formatted_message)


# a global registry of conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template

def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()

register_conv_template(
    Conversation(
        name="raw",
        roles=("system", "user", "assistant"),
        roles_templates={
            "system": "{system_message}\n",
            "user": "{user_message}\n",
            "assistant": "{assistant_message}\n",
        },
        generator_str="",
        system_in_user=False,
        messages=[],
        offset=0,
        stop_str=None,
        stop_token_ids=None,
    )
)

register_conv_template(
    Conversation(
        name="llama-2",
        roles=("system", "user", "assistant"),
        roles_templates={
            "system": "<<SYS>>\n{}\n<</SYS>>\n\n",
            "user": "<s>[INST] {} [/INST]",
            "assistant": " {} </s>",
        },
        generator_str=" ",
        system_in_user=True,
        messages=[],
        offset=0,
        stop_str="</s>",
        stop_token_ids=None,
    )
)

register_conv_template(
    Conversation(
        name="mistral",
        roles=("system", "user", "assistant"),
        roles_templates={
            "system": "<<SYS>>\n{}\n<</SYS>>\n\n",
            "user": "[INST] {} [/INST]",
            "assistant": " {} </s>",
        },
        generator_str=" ",
        system_in_user=True,
        messages=[],
        offset=0,
        start_str="<s>",
        stop_str="</s>",
        stop_token_ids=None,
    )
)


# Standard chatml format
# """
# <|im_start|>system
# You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.
# Knowledge cutoff: 2021-09-01
# Current date: 2023-03-01<|im_end|>
# <|im_start|>user
# How are you<|im_end|>
# <|im_start|>assistant
# I am doing well!<|im_end|>
# <|im_start|>user
# How are you now?<|im_end|>
# """
register_conv_template(
    Conversation(
        name="chatml",
        roles=("system", "user", "assistant"),
        roles_templates={
            "system": "<|im_start|>system\n{}<|im_end|>\n",
            "user": "<|im_start|>user\n{}<|im_end|>\n",
            "assistant": "<|im_start|>assistant\n{}<|im_end|>\n",
        },
        generator_str="<|im_start|>assistant\n",
        system_in_user=False,
        messages=[],
        offset=0,
        stop_str="<|im_end|>",
        stop_token_ids=None, # set this for the tokenizer to use or create another template with it set
    )
)
