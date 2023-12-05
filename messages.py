import json
from typing import NamedTuple
from typing import TypedDict, Required, Literal
from pydantic import BaseModel


class BColors:
    HEADER = '\033[95m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

terminal_colors = {
    "red": BColors.RED,
    "blue": BColors.BLUE,
    "cyan": BColors.CYAN,
    "green": BColors.GREEN,
    "warning": BColors.WARNING,
    "fail": BColors.FAIL,
    "endc": BColors.ENDC,
    "bold": BColors.BOLD,
    "underline": BColors.UNDERLINE,
}


# Helper message class, with and without pydantic

# Light version without pydantic but named tuple
class MessageNT(NamedTuple):
    role: str
    content: str

# More robust for further additions version with pydantic
class MessagePydantic(BaseModel):
    role: str
    content: str


class MessageTD(TypedDict, total=False):
    role: Required[str]
    content: Required[str]

Message = MessageTD

# Conversation class to wrap up a list of messages for interaction
# TODO: modify to incorporate the approach taken here: 
# https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L36
class Conversation:
    def __init__(self, messages=None, color_scheme=None):
        self.messages = messages or []
        self.color_scheme = color_scheme or {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "cyan",
        }
        self.roles = ("user", "assistant",)

    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append(Message(role=role, content=content))

    def display_conversation(self, detailed=False):
        """Display the conversation."""
        for message in self.messages:
            terminal_color = terminal_colors[self.color_scheme.get(message["role"], "white")]
            if detailed:
                non_role_strs = ' |'.join([f"{k}: {v}" for k, v in message.items() if k != 'role'])
                print(f"{terminal_color}{message['role']}: {non_role_strs}{terminal_colors['endc']}")
            else:
                print(f"{terminal_color}{message['role']}: {message['content']}{terminal_colors['endc']}")

    def delete_interaction(self, index=-1):
        """Delete an interaction from the conversation."""
        if index == -1:
            index = len(self.messages) - 2

        if index < 0 or index >= len(self.messages) or self.messages[index]["role"] != 'user':
            raise RuntimeError("Invalid index for deletion. Index does not correspond to a user message.")

        if index + 1 >= len(self.messages) or self.messages[index + 1]["role"] != 'assistant':
            raise RuntimeError("Invalid interaction. No assistant message following the user message.")

        del self.messages[index:index + 2]

    def json(self):
        return json.dumps(self.messages)
    
    def get_prompt(self):
        raise NotImplementedError


if __name__ == "__main__":
    messages=[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Who won the FA Cup in 2022?"
    }, {
        "role": "assistant",
        "content": "Liverpool won the FA Cup in 2022."
    },
    ]

    conversation = Conversation(messages=messages)
    conversation.display_conversation()
    conversation.display_conversation(detailed=True)
    print(conversation.json())
    print(conversation.messages)

    conversation.delete_interaction()
    conversation.display_conversation()
    conversation.add_message(role="user", content="Who won the FA Cup in 2022?")
    conversation.add_message(role="assistant", content="Liverpool won the FA Cup in 2022.")
    conversation.delete_interaction()
    conversation.display_conversation()

    conversation.messages.append({"role": "user", "content": "Who won the FA Cup in 2022?"})
    conversation.messages.append({"role": "assistant", "content": "Liverpool won the FA Cup in 2022."})
    conversation.display_conversation(detailed=True)

    