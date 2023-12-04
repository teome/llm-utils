# %%
import json
import requests
import openai

# %%

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


# %%
# %%[markdown]
# Helper message class, with and without pydantic

# Light version without pydantic
from typing import NamedTuple
from pydantic import BaseModel

class MessageNT(NamedTuple):
    role: str
    content: str

# More robust for further additions version with pydantic
class MessagePydantic(BaseModel):
    role: str
    content: str

Message = MessagePydantic

# %%[markdown]
# ## Conversation class to wrap up a list of messages for interactiona
class Conversation(list):
    def __init__(self, messages=None, color_scheme=None):
        super().__init__(messages or [])
        self.color_scheme = color_scheme or {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "cyan",
        }

    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.append(Message(role=role, content=content))

    def display_conversation(self, detailed=False):
        """Display the conversation."""
        for message in self:
            terminal_color = terminal_colors[self.color_scheme.get(message.role, "white")]
            print(f"{terminal_color}{message.role}: {message.content}{terminal_colors['endc']}")

    def delete_interaction(self, index=-1):
        """Delete an interaction from the conversation."""
        if index == -1:
            index = len(self) - 2

        if index < 0 or index >= len(self) or self[index].role != 'user':
            raise RuntimeError("Invalid index for deletion. Index does not correspond to a user message.")

        if index + 1 >= len(self) or self[index + 1].role != 'assistant':
            raise RuntimeError("Invalid interaction. No assistant message following the user message.")

        del self[index:index + 2]


# %%
# %%
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
messages = [Message(**x) for x in messages]
print(messages)

# %%
# print(messages)
# messages = [Message(role="system", content="You are a helpful assistant."),
#             Message(role="user", content="Who were the members of drexciya?")]
conversation = Conversation(messages=messages)
conversation.display_conversation()

# %%
