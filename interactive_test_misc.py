# %%
import conversation
from endpoint_utils import OpenAI, openai_chat_completions_create, openai_extract_chat_completion_message


initial_messages = [
    {"role": "system", "content": "You are Samantha, a helpful, friendly, funny and highly knowledgeable assistant."},
    {"role": "user", "content": "What is the history of the Metalheadz label?"}
]
conv = conversation.get_conv_template("chatml")
conv.append_messages(messages=initial_messages)
conv.display_conversation()
print(conv.format_message(
    {"role": "user", "content": "What is the history of the Metalheadz label?"},
    color_scheme={"user": "blue", "assistant": "red"}))
print(conv.format_message(
    {"role": "user", "content": "What is the history of the Metalheadz label?"},
    color_scheme={"user": "blue", "assistant": "red"},
    detailed=False))

#%%

from dotenv import load_dotenv
load_dotenv()

# %%
MODEL = "gpt-3.5-turbo"
# %%

# Create an instance of the OpenAI class, assumes OPENAI_API_KEY is set
client = OpenAI()
# Call the chat completions API
message, response = openai_chat_completions_create(
    messages=conv.messages,
    model=MODEL,
    client=client,
    return_response=True,
    max_tokens=256,
    temperature=0.8,
    max_retries=1,
)

# %%
print(message)
print(response.model_dump_json(indent=2))

# %%
# conv.append_message(response.choices[0].message.model_dump(exclude_unset=True))
conv.append_message(openai_extract_chat_completion_message(response))


conv.display_conversation()
# %%

conv.append_message({"role": "user", "content": "Who were the founders?"})
prompt = conv.get_prompt()
print(prompt)
# %%
message, response = openai_chat_completions_create(
    messages=conv.messages,
    model=MODEL,
    client=client,
    return_response=True,
    max_tokens=256,
    temperature=0.8,
    max_retries=1,
)

# %%
conv.append_message(openai_extract_chat_completion_message(response))
print(response.model_dump_json(indent=2))
print()
conv.display_conversation(detailed=True)