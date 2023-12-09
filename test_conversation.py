import pytest
from conversation import Conversation, get_conv_template, register_conv_template, conv_templates

@pytest.fixture
def conversation_llama_2():
    conversation = get_conv_template("llama-2")
    conversation.append_messages(
        messages=[
            {"role": "system", "content": "This is a system message"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    )
    return conversation


def test_get_prompt(conversation_llama_2):
    prompt = conversation_llama_2.get_prompt(add_generator_prompt=False)
    expected_prompt = "<s>[INST] <<SYS>>\nThis is a system message\n<</SYS>>\n\nHello [/INST] Hi there! </s>"
    assert prompt == expected_prompt


def test_get_prompt_with_generator_prompt(conversation_llama_2):
    conversation_llama_2.append_message("user", "How now brown cow?")
    prompt = conversation_llama_2.get_prompt(add_generator_prompt=True)
    expected_prompt = "<s>[INST] <<SYS>>\nThis is a system message\n<</SYS>>\n\nHello [/INST] Hi there! </s><s>[INST] How now brown cow? [/INST] "
    assert prompt == expected_prompt


def test_copy(conversation_llama_2):
    copied_conversation = conversation_llama_2.copy()
    assert copied_conversation.name == conversation_llama_2.name
    assert copied_conversation.roles == conversation_llama_2.roles
    assert copied_conversation.roles_templates == conversation_llama_2.roles_templates
    assert copied_conversation.generator_str == conversation_llama_2.generator_str
    assert copied_conversation.messages == conversation_llama_2.messages
    assert copied_conversation.offset == conversation_llama_2.offset
    assert copied_conversation.stop_str == conversation_llama_2.stop_str
    assert copied_conversation.stop_token_ids == conversation_llama_2.stop_token_ids


def test_register_conv_template():
    conversation = Conversation(name="test", roles=("user", "assistant"), messages=[])
    register_conv_template(conversation)
    assert conv_templates[conversation.name] == conversation


def test_get_conv_template():
    conversation = Conversation(name="test", roles=("user", "assistant"), messages=[])
    conv_templates[conversation.name] = conversation
    retrieved_conversation = get_conv_template(conversation.name)
    assert retrieved_conversation == conversation


def test_register_already_existing_without_flag():
    with pytest.raises(Exception):
        register_conv_template(Conversation(name="llama-2", roles=("user", "assistant"), messages=[]))


def test_to_openai_api_messages(conversation_llama_2):
    messages = conversation_llama_2.to_openai_api_messages()
    expected_messages = [
        {"role": "system", "content": "This is a system message"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert messages == expected_messages


def test_chatml_template():
    conversation = get_conv_template("chatml")
    assert conversation.name == "chatml"
    assert conversation.roles == ("system", "user", "assistant")
    assert conversation.generator_str == "<|im_start|>assistant\n"
    conversation.append_messages(
        messages=[
            {"role": "system", "content": """
You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.
Knowledge cutoff: 2021-09-01
Current date: 2023-03-01"""},
            {"role": "user", "content": "How are you"},
            {"role": "assistant", "content": "I am doing well!"},
            {"role": "user", "content": "How are you now?"},
        ]
    )
    assert conversation.offset == 0
    assert conversation.stop_str == "<|im_end|>"
    assert conversation.stop_token_ids is None
    prompt = conversation.get_prompt(add_generator_prompt=True)
    print(prompt)
    print()
    expected_prompt = "<|im_start|>system\nYou are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-03-01<|im_end|>\n<|im_start|>user\nHow are you<|im_end|>\n<|im_start|>assistant\nI am doing well!<|im_end|>\n<|im_start|>user\nHow are you now?<|im_end|>\n<|im_start|>assistant\n"
    print(expected_prompt)
    assert prompt == expected_prompt
