from conversation import Conversation, get_conv_template, register_conv_template, conv_templates

def test_conversation_llama_2():
    # Create a conversation using the llama-2 template
    conversation = get_conv_template("llama-2")
    conversation.append_messages(
        messages=[
            {"role": "system", "content": "This is a system message"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    )

    # Test the get_prompt() method
    prompt = conversation.get_prompt()
    expected_prompt = "[INST] <<SYS>>\nThis is a system message\n<</SYS>>\n\n[INST] Hello [/INST] Hi there! [/INST]"
    assert prompt == expected_prompt, f"Prompt: {prompt}\nExpected: {expected_prompt}"

    # Test the copy() method
    copied_conversation = conversation.copy()
    assert copied_conversation.name == conversation.name
    assert copied_conversation.roles == conversation.roles
    assert copied_conversation.roles_templates == conversation.roles_templates
    assert copied_conversation.generator_str == conversation.generator_str
    assert copied_conversation.messages == conversation.messages
    assert copied_conversation.offset == conversation.offset
    assert copied_conversation.stop_str == conversation.stop_str
    assert copied_conversation.stop_token_ids == conversation.stop_token_ids

    # Test the register_conv_template() function
    register_conv_template(conversation)
    assert conv_templates[conversation.name] == conversation

    # Test the get_conv_template() function
    retrieved_conversation = get_conv_template(conversation.name)
    assert retrieved_conversation == conversation

test_conversation_llama_2()