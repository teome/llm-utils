import json
import os
import openai
from dotenv import load_dotenv
import requests
import ast

load_dotenv()

from endpoint_utils import openai_chat_completions_create

client = openai.Client()

model_list = client.models.list().data
model_list = [m for m in model_list if m.id.startswith("gpt-4")]
OPEN_AI_MODEL = "gpt-4-1106-preview"

def run_python_code(code, globals_=None, locals_=None):
    if globals_ is None:
        globals_ = {}
    if locals_ is None:
        locals_ = {}
    parsed = ast.parse(code)
    code_obj = compile(parsed, filename="<ast>", mode="exec")
    exec(code_obj, globals_, locals_)

messages = [
    {"role": "system", "content": """\
You are a helpful and highly expert python coding assistant. \
When asked to generate code you always give the best code possible within code blocks surrounding the python code.
"""}
]


while True:
    prompt = input("Enter prompt: ")
    if prompt == "":
        print('Testing with default prompt')
        prompt = """\
        Write python code that will download a file from a URL available as an existing local variable `url` and save it to a file. \
        The `url` variable should not be set by the code you generate. \
        """

    messages_req = messages + [{"role": "user", "content": prompt}]
    content, response = openai_chat_completions_create(
        messages_req, model=OPEN_AI_MODEL, client=client, return_response=True, max_retries=2)
    completion_message = response.choices[0].message.model_dump(exclude_unset=True)

    url = "https://cf1.zzounds.com/media/productmedia/fit,600by600/quality,85/Prophet-5_Left_Angle_820175-435c1c5a0e3d3a898c11179315824fc9.jpg"

    content = completion_message["content"]
    assert '```python' in content, "No code block found in response\n\n" + content

    code = content.split('```python')[1].split('```')[0]
    print('Code found\n---\n' + code, flush=True)

    run_it = input('Run code? [y/N]: ')
    if run_it.lower() == 'y':
        run_python_code(code, globals_=globals(), locals_=locals())