import ast
import openai

from dotenv import load_dotenv

from llm_utils.endpoint_utils import openai_chat_completions_create

load_dotenv()

client = openai.Client()

model_list = client.models.list().data
model_list = [m for m in model_list if m.id.startswith("gpt-4")]
print("Available models: " + ", ".join([m.id for m in model_list]))

OPEN_AI_MODEL = "gpt-4-1106-preview"
print("Using model: " + OPEN_AI_MODEL)

def run_python_code(code, globals_=None, locals_=None):
    if globals_ is None:
        globals_ = {}
    if locals_ is None:
        locals_ = {}
    parsed = ast.parse(code)
    code_obj = compile(parsed, filename="<ast>", mode="exec")
    exec(code_obj, globals_, locals_)


# One line terminal colouring function, replacing the termcolor library
# taken from tinygrad https://github.com/tinygrad/tinygrad/blob/ca59054463b7d7567cf28d5ee81a008ed2ff8bab/tinygrad/helpers.py#L24
def colored(st, color=None, background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st  # noqa: E501


messages = [
    {"role": "system", "content": """\
You are a helpful and highly expert python coding assistant. \
When asked to generate code you always give the best code possible within code blocks surrounding the python code.
"""}
]


while True:
    prompt = input(colored("\nEnter prompt: ", "green"))
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
    print(colored("Response:\n", "green") + colored(content, "cyan"), flush=True)

    if '```python' not in content:
        print("No code block found in response\n\n")
        continue

    code = content.split('```python')[1].split('```')[0]
    print(colored('Code found\n---', "red"))
    print(colored(code, "cyan"))

    run_it = input(colored('Run code? [y/N]: ', 'red'))
    if run_it.lower() == 'y':
        run_python_code(code, globals_=globals(), locals_=locals())