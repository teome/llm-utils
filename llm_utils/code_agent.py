import argparse
import ast
import sys

import openai
from dotenv import load_dotenv


load_dotenv()


SYSTEM_PROMPT = """\
You are a helpful and highly expert python coding assistant. \
When asked to generate code you always give the best code possible within code blocks surrounding the python code. \
Before answering any question, outline the steps you will take to solve the problem. \
"""


def get_models(client):
    model_list = client.models.list().data
    model_list = [m.id for m in model_list if m.id.startswith("gpt-4")]
    print("Available models: " + ", ".join(model_list))
    return model_list


def run_python_code(code, globals_=None, locals_=None):
    if globals_ is None:
        globals_ = {}
    if locals_ is None:
        locals_ = {}
    parsed = ast.parse(code)
    code_obj = compile(parsed, filename="<ast>", mode="exec")
    exec(code_obj, globals_, locals_)  # pylint: disable=exec-used


# One line terminal colouring function, replacing the termcolor library
# taken from tinygrad https://github.com/tinygrad/tinygrad/blob/ca59054463b7d7567cf28d5ee81a008ed2ff8bab/tinygrad/helpers.py#L24
def colored(st, color=None, background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st  # noqa: E501 # pylint: disable=line-too-long


def process_prompt(
    model: str,
    client: openai.Client,
    system_prompt: str,
    max_tokens: int = 1024,
    append_response_content: bool = True
):

    messages = [{"role": "system", "content": system_prompt},]

    while True:
        prompt = input(colored("\nEnter prompt: ", "green"))
        if prompt == "":
            print('Testing with default prompt')
            prompt = """\
            Write python code that will download a file from a URL available as an existing local variable `url` and save it to a file. \
            The `url` variable should not be set by the code you generate. \
            """
            url = "https://cf1.zzounds.com/media/productmedia/fit,600by600/quality,85/Prophet-5_Left_Angle_820175-435c1c5a0e3d3a898c11179315824fc9.jpg"  # pylint: disable=unused-variable, possibly-unused-variable, line-too-long

        messages.append({"role": "user", "content": prompt})

        # content, response = openai_chat_completions_create(
        #     messages, model=model, client=client, max_tokens=max_tokens,
        #     return_response=True, max_retries=2)
        # completion_message = response.choices[0].message.model_dump(
        #     exclude_unset=True)


        # content = completion_message["content"]
        # print(colored("Response:\n", "green") +
        #       colored(content, "cyan"), flush=True)

        stream = client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            stream=True,
        )

        content = ""
        print(colored("Response:\n", "green"))
        for chunk in stream:
            chunk_content = chunk.choices[0].delta.content or ""
            content += chunk_content
            print(colored(chunk_content, "cyan"), end="")
        print("\n")

        if append_response_content:
            messages.append({"role": "assistant", "content": content})
        else:
            # clear to just system message for next run
            messages = messages[:1]

        if '```python' not in content:
            print("No code block found in response\n\n")
            continue

        code = content.split('```python')[1].split('```')[0]
        print(colored('Code found\n---', "red"))
        print(colored(code, "cyan"))

        run_it = input(colored('Run code? [y/N]: ', 'red'))
        if run_it.lower() == 'y':
            run_python_code(code, globals_=globals(), locals_=locals())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", help="Specify the model to use", default="gpt-4-1106-preview")
    parser.add_argument(
        "--append", help="Append response content to messages", action="store_true")
    parser.add_argument(
        "--max-tokens", help="Max tokens to use", default=1024, type=int)
    parser.add_argument(
        "--models", help="List available models", action="store_true")
    parser.add_argument(
        "--system-prompt", help="System prompt", default=SYSTEM_PROMPT)

    args = parser.parse_args()

    client = openai.Client()

    # make sure the model is valid
    model_list = get_models(client)

    if args.append:
        print("Appending response content to messages. The max tokens limit may be exceeded for multiple responses.")

    if args.model not in model_list:

        print("Invalid model specified: " + args.model)
        sys.exit(1)

    print("Using model: " + colored(args.model, "yellow"))
    print("Using system prompt: " + colored(args.system_prompt, "yellow"))
    process_prompt(args.model, client, args.system_prompt,
                   args.max_tokens, args.append)
