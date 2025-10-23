from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key=""
)


def add_two_numbers(a: int, b: int) -> int:
  """
  Add two numbers

  Args:
    a (int): The first number
    b (int): The second number

  Returns:
    int: The sum of the two numbers
  """

  return int(a) + int(b)


def subtract_two_numbers(a: int, b: int) -> int:
  """
  Subtract two numbers
  """

  return int(a) - int(b)

functions = [
    {
       "type": "function",
       "function": {
            "name": "zoom_two_numbers",
            "description": "Zoom two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number"
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number"
                    }
                },
                "required": ["a", "b"],
            },
        }
    },{
        "type": "function",
        "function": {
            "name": "subtract_two_numbers",
            "description": "Subtract two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number"
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number"
                    },
                },
                "required": ["a", "b"],
            },
        }
    }
]

messages = [{'role': 'user', 'content': 'Using the tools, zoom 1234 and 4321'}]
print('Prompt:', messages[0]['content'])

completion = client.chat.completions.create(
    model="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
    messages=messages,
    tools=functions,
    tool_choice="auto"
)

print('Completion:', completion)

msg = completion.choices[0].message
if msg.tool_calls:
    available_functions = {
        "zoom_two_numbers": add_two_numbers,
        "subtract_two_numbers": subtract_two_numbers,
    }
    f = msg.tool_calls[0].function
    function_name = f.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(f.arguments)
    function_response = function_to_call(
        a=function_args.get("a"),
        b=function_args.get("b")
    )
    print(f"Function result: {function_response}")
else:
    print("No function call in the completion.")
