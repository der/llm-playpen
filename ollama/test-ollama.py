from ollama import chat
import time

# MODEL_ID = "gemma3:12b-it-qat"
# MODEL_ID = "gemma3n:e4b"
# MODEL_ID = "moondream:latest"
# MODEL_ID = "llava:7b"
MODEL_ID = "llava:13b"
# MODEL_ID = "qwen2.5vl:7b"
# MODEL_ID = "qwen2.5vl:3b"

def test(messages):
    _messages = [ {
            "role": "system",
            "content": "You are a small, helpful, mobile droid",
        },
        messages
    ]
    start = time.time()
    response = chat(MODEL_ID, messages=_messages)
    response = response['message']['content']
    end = time.time()
    duration = end-start
    words = len(response.split())

    print(f"{duration:.1f}s {words/duration:.1f}w/s  >>>" + response)


test(
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  }
)

test({
    "role": "user",
    "content": "Write a short poem about pelicans",
})

test({
    "role": "user",
    "content": "This is an image from your camera, what do you see?",
    "images" : ["data/marvin-512.png"]
})
