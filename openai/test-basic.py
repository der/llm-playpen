from openai import OpenAI
import sys
import base64

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key=""
)

completion = client.chat.completions.create(
    model="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
    messages=[
        {
            "role": "user",
            "content": "Write a short poem about pelicans, in English."
        },
    ],
    stream=True,
)

for chunk in completion:
    content = chunk.choices[0].delta.content
    if content is not None:
        sys.stdout.write(content)
        sys.stdout.flush()


prompt = "Describe this image?"
with open("../data/marvin-scaled.png", "rb") as image_file:
    b64_image = base64.b64encode(image_file.read()).decode("utf-8")

print("\n\n---\n")
completion = client.chat.completions.create(
    model="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            ],
        }
    ],
)

print(completion.choices[0].message.content)
