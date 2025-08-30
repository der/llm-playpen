# Use a pipeline as a high-level helper
from transformers import pipeline
import torch
import time

pipe = pipeline(
#    "image-text-to-text",
    model="google/gemma-3n-e4b-it",
    device="cuda",
    torch_dtype=torch.bfloat16,
    torch_compile=False
)

    
def run_request(request):
    start = time.time()
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a small, helpful, mobile droid"},
            ],
        },
        {
            "role": "user",
            "content": request,
        }
    ]

    output = pipe(text=messages, max_new_tokens=200)
    response = output[0]["generated_text"][-1]["content"]
    end = time.time()
    duration = end-start
    words = len(response.split())

    print(f"{duration:.1f}s {words/duration:.1f}w/s  >>>" + response)

def test():
    run_request([
                {"type": "text", "text": "Write a short poem about pelicans"}
            ])
    run_request([
                {"type": "text", "text": "This is an image from your camera, what do you see?"},
                {"type": "image", "image": "marvin-256.png"},
            ])
    run_request([
                {"type": "text", "text": "This is an image from your camera, what do you see?"},
                {"type": "image", "image": "marvin-512.png"},
            ])
    run_request([
                {"type": "text", "text": "Transcribe this audio"},
                {"type": "audio", "audio": "JFKmoonspeech.mp3"},
            ])
    
test()
