import threading
import time
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText, BitsAndBytesConfig

# ------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------
MODEL_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"

# ------------------------------------------------------
# MODEL SINGLETON (load once, reuse)
# ------------------------------------------------------

_model = None
_processor = None
_tokenizer = None
_model_lock = threading.Lock()


def get_model_and_processor():
    global _model, _processor, _tokenizer
    with _model_lock:
        if _model is None or _processor is None:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            _processor = AutoProcessor.from_pretrained(MODEL_ID)
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            _model = (
                AutoModelForImageTextToText
                .from_pretrained(MODEL_ID, device_map="auto", quantization_config=quantization_config)
                .eval()
            )
            # _model = torch.compile(_model, mode="max-autotune")
        return _model, _processor, _tokenizer
    
def run_request(request):
    model, processor, _tokenizer = get_model_and_processor()
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

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device, dtype=model.dtype)

    with torch.inference_mode():
        generation = model.generate(
            **inputs, max_new_tokens=128, disable_compile=True,
        )

    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(
        generation[0][input_len:], skip_special_tokens=True
    )
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
                {"type": "image", "image": "data/marvin-512.png"},
            ])
    run_request([
                {"type": "text", "text": "This is an image from your camera, what do you see?"},
                {"type": "image", "image": "data/marvin-256.png"},
            ])
    # run_request([
    #             {"type": "text", "text": "Transcribe this audio"},
    #             {"type": "audio", "audio": "data/JFKmoonspeech.mp3"},
    #         ])
    
test()
