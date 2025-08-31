import threading
import time
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration, AutoTokenizer

# ------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------
MODEL_ID = "google/gemma-3n-e4b-it"
# MODEL_ID = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"
# MODEL_ID = "BernTheCreator/Gemma-3n-E4B-it-Q4_0-GGUF"

# ------------------------------------------------------
# MODEL SINGLETON (load once, reuse)
# ------------------------------------------------------

_model = None
_processor = None
_tokenizer = None
_model_lock = threading.Lock()

def get_current_temperature(location: str, unit: str):
    """
    Get the current temperature at a location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    """
    print("Get temperature tool called")
    return 42.  # A real function should probably actually get the temperature!

def get_current_wind_speed(location: str):
    """
    Get the current wind speed in km/h at a given location.
    
    Args:
        location: The location to get the wind speed for, in the format "City, Country"
    """
    print("Get wind tool called")
    return 6.  # A real function should probably actually get the wind speed!

tools = [get_current_temperature, get_current_wind_speed]

def get_model_and_processor():
    global _model, _processor, _tokenizer
    with _model_lock:
        if _model is None or _processor is None:
            _processor = AutoProcessor.from_pretrained(MODEL_ID)
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            _model = (
                Gemma3nForConditionalGeneration
                .from_pretrained(MODEL_ID, device_map="auto", dtype="auto")
                .eval()
            )
            # _model = torch.compile(_model, mode="max-autotune")
        return _model, _processor, _tokenizer
    
def run_request(request):
    model, processor, tokenizer = get_model_and_processor()
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

    inputs = tokenizer.apply_chat_template(
        messages,
        tools = tools,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generation = model.generate(
            **inputs, max_new_tokens=128, disable_compile=True,
        )
    print(tokenizer.decode(generation[0][len(inputs["input_ids"][0]):]))
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
            {"type": "text", "text": "Using the supplied tools, what is the temperature in paris?"}
        ])

test()
