from unsloth import FastModel
import torch
from transformers import TextStreamer
from PIL import Image
import gc

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    dtype = None, # fp16
    max_seq_length = 1024,
    load_in_4bit = True,
    full_finetuning = False,
)

def do_gemma_3n_inference(model, messages, max_new_tokens = 128):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")
    _ = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
    # Cleanup to reduce VRAM usage
    # del inputs
    # torch.cuda.empty_cache()
    # gc.collect()

print("Ready to start")
messages = [{
    "role" : "user",
    "content": [
        { "type": "image", "image" : "marvin.png" },
        { "type": "text",  "text" : "Describe this image." }
    ]
}]
# You might have to wait 1 minute for Unsloth's auto compiler
do_gemma_3n_inference(model, messages, max_new_tokens = 256)

messages = [{
    "role": "user",
    "content": [{ "type" : "text",
                  "text" : "Write a poem about the kraken." }]
}]
do_gemma_3n_inference(model, messages, max_new_tokens = 256)

audio_file = "JFKmoonspeech.mp3"
messages = [{
    "role" : "user",
    "content": [
        { "type": "audio", "audio" : audio_file },
        { "type": "text",  "text" : "Transcribe this audio?" }
    ]
}]
do_gemma_3n_inference(model, messages, max_new_tokens = 256)
