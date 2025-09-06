from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import time
import librosa
import torch

model_id = "openai/whisper-small.en"

# load model and processor from pre-trained
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id).to("cuda")

# load audio file: user is responsible for loading the audio files themselves
data, sr = librosa.load("data/JFKmoonspeech.wav")
                               
start = time.time()
inputs = processor(data, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=16_000).input_features.to("cuda")
predicted_ids = model.generate(inputs)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
end = time.time()
duration = end-start
print(f"{duration:.1f}s >>> {transcription}")
