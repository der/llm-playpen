from transformers import pipeline
import time

model_id = "openai/whisper-medium.en"
pipe = pipeline("automatic-speech-recognition", model=model_id)

def transcribe_speech(filepath):
    output = pipe(
        filepath,
    )
    return output["text"]

start = time.time()
transcription = transcribe_speech("data/JFKmoonspeech.wav")
end = time.time()
duration = end-start
print(f"{duration:.1f}s >>> {transcription}")
