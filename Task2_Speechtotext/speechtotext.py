import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

def load_audio(file_path):
    speech, sr = librosa.load(file_path, sr=16000)
    return speech

def transcribe_wav2vec2(audio_path):
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    speech = load_audio(audio_path)
    input_values = tokenizer(speech, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    print("Transcription:", transcription)
    return transcription

if _name_ == "_main_":
    file_path = input("Enter the path to your audio file (.wav recommended): ")
    transcribe_wav2vec2(file_path)
