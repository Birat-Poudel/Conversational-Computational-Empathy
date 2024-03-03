import whisper

model = whisper.load_model("tiny")

def predict_stt(filepath):
    result = model.transcribe(filepath)
    return result["text"]