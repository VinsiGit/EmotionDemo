from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydub import AudioSegment
import io
import librosa
import soundfile as sf
import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from pathlib import Path
# Load model and processor
model = Wav2Vec2ForSequenceClassification.from_pretrained("Dpngtm/wav2vec2-emotion-recognition")
processor = Wav2Vec2Processor.from_pretrained("Dpngtm/wav2vec2-emotion-recognition")
model.eval()  # Ensure model is in eval mode

# FastAPI setup
app = FastAPI(
    title="Emotion App API",
    description="API for emotion prediction from audio.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response model
class EmotionResponse(BaseModel):
    emotion: str
    confidence: float

@app.post("/predict_emotion", response_model=EmotionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    try:
        # Read file content
        raw = await file.read()
        ext = Path(file.filename).suffix.lower().replace('.', '')

        # Convert to mono WAV at 16kHz
        audio_seg = AudioSegment.from_file(io.BytesIO(raw), format=ext)
        audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)

        # Export to in-memory buffer
        wav_io = io.BytesIO()
        audio_seg.export(wav_io, format="wav")
        wav_io.seek(0)

        # Load audio with soundfile
        audio, rate = sf.read(wav_io, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono

        if rate != 16000:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
            rate = 16000

        # Ensure audio isn't silent
        if np.std(audio) < 1e-5:
            raise HTTPException(status_code=400, detail="Audio is too silent or empty.")

        # Process input
        inputs = processor(audio, sampling_rate=rate, return_tensors="pt", padding=True)

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()

        # Get label mapping from model config
        id2label = model.config.id2label
        emotion_labels = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

        # Get predicted emotion
        top_idx = torch.argmax(probs).item()
        confidence = probs[top_idx].item()
        predicted_emotion = emotion_labels[top_idx]

        return EmotionResponse(emotion=predicted_emotion, confidence=confidence)

    except Exception as e:
        print("Error in /predict_emotion:", str(e))
        raise HTTPException(status_code=500, detail="Error processing audio: " + str(e))
