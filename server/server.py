from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydub import AudioSegment
import io
import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://huggingface.co/superb/wav2vec2-base-superb-er
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
# model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")

# https://huggingface.co/prithivMLmods/Speech-Emotion-Classification

feature_extractor = AutoFeatureExtractor.from_pretrained("prithivMLmods/Speech-Emotion-Classification")
model = AutoModelForAudioClassification.from_pretrained("prithivMLmods/Speech-Emotion-Classification")

model = model.to(device).eval()

id2label = {
    "0": "Anger",
    "1": "Calm",
    "2": "Disgust",
    "3": "Fear",
    "4": "Happy",
    "5": "Neutral",
    "6": "Sad",
    "7": "Surprised"
}

app = FastAPI(
    title="Emotion AppAPI",
    description="API for emotion prediction from audio.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class EmotionResponse(BaseModel):
    emotion: str
    confidence: float


@app.post("/predict_emotion", response_model=EmotionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        # Convert WebM/Opus to WAV in memory
        audio_seg = AudioSegment.from_file(io.BytesIO(raw), format="webm")
        audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
        wav_io = io.BytesIO()
        audio_seg.export(wav_io, format="wav")
        wav_io.seek(0)
        audio, rate = sf.read(wav_io, dtype="float32")
        if rate != 16000:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
            rate = 16000

        # Extract features
        inputs = feature_extractor(
            audio,
            sampling_rate=rate,
            return_tensors="pt",
            padding=True
            )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            # probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
            probs = F.softmax(logits, dim=-1).squeeze().tolist()
            # confidence = probs[0][predicted_id].item()  # Get probability for top class
            if isinstance(probs[0], list):  # batch dimension
                probs = probs[0]

            top_idx = int(torch.argmax(torch.tensor(probs)))
            top_label = id2label[str(top_idx)]
            confidence = round(probs[top_idx], 4)

            return {
                    "emotion": top_label,
                    "confidence":round(confidence, 4)
                    }



    except Exception as e:
            print("Error in /predict_emotion:", e)  # Add this line
            raise HTTPException(status_code=500, detail=str(e))
