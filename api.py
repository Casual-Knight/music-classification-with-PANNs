from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
import torchaudio
from pytorch.models import Cnn14
from main import Classifier, genres
import uvicorn

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 PANNs
panns = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
              mel_bins=64, fmin=50, fmax=14000, classes_num=527)

pretrained = torch.load("Cnn14_mAP=0.431.pth", map_location=device)
panns.load_state_dict(pretrained["model"])
panns.to(device)
panns.eval()

# 分类器
classifier = Classifier()
classifier.load_state_dict(torch.load("trained_classifier.pth", map_location=device))
classifier.to(device)
classifier.eval()

app = FastAPI()


def extract_embedding(wav_tensor, sr):
    wav = torchaudio.functional.resample(wav_tensor, sr, 32000).to(device)
    with torch.no_grad():
        out = panns(wav)
        emb = out["embedding"].squeeze()
    return emb


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 读取音频
    wav, sr = torchaudio.load(file.file)
    emb = extract_embedding(wav, sr)

    # 分类
    with torch.no_grad():
        logits = classifier(emb.unsqueeze(0).to(device))
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_id = int(np.argmax(prob))
    result = {
        "genre": genres[top_id],
        "probabilities": {g: float(p) for g, p in zip(genres, prob)}
    }
    return result


@app.get("/model/info")
async def info():
    return {
        "panns_version": "Cnn14_mAP=0.431",
        "classifier_model": "trained_classifier.pth",
        "num_classes": len(genres),
        "genres": genres
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
