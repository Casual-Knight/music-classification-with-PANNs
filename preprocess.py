import os
import glob
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from pytorch.models import Cnn14

device = "cuda"

data_path = r"D:\DownLoad\archive\Data\genres_original"
cache_dir = "cache_gtzan"
os.makedirs(cache_dir, exist_ok=True)

# ---------- 加载 PANNs ----------
model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
              mel_bins=64, fmin=50, fmax=14000, classes_num=527)

pretrained = torch.load("Cnn14_mAP=0.431.pth", map_location=device)
model.load_state_dict(pretrained["model"])
model.to(device)
model.eval()

# 遍历所有 wav
files = glob.glob(os.path.join(data_path, "*", "*.wav"))

for f in tqdm(files):
    save_path = os.path.join(cache_dir, os.path.basename(f) + ".npy")
    if os.path.exists(save_path):
        continue

    try:
        wav, sr = torchaudio.load(f)
        wav = torchaudio.functional.resample(wav, sr, 32000)
        wav = wav.to(device)
    except:
        print("跳过损坏文件:", f)
        continue

    with torch.no_grad():
        out = model(wav)
        emb = out["embedding"]  # shape: (1,2048) 或 (2048,)
        emb = emb.squeeze().cpu().numpy()

    np.save(save_path, emb)

print("全部 embedding 提取完毕")
