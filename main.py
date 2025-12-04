import os
import glob
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing

device = 'cuda'
cache_dir = "cache_gtzan"
data_path = r"D:\DownLoad\archive\Data\genres_original"
batch_size = 32                 # 4060 建议 32 或 64
num_epochs = 10
lr = 1e-3
dropout_rate = 0.3
genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
num_classes = len(genres)


# Dataset：只读取 embedding
class EmbeddingDataset(Dataset):
    def __init__(self, wav_dir, cache_dir):
        self.files = glob.glob(os.path.join(wav_dir, "*", "*.wav"))
        self.labels = [f.split("\\")[-2] for f in self.files]
        self.label2id = {l:i for i,l in enumerate(sorted(set(self.labels)))}
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav = self.files[idx]
        label = self.label2id[self.labels[idx]]
        emb_path = os.path.join(self.cache_dir, os.path.basename(wav) + ".npy")

        try:
            emb = np.load(emb_path)  # shape: (2048,)
        except Exception:
            return None  # 返回 None，collate_fn 会过滤掉

        return torch.tensor(emb, dtype=torch.float32), label

def collate_fn(batch):
    batch = [x for x in batch if x is not None]  # 过滤掉 None
    if len(batch) == 0:
        return None, None
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


def train(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc="Training"):
        if images is None:
            continue
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def eval(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            #Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy, all_labels, all_preds


def main():
    dataset = EmbeddingDataset(data_path, cache_dir)
    indices = list(range(len(dataset)))
    labels = dataset.labels

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,
                              collate_fn=collate_fn)

    test_loader = DataLoader(Subset(dataset, test_idx),
                             batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True,
                             collate_fn=collate_fn)


    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler()   # AMP 混合精度

    best_acc = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scaler)
        test_loss, test_acc, all_labels, all_preds = eval(model, test_loader, criterion)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), "trained_classifier.pth")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genres)

    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)  # 保存到文件
    plt.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows 多进程兼容
    main()