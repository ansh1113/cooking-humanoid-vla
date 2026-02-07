"""
FINAL ATTEMPT - HEAVY REGULARIZATION
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import random

FEATURE_DIR = 'data/features_golden40'
DATA_FILE = 'data/processed/golden_40_ready.json'
MODEL_SAVE = 'models/indian_40_final.pt'

BATCH_SIZE = 32  # Larger batches
EPOCHS = 100
LR = 5e-5  # Lower LR
WEIGHT_DECAY = 0.05  # HEAVY weight decay
DROPOUT = 0.3  # More dropout
LABEL_SMOOTHING = 0.1  # Label smoothing!

device = torch.device('cuda')

class RegularizedVLA(nn.Module):
    def __init__(self, num_actions=40):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, 30, 512) * 0.01)
        encoder = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=6)
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, num_actions)
        )
    def forward(self, x):
        x = x + self.pos
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)

class AugmentedDataset(Dataset):
    def __init__(self, data, vocab, augment=False):
        self.data, self.vocab, self.augment = data, vocab, augment
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        url = item.get('video_url', '')
        vid_id = url.split('=')[-1].split('&')[0]
        path = f"{FEATURE_DIR}/{vid_id}_{item['start']:.2f}.pt"
        feat = torch.load(path, weights_only=True) if os.path.exists(path) else torch.zeros(30, 512)
        
        # HEAVY AUGMENTATION
        if self.augment:
            if random.random() < 0.5:
                feat = torch.roll(feat, random.randint(-3, 3), dims=0)
            if random.random() < 0.3:
                feat = feat + torch.randn_like(feat) * 0.03
            if random.random() < 0.2:
                mask_frames = random.randint(1, 3)
                mask_idx = random.sample(range(30), mask_frames)
                feat[mask_idx] = 0
        
        return feat, self.vocab[item['action']]

with open(DATA_FILE) as f:
    data = json.load(f)

actions = [d['action'] for d in data]
vocab = {a: i for i, a in enumerate(sorted(set(actions)))}

# Split
single_sample = {a for a, c in Counter(actions).items() if c == 1}
multi = [d for d in data if d['action'] not in single_sample]
single = [d for d in data if d['action'] in single_sample]

train_multi, val_multi = train_test_split(multi, test_size=0.15, stratify=[vocab[d['action']] for d in multi], random_state=42)
train_data = train_multi + single

train_loader = DataLoader(AugmentedDataset(train_data, vocab, augment=True), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(AugmentedDataset(val_multi, vocab, augment=False), batch_size=BATCH_SIZE)

model = RegularizedVLA(len(vocab)).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

print(f"🚀 FINAL TRAINING - HEAVY REGULARIZATION")
print(f"Train: {len(train_data)} | Val: {len(val_multi)}")

best_val, patience = 0, 0

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            correct += (torch.max(model(x), 1)[1] == y).sum().item()
            total += y.size(0)
    
    val_acc = 100 * correct / total
    scheduler.step(val_acc)
    
    if val_acc > best_val:
        best_val = val_acc
        patience = 0
        torch.save({'model': model.state_dict(), 'vocab': vocab}, MODEL_SAVE)
        print(f"Epoch {epoch+1:3d}: {val_acc:5.1f}% ✅")
    else:
        patience += 1
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1:3d}: {val_acc:5.1f}%")
        if patience >= 20:
            break

print(f"\n🏆 FINAL: {best_val:.1f}%")
