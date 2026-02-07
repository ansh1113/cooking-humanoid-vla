"""
🚀 NUCLEAR OPTION - EVERYTHING AT ONCE
- Curriculum Learning (easy → hard)
- Mixup Augmentation
- Multiple Models Ensemble
- Test-Time Augmentation
Target: 70%+ on 40 classes
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import random

FEATURE_DIR = 'data/features_golden40'
DATA_FILE = 'data/processed/golden_40_ready.json'
BATCH_SIZE = 64  # BIGGER batches
EPOCHS = 150     # MORE epochs
LR = 1e-4
device = torch.device('cuda')

# MIXUP AUGMENTATION
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class BIG_VLA(nn.Module):
    """BIGGER MODEL - 8 layers, 768 hidden"""
    def __init__(self, num_classes):
        super().__init__()
        hidden = 768
        self.proj = nn.Linear(512, hidden)
        self.pos = nn.Parameter(torch.randn(1, 30, hidden) * 0.01)
        
        # 8 LAYERS!
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden, 
            nhead=12, 
            dim_feedforward=hidden*4, 
            dropout=0.2,
            batch_first=True,
            norm_first=True  # Pre-norm
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=8)
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden//2, num_classes)
        )
    
    def forward(self, x):
        x = self.proj(x)
        x = x + self.pos
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)

class AugDataset(Dataset):
    def __init__(self, data, vocab, augment=False):
        self.data = data
        self.vocab = vocab
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        url = item.get('video_url', '')
        vid_id = url.split('=')[-1].split('&')[0]
        path = f"{FEATURE_DIR}/{vid_id}_{item['start']:.2f}.pt"
        feat = torch.load(path, weights_only=True) if os.path.exists(path) else torch.zeros(30, 512)
        
        # AGGRESSIVE AUGMENTATION
        if self.augment:
            # Time shift
            if random.random() < 0.6:
                shift = random.randint(-4, 4)
                feat = torch.roll(feat, shift, dims=0)
            # Dropout frames
            if random.random() < 0.3:
                mask = torch.rand(30) > 0.2
                feat = feat * mask.unsqueeze(1)
            # Gaussian noise
            if random.random() < 0.4:
                feat = feat + torch.randn_like(feat) * 0.02
            # Random scaling
            if random.random() < 0.3:
                scale = 0.8 + 0.4 * random.random()
                feat = feat * scale
        
        return feat, self.vocab[item['action']]

# CURRICULUM LEARNING - sort by class difficulty
def get_curriculum_order(data, vocab):
    action_counts = Counter(d['action'] for d in data)
    
    # Easy = high sample count, Hard = low sample
    easy_actions = {a for a, c in action_counts.items() if c >= 60}
    medium_actions = {a for a, c in action_counts.items() if 30 <= c < 60}
    hard_actions = {a for a, c in action_counts.items() if c < 30}
    
    # Phase 1: Easy only
    phase1 = [d for d in data if d['action'] in easy_actions]
    # Phase 2: Easy + Medium
    phase2 = [d for d in data if d['action'] in easy_actions | medium_actions]
    # Phase 3: All
    phase3 = data
    
    return phase1, phase2, phase3

print("🚀 NUCLEAR TRAINING")

with open(DATA_FILE) as f:
    data = json.load(f)

actions = [d['action'] for d in data]
vocab = {a: i for i, a in enumerate(sorted(set(actions)))}
print(f"Classes: {len(vocab)}")

# Split
single = {a for a, c in Counter(actions).items() if c == 1}
multi = [d for d in data if d['action'] not in single]
single_data = [d for d in data if d['action'] in single]

train_multi, val_multi = train_test_split(multi, test_size=0.15, stratify=[vocab[d['action']] for d in multi], random_state=42)
train_data = train_multi + single_data

print(f"Train: {len(train_data)} | Val: {len(val_multi)}")

# CURRICULUM
phase1, phase2, phase3 = get_curriculum_order(train_data, vocab)
print(f"Curriculum: P1={len(phase1)} P2={len(phase2)} P3={len(phase3)}")

# Model
model = BIG_VLA(len(vocab)).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=LR*5, 
    epochs=EPOCHS, 
    steps_per_epoch=len(train_data)//BATCH_SIZE
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

val_loader = DataLoader(AugDataset(val_multi, vocab), batch_size=BATCH_SIZE)

best = 0
patience = 0

for epoch in range(EPOCHS):
    # CURRICULUM: phase selection
    if epoch < 20:
        curr_data = phase1
    elif epoch < 50:
        curr_data = phase2
    else:
        curr_data = phase3
    
    train_loader = DataLoader(AugDataset(curr_data, vocab, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    
    # Train
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        # MIXUP
        if random.random() < 0.5:
            x, y_a, y_b, lam = mixup_data(x, y)
            optimizer.zero_grad()
            out = model(x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    # Val with TTA (Test-Time Augmentation)
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            # TTA: 5 augmented predictions
            preds = []
            for _ in range(5):
                x_aug = x.clone()
                # Random shift
                shift = random.randint(-2, 2)
                x_aug = torch.roll(x_aug, shift, dims=1)
                preds.append(torch.softmax(model(x_aug), dim=1))
            
            # Average predictions
            out = torch.stack(preds).mean(dim=0)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    val_acc = 100 * correct / total
    
    if val_acc > best:
        best = val_acc
        patience = 0
        torch.save({'model': model.state_dict(), 'vocab': vocab}, 'models/nuclear.pt')
        print(f"Epoch {epoch+1:3d}: {val_acc:5.1f}% ✅")
    else:
        patience += 1
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:3d}: {val_acc:5.1f}%")
        if patience >= 30:
            print("Early stop")
            break

print(f"\n🏆 FINAL: {best:.1f}%")
