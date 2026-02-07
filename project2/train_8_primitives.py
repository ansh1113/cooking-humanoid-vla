"""
TEST: 8 ROBOT PRIMITIVES
If this works (70%+), problem is semantic confusion
If this fails (10%), problem is features/model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import random

FEATURE_DIR = 'data/features_golden40'
DATA_FILE = 'data/processed/golden_40_ready.json'
MODEL_SAVE = 'models/primitive_test.pt'

PRIMITIVE_MAP = {
    'stirring curry': 'STIR', 'stirring gently': 'STIR', 'sautéing': 'STIR',
    'frying onions': 'STIR', 'simmering': 'STIR', 'boiling': 'STIR',
    'roasting': 'STIR', 'shallow frying': 'STIR', 'deep frying': 'STIR',
    
    'adding water': 'POUR', 'adding oil': 'POUR', 'adding ghee': 'POUR',
    'adding butter': 'POUR',
    
    'adding vegetables': 'TRANSFER', 'adding tomatoes': 'TRANSFER',
    'adding paneer': 'TRANSFER', 'adding chicken': 'TRANSFER',
    'adding rice': 'TRANSFER', 'plating': 'TRANSFER', 'serving': 'TRANSFER',
    
    'adding masala': 'SPRINKLE', 'adding turmeric': 'SPRINKLE',
    'adding chili powder': 'SPRINKLE', 'tempering spices': 'SPRINKLE',
    'garnishing with coriander': 'SPRINKLE', 'garnishing with cream': 'SPRINKLE',
    'adding coriander': 'SPRINKLE',
    
    'mixing thoroughly': 'MIX',
    'kneading dough': 'KNEAD',
    
    'grinding paste': 'PROCESS', 'chopping onion': 'PROCESS',
    'chopping vegetables': 'PROCESS', 'mincing ginger garlic': 'PROCESS',
    'adding ginger garlic paste': 'POUR', 'peeling': 'PROCESS',
    'slicing tomato': 'PROCESS',
    
    'pressure cooking': 'WAIT', 'steaming': 'WAIT', 'grilling': 'WAIT',
    'washing ingredients': 'WAIT',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleVLA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, 30, 512) * 0.02)
        encoder = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=4)
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = x + self.pos
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)

class PrimitiveDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        url = item.get('video_url', '')
        vid_id = url.split('=')[-1].split('&')[0]
        start = item['start']
        path = f"{FEATURE_DIR}/{vid_id}_{start:.2f}.pt"
        if os.path.exists(path):
            feats = torch.load(path, weights_only=True)
        else:
            feats = torch.zeros(30, 512)
        primitive = PRIMITIVE_MAP.get(item['action'], 'WAIT')
        label = self.vocab[primitive]
        return feats, label

print("🧪 TESTING 8 PRIMITIVES")

with open(DATA_FILE) as f:
    data = json.load(f)

primitives = [PRIMITIVE_MAP.get(d['action'], 'WAIT') for d in data]
counts = Counter(primitives)
vocab = {p: i for i, p in enumerate(sorted(counts.keys()))}

print(f"\n📊 PRIMITIVES ({len(vocab)} classes):")
for p, c in counts.most_common():
    print(f"  {c:3d}x {p}")

train_idx, val_idx = train_test_split(range(len(data)), test_size=0.15, stratify=primitives, random_state=42)
train_data = [data[i] for i in train_idx]
val_data = [data[i] for i in val_idx]

train_loader = DataLoader(PrimitiveDataset(train_data, vocab), batch_size=32, shuffle=True)
val_loader = DataLoader(PrimitiveDataset(val_data, vocab), batch_size=32)

model = SimpleVLA(len(vocab)).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()

print(f"\n🚀 Training...")
best = 0

for epoch in range(40):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, pred = torch.max(model(x), 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    val_acc = 100 * correct / total
    if val_acc > best:
        best = val_acc
        print(f"Epoch {epoch+1:2d}: {val_acc:5.1f}% ✅")
    else:
        print(f"Epoch {epoch+1:2d}: {val_acc:5.1f}%")

print(f"\n🏆 RESULT: {best:.1f}%")
if best > 60:
    print("✅ PROBLEM = TOO MANY CONFUSING CLASSES")
    print("   Solution: Use primitives or filter to 20+ samples")
else:
    print("❌ PROBLEM = FEATURES/MODEL BROKEN")
    print("   Something fundamentally wrong with setup")
