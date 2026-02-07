"""
TRAIN 40 CLASSES WITH EXACT PHASE 2 SETTINGS
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

FEATURE_DIR = 'data/features_golden40'
DATA_FILE = 'data/processed/golden_40_ready.json'
MODEL_SAVE = 'models/indian_40_phase2_settings.pt'

BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 0.01
DROPOUT = 0.1

device = torch.device('cuda')
print(f"🖥️  Device: {device}")

class TemporalVLA(nn.Module):
    def __init__(self, hidden_dim=512, num_actions=40, num_layers=6, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(512, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 30, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

class CookingDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        url = item.get('video_url', '')
        video_id = url.split('=')[-1].split('&')[0]
        start = item['start']
        
        feature_path = f"{FEATURE_DIR}/{video_id}_{start:.2f}.pt"
        features = torch.load(feature_path, weights_only=True) if os.path.exists(feature_path) else torch.zeros(30, 512)
        
        action = item['action']
        label = self.vocab.get(action, 0)
        
        return features, label

print("\n📋 Loading data...")
with open(DATA_FILE) as f:
    data = json.load(f)

print(f"✅ {len(data)} segments")

actions = [d['action'] for d in data]
action_counts = Counter(actions)
vocab = {action: idx for idx, action in enumerate(sorted(action_counts.keys()))}
num_actions = len(vocab)

print(f"🎯 {num_actions} actions")
print(f"\n📊 Top 10:")
for action, count in action_counts.most_common(10):
    print(f"   {count:3d}x {action}")

# Split (handle single-sample actions)
single_sample = {a for a, c in action_counts.items() if c == 1}
multi_data = [d for d in data if d['action'] not in single_sample]
single_data = [d for d in data if d['action'] in single_sample]

if len(multi_data) > 0:
    multi_labels = [vocab[d['action']] for d in multi_data]
    train_multi, val_multi = train_test_split(multi_data, test_size=0.15, stratify=multi_labels, random_state=42)
else:
    train_multi, val_multi = [], []

train_data = train_multi + single_data
val_data = val_multi

print(f"📊 Train: {len(train_data)} | Val: {len(val_data)}")

# Datasets
train_dataset = CookingDataset(train_data, vocab)
val_dataset = CookingDataset(val_data, vocab)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = TemporalVLA(hidden_dim=512, num_actions=num_actions, num_layers=6, dropout=DROPOUT).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"🏗️  Model: {total_params:,} parameters")

# EXACT PHASE 2 SETTINGS
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f"\n🚀 Training (PHASE 2 settings)...")
print("="*70)

best_val_acc = 0
best_epoch = 0

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    
    for feat, lab in train_loader:
        feat, lab = feat.to(device), lab.to(device)
        optimizer.zero_grad()
        out = model(feat)
        loss = criterion(out, lab)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, pred = torch.max(out, 1)
        train_correct += (pred == lab).sum().item()
        train_total += lab.size(0)
    
    train_acc = 100 * train_correct / train_total
    avg_loss = train_loss / len(train_loader)
    
    # Val
    model.eval()
    val_correct, val_total = 0, 0
    
    with torch.no_grad():
        for feat, lab in val_loader:
            feat, lab = feat.to(device), lab.to(device)
            out = model(feat)
            _, pred = torch.max(out, 1)
            val_correct += (pred == lab).sum().item()
            val_total += lab.size(0)
    
    val_acc = 100 * val_correct / val_total
    scheduler.step()
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}%", end="")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save({'model_state_dict': model.state_dict(), 'vocab': vocab, 'num_actions': num_actions, 'val_acc': val_acc}, MODEL_SAVE)
        print(" ✅")
    else:
        print()

print("\n" + "="*70)
print(f"✅ Best Val: {best_val_acc:.2f}% (Epoch {best_epoch})")
print(f"💾 Saved: {MODEL_SAVE}")

