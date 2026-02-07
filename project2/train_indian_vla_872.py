"""
TRAIN INDIAN VLA WITH 872 LABELS
Small model (1.5M params) + dropout=0.1 + weight_decay=1e-4
Target: 70-80% accuracy
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

print("="*70)
print("🧠 TRAINING INDIAN COOKING VLA - 872 LABELS")
print("="*70)

FEATURE_DIR = 'data/features_golden40'
DATA_FILE = 'data/processed/golden_40_ready.json'
MODEL_SAVE = 'models/indian_vla_best.pt'
VOCAB_SAVE = 'models/vocab_actions.json'
HISTORY_SAVE = 'models/training_history.json'

# Hyperparameters (optimized for 872 samples)
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4  # Light regularization
DROPOUT = 0.1        # Light dropout (like Phase 2)
PATIENCE = 15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")

class SmallTemporalVLA(nn.Module):
    """Small VLA - 3 layers, 256 hidden"""
    def __init__(self, input_dim=512, hidden_dim=512, num_actions=39, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 30, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
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

class IndianCookingDataset(Dataset):
    def __init__(self, data, vocab, augment=False):
        self.data = data
        self.vocab = vocab
        self.augment = augment
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        url = item.get('video_url', item.get('url', ''))
        video_id = url.split('=')[-1].split('&')[0]
        start = item['start']
        
        feature_path = f"{FEATURE_DIR}/{video_id}_{start:.2f}.pt"
        
        if os.path.exists(feature_path):
            features = torch.load(feature_path, weights_only=True)
        else:
            features = torch.zeros(30, 512)
        
        if self.augment and random.random() < 0.3:
            shift = random.randint(-1, 1)
            features = torch.roll(features, shift, dims=0)
        
        action = item['action']
        label = self.vocab.get(action, 0)
        
        return features, label

def train_model():
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
    
    # Smart split
    single_sample = {a for a, c in action_counts.items() if c == 1}
    multi_data = [d for d in data if d['action'] not in single_sample]
    single_data = [d for d in data if d['action'] in single_sample]
    
    multi_labels = [vocab[d['action']] for d in multi_data]
    train_multi, val_multi = train_test_split(multi_data, test_size=0.15, stratify=multi_labels, random_state=42)
    
    train_data = train_multi + single_data
    val_data = val_multi
    
    print(f"📊 Train: {len(train_data)} | Val: {len(val_data)}")
    
    # Class weights
    train_labels = [vocab[d['action']] for d in train_data]
    unique = np.unique(train_labels)
    weights = compute_class_weight('balanced', classes=unique, y=train_labels)
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    # Datasets
    train_dataset = IndianCookingDataset(train_data, vocab, augment=True)
    val_dataset = IndianCookingDataset(val_data, vocab, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Model
    model = SmallTemporalVLA(hidden_dim=512, num_actions=num_actions, dropout=DROPOUT).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️  Model: {total_params:,} parameters")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, verbose=True)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    print(f"\n🚀 Training (patience={PATIENCE})...")
    print("="*70)
    
    best_val = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
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
        scheduler.step(val_acc)
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}%", end="")
        
        if val_acc > best_val:
            best_val = val_acc
            patience_counter = 0
            os.makedirs('models', exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'vocab': vocab, 'num_actions': num_actions, 'val_acc': val_acc}, MODEL_SAVE)
            print(" ✅")
        else:
            patience_counter += 1
            print(f" ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n⏹️  Early stop")
                break
    
    print("\n" + "="*70)
    print(f"✅ Best Val: {best_val:.2f}%")
    
    with open(VOCAB_SAVE, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    with open(HISTORY_SAVE, 'w') as f:
        json.dump(history, f, indent=2)
    
    return best_val

if __name__ == "__main__":
    acc = train_model()
    print(f"\n🎯 Final: {acc:.2f}%")
    print("🎉 Indian Cooking Brain READY!")
