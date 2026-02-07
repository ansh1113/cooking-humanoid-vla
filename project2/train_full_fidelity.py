"""
🔥 FULL FIDELITY VLA TRAINING
Dataset: ~1,600 samples
Classes: 40 Fine-Grained Actions (No Merging)
Techniques: Focal Loss + Label Smoothing + Data Augmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import random

# --- CONFIG ---
FEATURE_DIR = 'data/features_golden40'
DATA_FILE = 'data/processed/golden_40_dataset.json'
MODEL_SAVE = 'models/full_fidelity_vla.pt'
VOCAB_SAVE = 'models/vocab_full.json'

# Hyperparameters for High Fidelity
BATCH_SIZE = 32
EPOCHS = 75          # More epochs needed for fine-grained
LR = 1e-4            # Lower LR for stability
DROPOUT = 0.4        # Moderate regularization
LABEL_SMOOTHING = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. FOCAL LOSS (The Secret Weapon) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=LABEL_SMOOTHING)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# --- 2. ARCHITECTURE (Standard Size) ---
class FullFidelityVLA(nn.Module):
    def __init__(self, num_classes, input_dim=512, hidden_dim=512, num_layers=6, nhead=8):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(1, 30, input_dim) * 0.02)
        
        # Standard Transformer (Not Small)
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=2048, 
            dropout=DROPOUT, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        x = x + self.pos_enc
        x = self.transformer(x)
        x = x.mean(dim=1) # Global Average Pooling
        return self.head(x)

# --- 3. DATASET ---
class CookingDataset(Dataset):
    def __init__(self, data, vocab, augment=False):
        self.data = data
        self.vocab = vocab
        self.augment = augment
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Flexible ID lookup
        if 'video_file' in item:
             vid_id = item['video_file'].replace('temp_golden_', '').replace('.mp4', '')
        else:
             vid_id = item['video_url'].split('=')[-1].split('&')[0]
             
        start = item['start']
        
        # Try exact float match, then loose match
        path = f"{FEATURE_DIR}/{vid_id}_{start:.2f}.pt"
        if not os.path.exists(path):
             path = f"{FEATURE_DIR}/{vid_id}_{start:.1f}0.pt"
             
        if os.path.exists(path):
            feats = torch.load(path, weights_only=True)
        else:
            feats = torch.zeros(30, 512) # Should be rare
            
        # Data Augmentation (Crucial for 40 classes)
        if self.augment:
            # Time Shift (Context Jitter)
            if random.random() < 0.5:
                shift = random.randint(-2, 2)
                feats = torch.roll(feats, shift, dims=0)
            # Feature Noise (Robustness)
            if random.random() < 0.2:
                feats = feats + (torch.randn_like(feats) * 0.01)
                
        label = self.vocab[item['action']]
        return feats, label

def main():
    print(f"🚀 TRAINING FULL FIDELITY VLA (No Shortcuts)")
    
    with open(DATA_FILE) as f:
        data = json.load(f)
        
    # Build Vocab
    actions = [d['action'] for d in data]
    counts = Counter(actions)
    vocab = {a: i for i, a in enumerate(sorted(counts.keys()))}
    
    print(f"🎯 Classes: {len(vocab)} (Fine-Grained)")
    
    # Stratified Split
    train_idx, val_idx = train_test_split(range(len(data)), test_size=0.15, stratify=actions, random_state=42)
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    
    print(f"📊 Train: {len(train_data)} | Val: {len(val_data)}")
    
    # Loaders
    train_loader = DataLoader(CookingDataset(train_data, vocab, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CookingDataset(val_data, vocab, augment=False), batch_size=BATCH_SIZE)
    
    # Model Setup
    model = FullFidelityVLA(len(vocab)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Use Focal Loss instead of standard CrossEntropy
    criterion = FocalLoss()
    
    print(f"\n🔥 STARTING TRAINING...")
    best_val = 0
    
    for epoch in range(EPOCHS):
        model.train()
        t_corr, t_tot = 0, 0
        train_losses = []
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            _, p = torch.max(out, 1)
            t_corr += (p == y).sum().item()
            t_tot += y.size(0)
            
        train_acc = 100 * t_corr / t_tot
        avg_loss = sum(train_losses) / len(train_losses)
        
        # Validation
        model.eval()
        v_corr, v_tot = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, p = torch.max(out, 1)
                v_corr += (p == y).sum().item()
                v_tot += y.size(0)
        
        val_acc = 100 * v_corr / v_tot
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}%", end="")
        
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'model': model.state_dict(), 'vocab': vocab, 'acc': val_acc}, MODEL_SAVE)
            print(" ✅")
        else:
            print()
            
    print(f"\n🏆 FINAL ACCURACY: {best_val:.1f}%")
    
    # Save Vocab
    with open(VOCAB_SAVE, 'w') as f:
        json.dump(vocab, f)

if __name__ == "__main__":
    main()
