"""
REAL VLA: DOMAIN ADAPTATION 🧠
Fine-tuning the Western Brain to understand Indian Physics.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from phase2_temporal_vla_model import TemporalVLA

# CONFIG
WESTERN_MODEL_PATH = 'models/temporal_vla_phase2_best.pt'
WESTERN_VOCAB_PATH = 'data/processed/action_vocab_phase2_fixed.json'
INDIAN_DATA_FILE = 'data/processed/verb_training_data.json'
FEATURE_DIR = 'data/features_phase3'

SAVE_PATH = 'models/indian_finetuned_vla.pt'
NEW_VOCAB_PATH = 'data/processed/indian_verb_vocab.json'

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4 # Low learning rate for fine-tuning

class IndianVerbDataset(Dataset):
    def __init__(self, data, vocab=None):
        self.data = data
        if vocab is None:
            self.vocab = {'CUT':0, 'MIX':1, 'ADD':2, 'KNEAD':3}
        else:
            self.vocab = vocab
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        if 'feature_path' in item and os.path.exists(item['feature_path']):
            path = item['feature_path']
        else:
            vid_id = item['video_url'].split('=')[-1]
            start = str(item['start']).replace('.', '_')
            path = f"{FEATURE_DIR}/{vid_id}_{start}.pt"
        
        if not os.path.exists(path):
            return torch.zeros(30, 512), torch.tensor(0)
            
        feats = torch.load(path)
        label = self.vocab.get(item['verb_label'], 0)
        return feats, torch.tensor(label)

def train_adaptation(device):
    print("🧠 Loading Western Brain...")
    
    # 1. Load Data
    with open(INDIAN_DATA_FILE) as f:
        full_data = json.load(f)
    # Filter for Indian/relevant data only
    data = [d for d in full_data if d.get('category') != 'western_supplement']
    print(f"📉 Fine-tuning on {len(data)} Indian samples...")
    
    # 2. Setup Dataset
    train_data, val_data = train_test_split(data, test_size=0.15, random_state=42)
    train_ds = IndianVerbDataset(train_data)
    val_ds = IndianVerbDataset(val_data)
    
    # Save the NEW vocabulary (4 verbs)
    with open(NEW_VOCAB_PATH, 'w') as f:
        json.dump(train_ds.vocab, f)

    # 3. Load Western Model
    checkpoint = torch.load(WESTERN_MODEL_PATH, map_location=device)
    with open(WESTERN_VOCAB_PATH) as f:
        old_vocab = json.load(f)
        
    # Initialize with OLD architecture (21 classes) to load weights
    model = TemporalVLA(512, 512, len(old_vocab), 8, 6).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 4. SURGERY: Freeze Brain, Replace Head
    # Freeze the Transformer
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the Head (Classifier)
    # New head has 4 outputs (CUT, MIX, ADD, KNEAD) instead of 21
    model.action_head = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 4) # 4 New Classes
    ).to(device)
    
    # 5. Training Setup
    # Class weights to fight the "MIX" bias
    labels = [d['verb_label'] for d in train_data]
    weights = compute_class_weight('balanced', classes=np.array(['CUT','MIX','ADD','KNEAD']), y=labels)
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    optimizer = optim.AdamW(model.action_head.parameters(), lr=LR) # Optimize HEAD ONLY
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    print("🚀 Starting Fine-Tuning (Head Only)...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(feats)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
        # Validation
        model.eval()
        v_corr = 0
        v_tot = 0
        with torch.no_grad():
            for f, l in val_loader:
                f, l = f.to(device), l.to(device)
                o = model(f)
                _, p = torch.max(o, 1)
                v_corr += (p == l).sum().item()
                v_tot += l.size(0)
        
        val_acc = 100*v_corr/v_tot
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.3f} | Train: {100*correct/total:.1f}% | Val: {val_acc:.1f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"💾 Saved Fine-Tuned Model: {SAVE_PATH} (Best Acc: {best_acc:.1f}%)")

if __name__ == "__main__":
    train_adaptation(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
