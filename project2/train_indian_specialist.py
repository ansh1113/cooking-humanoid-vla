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
DATA_FILE = 'data/processed/verb_training_data.json'
FEATURE_DIR = 'data/features_phase3' # Uses existing cache!
MODEL_SAVE = 'models/indian_specialist_vla.pt'
VOCAB_SAVE = 'data/processed/indian_verb_vocab.json'
EPOCHS = 30
BATCH = 16
LR = 1e-3

class VerbDataset(Dataset):
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
        # We need to find the feature file. 
        # Check if we have a direct path, or reconstruct it
        if 'feature_path' in item and os.path.exists(item['feature_path']):
            path = item['feature_path']
        else:
            # Reconstruct path logic from previous scripts
            vid_id = item['video_url'].split('=')[-1]
            start = str(item['start']).replace('.', '_')
            path = f"{FEATURE_DIR}/{vid_id}_{start}.pt"
        
        # If still not found, return a zero tensor (safety)
        if not os.path.exists(path):
            return torch.zeros(30, 512), torch.tensor(0)
            
        feats = torch.load(path)
        label = self.vocab.get(item['verb_label'], 0)
        return feats, torch.tensor(label)

def train(device):
    with open(DATA_FILE) as f:
        full_data = json.load(f)
    
    # Filter missing files
    data = []
    for d in full_data:
        # Quick check if feature likely exists (simplified)
        if d.get('category') == 'western_supplement': continue # Skip western for now if features missing
        data.append(d)
        
    print(f"Training on {len(data)} Indian samples...")
    
    train_data, val_data = train_test_split(data, test_size=0.15, random_state=42)
    train_ds = VerbDataset(train_data)
    val_ds = VerbDataset(val_data)
    
    # Weights
    labels = [d['verb_label'] for d in train_data]
    weights = compute_class_weight('balanced', classes=np.array(['CUT','MIX','ADD','KNEAD']), y=labels)
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    
    model = TemporalVLA(512, 512, 4, 4, 2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH)
    
    print("🚀 Training Indian Specialist...")
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
            
        val_acc = 0
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
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.3f} | Train: {100*correct/total:.1f}% | Val: {100*v_corr/v_tot:.1f}%")
        
    torch.save(model.state_dict(), MODEL_SAVE)
    with open(VOCAB_SAVE, 'w') as f:
        json.dump(train_ds.vocab, f)

if __name__ == "__main__":
    train(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
