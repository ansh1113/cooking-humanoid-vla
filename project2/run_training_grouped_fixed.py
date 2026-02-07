"""
PHASE 3 FIX: TRAINING WITH CLASS WEIGHTS + SINGLETON FILTERING + NUMPY FIX
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from phase2_temporal_vla_model import TemporalVLA

# --- CONFIG ---
DATA_FILE = 'data/processed/training_labels_final.json'
MODEL_SAVE_PATH = 'models/temporal_vla_phase3_grouped_weighted.pt'
VOCAB_SAVE_PATH = 'data/processed/phase3_grouped_vocab.json'
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

# ============================================================================
# DATASET
# ============================================================================

class GroupedCookingDataset(Dataset):
    def __init__(self, data, vocab=None):
        self.data = data
        if vocab is None:
            labels = sorted(list(set(d['grouped_label'] for d in self.data)))
            self.vocab = {l: i for i, l in enumerate(labels)}
        else:
            self.vocab = vocab
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.load(item['feature_path']) 
        label_str = item['grouped_label']
        label_idx = self.vocab.get(label_str, 0)
        return features, torch.tensor(label_idx)

# ============================================================================
# TRAINING
# ============================================================================

def train_model(device):
    print(f"🖥️  Running on: {device}")
    
    # 1. Load Data
    with open(DATA_FILE) as f:
        full_data = json.load(f)
    
    # Filter only entries with valid features
    valid_data = [d for d in full_data if 'feature_path' in d and 'grouped_label' in d]
    
    # REMOVE SINGLETONS
    label_counts = Counter([d['grouped_label'] for d in valid_data])
    valid_data = [d for d in valid_data if label_counts[d['grouped_label']] >= 2]
    print(f"📥 Loaded {len(valid_data)} samples (filtered singletons)")
    
    train_data, val_data = train_test_split(valid_data, test_size=0.15, random_state=42, stratify=[d['grouped_label'] for d in valid_data])
    
    train_dataset = GroupedCookingDataset(train_data)
    val_dataset = GroupedCookingDataset(val_data, vocab=train_dataset.vocab)
    
    # 2. CALCULATE CLASS WEIGHTS (FIXED: Added np.array wrapper)
    print("⚖️  Calculating Class Weights...")
    all_train_labels = [d['grouped_label'] for d in train_data]
    unique_classes = sorted(list(set(all_train_labels)))
    
    # --- THE FIX IS HERE ---
    class_weights = compute_class_weight('balanced', classes=np.array(unique_classes), y=all_train_labels)
    # -----------------------
    
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"   Max Weight: {weights_tensor.max().item():.2f}")
    print(f"   Min Weight: {weights_tensor.min().item():.2f}")

    # Save Vocab
    with open(VOCAB_SAVE_PATH, 'w') as f:
        json.dump(train_dataset.vocab, f, indent=2)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 3. Model
    model = TemporalVLA(
        embedding_dim=512,
        hidden_dim=512,
        num_actions=len(train_dataset.vocab),
        num_heads=4,
        num_layers=2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)
    
    best_acc = 0.0
    
    print("\n🧠 TRAINING GROUPED VLA (WEIGHTED)")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
            
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                outputs = model(feats)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = 100 * correct_val / total_val
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': train_dataset.vocab
            }, MODEL_SAVE_PATH)
            
    print(f"\n🏆 Best Validation Accuracy: {best_acc:.2f}%")
    print(f"💾 Model Saved: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(device)
