"""
PHASE 3 RETRAIN: Using grouped labels (40 classes instead of 791)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from phase2_temporal_vla_model import TemporalVLA

# --- CONFIG ---
DATA_FILE = 'data/processed/training_labels_final.json'
FEATURE_DIR = 'data/features_phase3'
MODEL_SAVE_PATH = 'models/temporal_vla_phase3_grouped.pt'
VOCAB_SAVE_PATH = 'data/processed/phase3_grouped_vocab.json'
BATCH_SIZE = 32  # Increased since we have fewer classes
EPOCHS = 30      # More epochs for better learning
LR = 1e-3        # Higher LR since fewer classes

class SmartCookingDataset(Dataset):
    def __init__(self, data, vocab=None):
        self.data = data
        
        # Build Vocab if not provided
        if vocab is None:
            labels = sorted(list(set(d['grouped_label'] for d in self.data)))
            self.vocab = {l: i for i, l in enumerate(labels)}
        else:
            self.vocab = vocab
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        # Load pre-computed tensor
        features = torch.load(item['feature_path'], weights_only=False)
        
        # Get label index
        label_str = item['grouped_label']
        label_idx = self.vocab.get(label_str, 0)
        
        return features, torch.tensor(label_idx)

def train_model(data, device):
    print("="*60)
    print("🧠 TRAINING GROUPED VLA MODEL")
    print("="*60)
    
    # 1. Prepare Data
    train_data, val_data = train_test_split(data, test_size=0.15, random_state=42)
    
    train_dataset = SmartCookingDataset(train_data)
    val_dataset = SmartCookingDataset(val_data, vocab=train_dataset.vocab)
    
    # Save Vocab
    with open(VOCAB_SAVE_PATH, 'w') as f:
        json.dump(train_dataset.vocab, f, indent=2)
    
    print(f"📚 Vocabulary size: {len(train_dataset.vocab)} grouped actions")
    print(f"📊 Training samples: {len(train_data)}")
    print(f"📊 Validation samples: {len(val_data)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 2. Init Model
    model = TemporalVLA(
        embedding_dim=512,
        hidden_dim=512,
        num_actions=len(train_dataset.vocab),
        num_heads=8,
        num_layers=6
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 3. Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        # TRAIN
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
        
        # VALIDATE
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                outputs = model(feats)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': train_dataset.vocab,
                'best_val_acc': best_acc
            }, MODEL_SAVE_PATH)
        
        scheduler.step()
    
    print("\n" + "="*60)
    print(f"🎉 TRAINING COMPLETE!")
    print(f"🏆 Best Validation Accuracy: {best_acc:.2f}%")
    print(f"💾 Model Saved: {MODEL_SAVE_PATH}")
    print("="*60)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Running on: {device}\n")
    
    # Load data
    with open(DATA_FILE) as f:
        data = json.load(f)
    
    print(f"📥 Loaded {len(data)} samples with grouped labels")
    
    train_model(data, device)
