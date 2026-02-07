"""
Train on ONLY Indian cooking data - focused and clean
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from sklearn.model_selection import train_test_split
from phase2_temporal_vla_model import TemporalVLA

# Load all data
with open('data/processed/training_labels_final.json') as f:
    all_data = json.load(f)

# Filter to ONLY Indian + unknown (which includes Paneer test)
indian_data = [d for d in all_data if d.get('category') in ['indian_nonveg', 'indian_vegetarian', 'unknown']]

print(f"📊 Total samples: {len(all_data)}")
print(f"🍛 Indian samples: {len(indian_data)}")

# Group by action frequency
from collections import Counter
action_counts = Counter(d['label'] for d in indian_data)

# Keep actions with 2+ samples, group rest as "rare_action"
MIN_SAMPLES = 2
common_actions = {action for action, count in action_counts.items() if count >= MIN_SAMPLES}

print(f"✅ Actions with {MIN_SAMPLES}+ samples: {len(common_actions)}")

for item in indian_data:
    item['train_label'] = item['label'] if item['label'] in common_actions else 'rare_action'

# Check distribution
train_labels = Counter(d['train_label'] for d in indian_data)
print(f"\n📈 Final distribution (top 20):")
for i, (action, count) in enumerate(train_labels.most_common(20), 1):
    print(f"   {i:2d}. {action:40s}: {count:3d}")

print(f"\n✅ Total unique actions: {len(train_labels)}")

# Save for training
class IndianCookingDataset(Dataset):
    def __init__(self, data, vocab=None):
        self.data = data
        if vocab is None:
            labels = sorted(list(set(d['train_label'] for d in self.data)))
            self.vocab = {l: i for i, l in enumerate(labels)}
        else:
            self.vocab = vocab
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.load(item['feature_path'], weights_only=False)
        label_idx = self.vocab[item['train_label']]
        return features, torch.tensor(label_idx)

# Train/val split
train_data, val_data = train_test_split(indian_data, test_size=0.2, random_state=42)

train_dataset = IndianCookingDataset(train_data)
val_dataset = IndianCookingDataset(val_data, vocab=train_dataset.vocab)

print(f"\n📊 Training samples: {len(train_data)}")
print(f"📊 Validation samples: {len(val_data)}")
print(f"📚 Vocabulary size: {len(train_dataset.vocab)}")

# Save vocab
with open('data/processed/indian_vocab.json', 'w') as f:
    json.dump(train_dataset.vocab, f, indent=2)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = TemporalVLA(
    embedding_dim=512,
    hidden_dim=512,
    num_actions=len(train_dataset.vocab),
    num_heads=8,
    num_layers=6
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

best_acc = 0.0

for epoch in range(50):
    # Train
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
    
    # Validate
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
    
    print(f"Epoch {epoch+1:2d}/50 | Loss: {total_loss/len(train_loader):.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': train_dataset.vocab,
            'best_val_acc': best_acc
        }, 'models/indian_cooking_vla.pt')
    
    scheduler.step()

print(f"\n🎉 Best Accuracy: {best_acc:.2f}%")
