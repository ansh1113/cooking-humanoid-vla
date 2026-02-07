"""
PHASE 2: TRAINING THE INDIAN COOKING VLA
Temporal Vision-Language-Action Model with Anti-Overfitting Measures
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
print("🧠 TRAINING INDIAN COOKING VLA")
print("="*70)

# ============================================================================
# CONFIG
# ============================================================================
FEATURE_DIR = 'data/features_golden40'
DATA_FILE = 'data/processed/golden_40_dataset.json'
MODEL_SAVE = 'models/indian_vla_best.pt'
VOCAB_SAVE = 'models/vocab_actions.json'
HISTORY_SAVE = 'models/training_history.json'

# Hyperparameters (with aggressive regularization)
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-2  # Strong regularization
DROPOUT = 0.5        # Aggressive dropout
PATIENCE = 10        # Early stopping patience

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class TemporalVLA(nn.Module):
    """
    6-Layer Transformer with Aggressive Regularization
    Input: 30×512 CLIP features
    Output: action classes
    """
    def __init__(self, input_dim=512, hidden_dim=512, num_actions=34, 
                 num_layers=6, num_heads=8, dropout=0.5):
        super().__init__()
        
        # Positional encoding for temporal awareness
        self.pos_encoding = nn.Parameter(torch.randn(1, 30, input_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
    def forward(self, x):
        # x shape: (batch, 30, 512)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

# ============================================================================
# DATASET
# ============================================================================
class IndianCookingDataset(Dataset):
    def __init__(self, data, vocab, augment=False):
        self.data = data
        self.vocab = vocab
        self.augment = augment
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load features
        url = item.get('video_url', item.get('url', ''))
        video_id = url.split('=')[-1]
        start = item['start']
        
        feature_path = f"{FEATURE_DIR}/{video_id}_{start:.2f}.pt"
        
        if not os.path.exists(feature_path):
            features = torch.zeros(30, 512)
        else:
            features = torch.load(feature_path)
        
        # Time jitter augmentation (training only)
        if self.augment and random.random() < 0.5:
            shift = random.randint(-2, 2)
            features = torch.roll(features, shift, dims=0)
        
        # Get label
        action = item['action']
        label = self.vocab.get(action, 0)
        
        return features, label

# ============================================================================
# TRAINING
# ============================================================================
def train_model():
    print("\n📋 Loading data...")
    with open(DATA_FILE) as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} segments")
    
    # Build vocabulary
    actions = [d['action'] for d in data]
    action_counts = Counter(actions)
    vocab = {action: idx for idx, action in enumerate(sorted(action_counts.keys()))}
    num_actions = len(vocab)
    
    print(f"🎯 Actions: {num_actions}")
    print(f"\n📊 Distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"   {count:3d}x {action}")
    
    # Smart split: stratified for actions with 2+ samples, random for single samples
    print("\n🔀 Splitting data (smart stratified)...")
    
    # Separate single-sample actions
    single_sample_actions = {action for action, count in action_counts.items() if count == 1}
    
    multi_data = [d for d in data if d['action'] not in single_sample_actions]
    single_data = [d for d in data if d['action'] in single_sample_actions]
    
    print(f"   Multi-sample actions: {len(multi_data)} segments")
    print(f"   Single-sample actions: {len(single_data)} segments (all go to train)")
    
    # Stratified split for multi-sample actions
    if len(multi_data) > 0:
        multi_labels = [vocab[d['action']] for d in multi_data]
        train_multi, val_multi = train_test_split(
            multi_data,
            test_size=0.15,
            stratify=multi_labels,
            random_state=42
        )
    else:
        train_multi, val_multi = [], []
    
    # All single-sample actions go to training
    train_data = train_multi + single_data
    val_data = val_multi
    
    print(f"   Final - Train: {len(train_data)} | Val: {len(val_data)}")
    
    # Compute class weights
    train_labels = [vocab[d['action']] for d in train_data]
    unique_labels = np.unique(train_labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print(f"\n⚖️  Class weighting enabled")
    
    # Datasets
    train_dataset = IndianCookingDataset(train_data, vocab, augment=True)
    val_dataset = IndianCookingDataset(val_data, vocab, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Model
    print(f"\n🏗️  Building model...")
    model = TemporalVLA(
        input_dim=512,
        hidden_dim=512,
        num_actions=num_actions,
        num_layers=6,
        num_heads=8,
        dropout=DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )
    
    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    print(f"\n🚀 Training for {EPOCHS} epochs (early stop patience={PATIENCE})...")
    print("="*70)
    
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        # TRAINING
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = 100 * train_correct / train_total
        avg_loss = train_loss / len(train_loader)
        
        # VALIDATION
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # History
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train: {train_acc:5.1f}% | "
              f"Val: {val_acc:5.1f}%", end="")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'num_actions': num_actions,
                'val_acc': val_acc
            }, MODEL_SAVE)
            
            print(" ✅ (Best!)")
        else:
            patience_counter += 1
            print(f" (patience: {patience_counter}/{PATIENCE})")
            
            if patience_counter >= PATIENCE:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved: {MODEL_SAVE}")
    
    # Save vocab and history
    with open(VOCAB_SAVE, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    with open(HISTORY_SAVE, 'w') as f:
        json.dump(history, f, indent=2)
    
    return best_val_acc, vocab

if __name__ == "__main__":
    best_acc, vocab = train_model()
    print(f"\n🎯 Final: {best_acc:.2f}% validation accuracy")
    print(f"🎉 Indian Cooking Brain READY!")
