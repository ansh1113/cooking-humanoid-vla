"""
PHASE 2: Train Temporal VLA
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

# Import model
from phase2_temporal_vla_model import TemporalVLA

class ActionDataset(Dataset):
    """Dataset for action sequences"""
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        embeddings = seq['embeddings']
        action_id = seq['action_id']
        
        # Pad or truncate to 30 frames
        if len(embeddings) < 30:
            # Pad with zeros
            padding = torch.zeros(30 - len(embeddings), embeddings.shape[1])
            embeddings = torch.cat([embeddings, padding], dim=0)
        elif len(embeddings) > 30:
            # Truncate
            embeddings = embeddings[:30]
        
        return embeddings, action_id

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for embeddings, labels in tqdm(dataloader, desc="Training"):
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(embeddings)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            logits = model(embeddings)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def main():
    print("="*70)
    print("🚀 PHASE 2: TRAINING TEMPORAL VLA")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load dataset
    print("\n📥 Loading dataset...")
    dataset = torch.load('data/processed/training_dataset_phase2_fixed.pt')
    
    train_dataset = ActionDataset(dataset['train'])
    val_dataset = ActionDataset(dataset['val'])
    test_dataset = ActionDataset(dataset['test'])
    
    print(f"   Train: {len(train_dataset)} sequences")
    print(f"   Val:   {len(val_dataset)} sequences")
    print(f"   Test:  {len(test_dataset)} sequences")
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Model
    print("\n🧠 Initializing model...")
    num_actions = len(dataset['action_vocab'])
    model = TemporalVLA(
        embedding_dim=512,
        hidden_dim=512,
        num_actions=num_actions,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    ).to(device)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training
    print("\n🔥 Starting training...")
    print("="*70)
    
    best_val_acc = 0
    best_epoch = 0
    num_epochs = 50
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'action_vocab': dataset['action_vocab']
            }, 'models/temporal_vla_phase2_best.pt')
            print(f"💾 Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    # Test on best model
    print("\n🧪 Testing best model...")
    checkpoint = torch.load('models/temporal_vla_phase2_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"\n📊 Final Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Acc:  {test_acc:.2f}%")
    
    print("\n" + "="*70)
    print("🎉 PHASE 2 COMPLETE!")
    print(f"🎯 Target was 70%+ accuracy")
    print(f"✅ Achieved: {test_acc:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()
