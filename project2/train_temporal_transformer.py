"""
Day 2: Train Temporal Transformer - The Brain
Learns to predict next visual state from sequence
Self-supervised: No labels needed!
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm

class TemporalTransformer(nn.Module):
    """Predicts next embedding from sequence of past embeddings"""
    def __init__(self, embedding_dim=512, hidden_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x):
        """
        x: (batch, seq_len, embedding_dim)
        Returns: (batch, embedding_dim) - predicted next embedding
        """
        # Process sequence
        features = self.transformer(x)
        
        # Use last timestep to predict next
        last_feature = features[:, -1, :]
        
        # Predict next embedding
        next_embedding = self.predictor(last_feature)
        
        # Normalize (embeddings should be unit vectors)
        next_embedding = next_embedding / next_embedding.norm(dim=-1, keepdim=True)
        
        return next_embedding

class VideoSequenceDataset(Dataset):
    """Dataset of video sequences for self-supervised learning"""
    def __init__(self, embeddings_dir, sequence_length=30):
        self.embeddings_dir = Path(embeddings_dir)
        self.sequence_length = sequence_length
        
        # Load all embedding files
        self.video_files = sorted(list(self.embeddings_dir.glob('*.pt')))
        
        # Build index of valid sequences
        self.sequences = []
        for video_file in self.video_files:
            embeddings = torch.load(video_file)
            num_frames = embeddings.shape[0]
            
            # Create overlapping sequences
            for i in range(0, num_frames - sequence_length - 1, sequence_length // 2):
                self.sequences.append({
                    'video_file': video_file,
                    'start_idx': i,
                    'end_idx': i + sequence_length
                })
        
        print(f"   Created {len(self.sequences)} training sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        
        # Load embeddings for this video
        embeddings = torch.load(seq_info['video_file'])
        
        # Extract sequence
        start = seq_info['start_idx']
        end = seq_info['end_idx']
        
        # Input: frames[start:end], Target: frames[end]
        input_seq = embeddings[start:end]
        target = embeddings[end]
        
        return input_seq, target

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for input_seq, target in tqdm(dataloader, desc="Training"):
        input_seq = input_seq.to(device)
        target = target.to(device)
        
        # Forward pass
        predicted = model(input_seq)
        
        # Loss: Cosine similarity (want predicted close to target)
        # 1 - cosine = distance, minimize distance
        loss = 1 - (predicted * target).sum(dim=1).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_seq, target in tqdm(dataloader, desc="Validating"):
            input_seq = input_seq.to(device)
            target = target.to(device)
            
            predicted = model(input_seq)
            loss = 1 - (predicted * target).sum(dim=1).mean()
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    print("="*70)
    print("🧠 DAY 2: TRAINING TEMPORAL TRANSFORMER")
    print("="*70)
    
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Create dataset
    print("\n📊 Creating dataset...")
    dataset = VideoSequenceDataset('data/processed/embeddings', sequence_length=30)
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"   Train: {train_size}, Val: {val_size}")
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model
    print("\n🏗️  Building model...")
    model = TemporalTransformer(
        embedding_dim=512,
        hidden_dim=256,
        num_layers=4,
        num_heads=8
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    print("\n🎓 Training...")
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):
        print(f"\nEpoch {epoch+1}/50")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'models/temporal_vla_brain.pt')
            print(f"   ✅ New best! Val Loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
    
    print("\n" + "="*70)
    print(f"✅ Training complete! Best Val Loss: {best_val_loss:.4f}")
    print(f"💾 Model saved: models/temporal_vla_brain.pt")
    print("="*70)

if __name__ == "__main__":
    main()
