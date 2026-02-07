"""
PHASE 3: TRAINING THE "SUPER-BRAIN"
1. Caches CLIP features from video clips (MP4 -> Tensor)
2. Trains Temporal VLA on 900+ fine-grained classes
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import os
import open_clip
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from phase2_temporal_vla_model import TemporalVLA

# --- CONFIG ---
DATA_FILE = 'data/processed/training_labels_with_clips.json'
FEATURE_DIR = 'data/features_phase3'
MODEL_SAVE_PATH = 'models/temporal_vla_phase3_smart.pt'
VOCAB_SAVE_PATH = 'data/processed/phase3_action_vocab.json'
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

# ============================================================================
# 1. FEATURE CACHING (The Speed Hack)
# ============================================================================

def extract_clip_features(device):
    print("="*60)
    print("📼 STEP 1: CACHING VIDEO FEATURES")
    print("="*60)
    
    if not os.path.exists(FEATURE_DIR):
        os.makedirs(FEATURE_DIR)
        
    with open(DATA_FILE) as f:
        raw_data = json.load(f)
        
    print(f"📥 Loading CLIP model on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    
    valid_samples = []
    
    print(f"🔄 Processing {len(raw_data)} clips...")
    for item in tqdm(raw_data):
        clip_path = item['clip_path']
        video_id = item['video_url'].split('=')[-1]
        start_time = str(item['start']).replace('.', '_')
        save_name = f"{video_id}_{start_time}.pt"
        save_path = os.path.join(FEATURE_DIR, save_name)
        
        # Skip if already cached
        if os.path.exists(save_path):
            try:
                # Verify file integrity
                _ = torch.load(save_path)
                item['feature_path'] = save_path
                valid_samples.append(item)
                continue
            except:
                pass # Re-process corrupt files

        # Extract 30 frames
        cap = cv2.VideoCapture(clip_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Safety check for empty videos
        if total_frames < 1:
            continue
        
        # Sample 30 frames evenly
        indices = np.linspace(0, total_frames-1, 30, dtype=int)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                frames.append(preprocess(pil_img))
        cap.release()
        
        # Pad if short
        if len(frames) == 0:
            continue
        if len(frames) < 30:
            frames += [frames[-1]] * (30 - len(frames))
        
        # Encode with CLIP
        img_tensor = torch.stack(frames).to(device)
        with torch.no_grad():
            emb = model.encode_image(img_tensor) # [30, 512]
            emb = emb / emb.norm(dim=-1, keepdim=True)
            
        # Save Tensor
        torch.save(emb.cpu(), save_path)
        item['feature_path'] = save_path
        valid_samples.append(item)
        
    print(f"✅ Cached {len(valid_samples)} valid samples.")
    return valid_samples

# ============================================================================
# 2. TRAINING DATASET
# ============================================================================

class SmartCookingDataset(Dataset):
    def __init__(self, data, vocab=None):
        self.data = data
        
        # Build Vocab if not provided
        if vocab is None:
            labels = sorted(list(set(d['label'] for d in self.data)))
            self.vocab = {l: i for i, l in enumerate(labels)}
        else:
            self.vocab = vocab
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        # Load pre-computed tensor
        features = torch.load(item['feature_path']) # [30, 512]
        
        # Get label index
        label_str = item['label']
        label_idx = self.vocab.get(label_str, 0) # Default 0 if unseen
        
        return features, torch.tensor(label_idx)

# ============================================================================
# 3. TRAINING LOOP
# ============================================================================

def train_model(data, device):
    print("\n" + "="*60)
    print("🧠 STEP 2: TRAINING TEMPORAL VLA")
    print("="*60)
    
    # 1. Prepare Data
    train_data, val_data = train_test_split(data, test_size=0.15, random_state=42)
    
    train_dataset = SmartCookingDataset(train_data)
    val_dataset = SmartCookingDataset(val_data, vocab=train_dataset.vocab)
    
    # Save Vocab for Inference Script
    with open(VOCAB_SAVE_PATH, 'w') as f:
        json.dump(train_dataset.vocab, f, indent=2)
    print(f"📚 Vocabulary size: {len(train_dataset.vocab)} actions")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 2. Init Model (From Scratch)
    model = TemporalVLA(
        embedding_dim=512,
        hidden_dim=512,
        num_actions=len(train_dataset.vocab), # 900+ classes
        num_heads=8,
        num_layers=6
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # 3. Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        # TRAIN
        model.train()
        total_loss = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(feats) # [B, num_classes]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
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
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': train_dataset.vocab
            }, MODEL_SAVE_PATH)
            
    print("\n" + "="*60)
    print(f"🎉 TRAINING COMPLETE! Best Accuracy: {best_acc:.2f}%")
    print(f"💾 Model Saved: {MODEL_SAVE_PATH}")
    print("="*60)

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs('models', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Running on: {device}")
    
    # Run Pipeline
    valid_data = extract_clip_features(device)
    train_model(valid_data, device)
