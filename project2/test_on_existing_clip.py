"""
Test on an existing video clip from our dataset
"""
import torch
import cv2
import numpy as np
from PIL import Image
import open_clip
import json
import os
from phase2_temporal_vla_model import TemporalVLA

# Load model and vocab
checkpoint = torch.load('models/temporal_vla_phase2_best.pt', map_location='cpu', weights_only=False)
vocab = checkpoint['action_vocab']
idx_to_action = {v: k for k, v in vocab.items()}

print("✅ Loaded Phase 2 model (78% accuracy)")
print(f"📚 Knows {len(vocab)} actions")

device = torch.device('cpu')

model = TemporalVLA(
    embedding_dim=512,
    hidden_dim=512,
    num_actions=len(vocab),
    num_heads=8,
    num_layers=6
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("📥 Loading CLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', 
    pretrained='laion2b_s34b_b79k',
    device=device
)

# Get a random clip from training clips
clips_dir = 'data/training_clips'
all_clips = [f for f in os.listdir(clips_dir) if f.endswith('.mp4')][:10]

print(f"\n🎬 Testing on 10 random clips from dataset:")
print("="*70)

for clip_file in all_clips:
    clip_path = os.path.join(clips_dir, clip_file)
    
    # Extract frames
    cap = cv2.VideoCapture(clip_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 1:
        continue
    
    indices = np.linspace(0, total_frames-1, 30, dtype=int)
    
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(preprocess(pil_img))
    
    cap.release()
    
    if len(frames) < 30:
        frames += [frames[-1]] * (30 - len(frames))
    
    frames = torch.stack(frames).to(device)
    
    # Predict
    with torch.no_grad():
        embeddings = clip_model.encode_image(frames)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.unsqueeze(0)
        
        logits = model(embeddings)
        probs = torch.softmax(logits, dim=-1)
        
        # Top 3
        top3_probs, top3_indices = torch.topk(probs[0], 3)
    
    print(f"\n📹 {clip_file}:")
    for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices), 1):
        action = idx_to_action[idx.item()]
        confidence = prob.item() * 100
        marker = "✅" if i == 1 else "  "
        print(f"   {marker} {i}. {action:30s} ({confidence:5.1f}%)")

print("\n" + "="*70)
print("✅ Testing complete!")
