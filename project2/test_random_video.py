"""
Test the Phase 2 model (78% accuracy) on a random YouTube video
"""
import torch
import cv2
import numpy as np
from PIL import Image
import open_clip
import json
from phase2_temporal_vla_model import TemporalVLA
from stage1_generate_plan import download_youtube_video

# Load model and vocab
checkpoint = torch.load('models/temporal_vla_phase2_best.pt')
vocab = checkpoint['vocab']
idx_to_action = {v: k for k, v in vocab.items()}

print("✅ Loaded model with 78.15% accuracy")
print(f"📚 Actions: {len(vocab)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = TemporalVLA(
    embedding_dim=512,
    hidden_dim=512,
    num_actions=len(vocab),
    num_heads=8,
    num_layers=6
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load CLIP
print("📥 Loading CLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', 
    pretrained='laion2b_s34b_b79k',
    device=device
)

def extract_frames(video_path, num_frames=30):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 1:
        return None
    
    # Sample evenly
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
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
    
    if len(frames) < num_frames:
        # Pad
        frames += [frames[-1]] * (num_frames - len(frames))
    
    return torch.stack(frames)

def predict_action(video_path):
    """Predict action from video"""
    print(f"\n🎬 Processing video...")
    
    # Extract frames
    frames = extract_frames(video_path)
    if frames is None:
        print("❌ Failed to extract frames")
        return
    
    frames = frames.to(device)
    
    # Get CLIP embeddings
    print("🔍 Extracting CLIP features...")
    with torch.no_grad():
        embeddings = clip_model.encode_image(frames)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    
    # Predict
    print("🧠 Predicting action...")
    embeddings = embeddings.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        logits = model(embeddings)
        probs = torch.softmax(logits, dim=-1)
        
        # Top 5 predictions
        top5_probs, top5_indices = torch.topk(probs[0], 5)
    
    print("\n" + "="*60)
    print("🎯 PREDICTIONS:")
    print("="*60)
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices), 1):
        action = idx_to_action[idx.item()]
        confidence = prob.item() * 100
        print(f"{i}. {action:30s} ({confidence:5.2f}%)")
    print("="*60)

# Test video URL
print("\n" + "="*60)
print("🧪 TESTING PHASE 2 MODEL (78% ACCURACY)")
print("="*60)

# Example Western cooking videos to test:
test_videos = [
    ("https://www.youtube.com/watch?v=Upqp21Dm5vg", "Gordon Ramsay scrambled eggs"),
    ("https://www.youtube.com/watch?v=PUP7U5vTMM0", "How to dice an onion"),
    ("https://www.youtube.com/watch?v=1Z1v8FrLhzw", "Pasta cooking tutorial"),
]

print("\nSuggested test videos:")
for i, (url, desc) in enumerate(test_videos, 1):
    print(f"{i}. {desc}")
    print(f"   {url}")

print("\n" + "-"*60)
choice = input("Enter video number (1-3) or paste your own YouTube URL: ").strip()

if choice in ['1', '2', '3']:
    video_url = test_videos[int(choice)-1][0]
    desc = test_videos[int(choice)-1][1]
    print(f"\n📹 Testing on: {desc}")
else:
    video_url = choice
    print(f"\n📹 Testing on: {video_url}")

# Download
print("\n📥 Downloading video...")
result = download_youtube_video(video_url)
if isinstance(result, tuple):
    video_path, title = result
else:
    video_path = result
    title = "Unknown"

print(f"✅ {title}")

# Predict
predict_action(video_path)

# Cleanup
import os
if os.path.exists(video_path):
    os.remove(video_path)

print("\n✅ Test complete!")
