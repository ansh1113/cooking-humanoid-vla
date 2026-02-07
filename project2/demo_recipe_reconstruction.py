"""
DEMO: Reconstruct a recipe from existing clips
Shows the model predicting a sequence of cooking actions
"""
import torch
import cv2
import numpy as np
from PIL import Image
import open_clip
import json
import os
from phase2_temporal_vla_model import TemporalVLA

# Load model
checkpoint = torch.load('models/temporal_vla_phase2_best.pt', map_location='cpu', weights_only=False)
vocab = checkpoint['action_vocab']
idx_to_action = {v: k for k, v in vocab.items()}

print("="*70)
print("🎬 RECIPE RECONSTRUCTION DEMO")
print("="*70)
print(f"✅ Model: 78% accuracy on 21 cooking actions")
print()

device = torch.device('cpu')
model = TemporalVLA(512, 512, len(vocab), 8, 6).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("📥 Loading CLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
)

# Load labels to find sequences from same video
with open('data/processed/training_labels_final.json') as f:
    all_labels = json.load(f)

# Group by video
from collections import defaultdict
videos = defaultdict(list)
for item in all_labels:
    if 'clip_path' in item:
        videos[item['video_url']].append(item)

# Sort by start time
for url in videos:
    videos[url].sort(key=lambda x: x['start'])

# Find videos with 5+ clips
good_videos = {url: clips for url, clips in videos.items() if len(clips) >= 5}

print(f"\n📹 Found {len(good_videos)} videos with 5+ clips")
print("\nTop 5 videos:")
for i, (url, clips) in enumerate(list(good_videos.items())[:5], 1):
    title = clips[0].get('video_title', 'Unknown')[:50]
    print(f"{i}. {title}... ({len(clips)} clips)")

choice = int(input("\nSelect video (1-5): ")) - 1
selected_url = list(good_videos.keys())[choice]
selected_clips = good_videos[selected_url]

print(f"\n🎬 Reconstructing recipe from {len(selected_clips)} clips...")
print(f"📹 {selected_clips[0].get('video_title', 'Unknown')}")
print()

predictions = []

for i, clip_data in enumerate(selected_clips):
    clip_path = clip_data['clip_path']
    
    # Extract frames
    cap = cv2.VideoCapture(clip_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 1:
        continue
    
    indices = np.linspace(0, total_frames-1, 30, dtype=int)
    
    frames = []
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess(Image.fromarray(frame_rgb)))
    
    cap.release()
    
    if len(frames) < 30:
        frames += [frames[-1]] * (30 - len(frames))
    
    frames_tensor = torch.stack(frames).to(device)
    
    # Predict
    with torch.no_grad():
        emb = clip_model.encode_image(frames_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        logits = model(emb.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        
        # Top 3
        top3_probs, top3_indices = torch.topk(probs[0], 3)
    
    predicted = idx_to_action[top3_indices[0].item()]
    conf = top3_probs[0].item() * 100
    actual = clip_data.get('label', 'unknown')
    
    predictions.append({
        'time': f"{clip_data['start']:.0f}s",
        'predicted': predicted,
        'actual': actual,
        'confidence': conf,
        'correct': predicted.lower() in actual.lower() or actual.lower() in predicted.lower()
    })
    
    correct_mark = "✅" if predictions[-1]['correct'] else "❌"
    print(f"{i+1:2d}. [{clip_data['start']:5.0f}s] {correct_mark} Predicted: {predicted:25s} ({conf:5.1f}%)")
    print(f"                  Actual: {actual:25s}")

# Summary
print("\n" + "="*70)
print("📊 RESULTS:")
print("="*70)

correct = sum(1 for p in predictions if p['correct'])
total = len(predictions)
accuracy = 100 * correct / total

print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
print()

# Show recipe sequence
print("📋 PREDICTED RECIPE SEQUENCE:")
prev = None
step_num = 1
for p in predictions:
    if p['predicted'] != prev:
        print(f"{step_num:2d}. {p['predicted']}")
        prev = p['predicted']
        step_num += 1

print("="*70)
