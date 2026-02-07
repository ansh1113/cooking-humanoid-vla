"""
Test Phase 2 model on a FULL cooking video
Extracts multiple clips and predicts the entire recipe sequence
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
checkpoint = torch.load('models/temporal_vla_phase2_best.pt', map_location='cpu', weights_only=False)
vocab = checkpoint['action_vocab']  # FIXED: It's 'action_vocab' not 'vocab'
idx_to_action = {v: k for k, v in vocab.items()}

print("✅ Loaded Phase 2 model (78% accuracy)")
print(f"📚 Knows {len(vocab)} actions")

device = torch.device('cpu')

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

def extract_clips_from_video(video_path, clip_duration=5, overlap=2):
    """Extract overlapping clips from video"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📹 Video: {duration:.1f}s, {fps:.1f} fps, {total_frames} frames")
    
    clips = []
    stride = clip_duration - overlap
    num_clips = int((duration - clip_duration) / stride) + 1
    
    print(f"📊 Extracting {num_clips} clips ({clip_duration}s each, {overlap}s overlap)...")
    
    for i in range(num_clips):
        start_time = i * stride
        end_time = start_time + clip_duration
        
        if end_time > duration:
            break
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        frame_indices = np.linspace(start_frame, end_frame-1, 30, dtype=int)
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in frame_indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                frames.append(preprocess(pil_img))
        
        if len(frames) == 30:
            clips.append({
                'frames': torch.stack(frames),
                'start_time': start_time,
                'end_time': end_time
            })
    
    cap.release()
    return clips

def predict_recipe_sequence(video_path):
    """Predict the full recipe from a video"""
    print("\n🎬 Analyzing full video...")
    
    clips = extract_clips_from_video(video_path, clip_duration=5, overlap=2)
    
    if not clips:
        print("❌ No clips extracted!")
        return
    
    print(f"\n🧠 Predicting actions for {len(clips)} clips...")
    
    predictions = []
    
    for i, clip in enumerate(clips):
        frames = clip['frames'].to(device)
        
        with torch.no_grad():
            embeddings = clip_model.encode_image(frames)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.unsqueeze(0)
            
            logits = model(embeddings)
            probs = torch.softmax(logits, dim=-1)
            
            top_prob, top_idx = torch.max(probs[0], dim=0)
        
        action = idx_to_action[top_idx.item()]
        confidence = top_prob.item() * 100
        
        predictions.append({
            'clip_num': i + 1,
            'time': f"{clip['start_time']:.1f}s - {clip['end_time']:.1f}s",
            'action': action,
            'confidence': confidence
        })
        
        print(f"   Clip {i+1}/{len(clips)}: [{clip['start_time']:5.1f}s - {clip['end_time']:5.1f}s] → {action} ({confidence:.1f}%)")
    
    # Deduplicate
    print("\n" + "="*70)
    print("📋 PREDICTED RECIPE SEQUENCE:")
    print("="*70)
    
    deduped = []
    prev_action = None
    
    for pred in predictions:
        if pred['action'] != prev_action:
            deduped.append(pred)
            prev_action = pred['action']
    
    for i, step in enumerate(deduped, 1):
        print(f"{i:2d}. [{step['time']:20s}] {step['action']:30s} (conf: {step['confidence']:5.1f}%)")
    
    print("="*70)
    
    return deduped

# Main
print("\n" + "="*70)
print("🧪 FULL RECIPE PREDICTION TEST")
print("="*70)

test_videos = [
    ("https://www.youtube.com/watch?v=Upqp21Dm5vg", "Gordon Ramsay Scrambled Eggs (5min)"),
    ("https://www.youtube.com/watch?v=PUP7U5vTMM0", "How to Dice an Onion (2min)"),
    ("https://www.youtube.com/watch?v=bJUiWdM__Qw", "Simple Pasta Recipe (8min)"),
]

print("\n📹 Suggested test videos:")
for i, (url, desc) in enumerate(test_videos, 1):
    print(f"{i}. {desc}")
    print(f"   {url}")

print("\n" + "-"*60)
choice = input("Enter video number (1-3) or paste your own URL: ").strip()

if choice in ['1', '2', '3']:
    video_url = test_videos[int(choice)-1][0]
    desc = test_videos[int(choice)-1][1]
    print(f"\n📹 Testing on: {desc}")
else:
    video_url = choice

print("\n📥 Downloading video...")
result = download_youtube_video(video_url)
if isinstance(result, tuple):
    video_path, title = result
else:
    video_path = result
    title = "Unknown"

print(f"✅ {title}")

recipe = predict_recipe_sequence(video_path)

import os
if os.path.exists(video_path):
    os.remove(video_path)

print("\n✅ Recipe prediction complete!")
