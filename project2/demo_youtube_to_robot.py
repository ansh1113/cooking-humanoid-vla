"""
Day 3: YouTube to Robot Demo
Zero-shot execution: Watch YouTube → Predict actions → Execute
"""
import torch
import torch.nn as nn
from pathlib import Path
import yt_dlp
import cv2
import open_clip
from PIL import Image
import numpy as np
import subprocess
import sys

# Import the model architecture
class TemporalTransformer(nn.Module):
    """Same architecture as training"""
    def __init__(self, embedding_dim=512, hidden_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x):
        features = self.transformer(x)
        last_feature = features[:, -1, :]
        next_embedding = self.predictor(last_feature)
        next_embedding = next_embedding / next_embedding.norm(dim=-1, keepdim=True)
        return next_embedding

# Project 1 Actions (our vocabulary)
ACTIONS = [
    'PickupObject', 'PutObject', 'SliceObject', 'CookObject',
    'OpenObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff',
    'DropHandObject', 'ThrowObject', 'BreakObject'
]

# Action descriptions for CLIP matching
ACTION_DESCRIPTIONS = {
    'PickupObject': 'a photo of a hand picking up an object',
    'PutObject': 'a photo of a hand placing an object down',
    'SliceObject': 'a photo of cutting food with a knife',
    'CookObject': 'a photo of cooking food in a pan',
    'OpenObject': 'a photo of opening a container',
    'CloseObject': 'a photo of closing a container',
    'ToggleObjectOn': 'a photo of turning on an appliance',
    'ToggleObjectOff': 'a photo of turning off an appliance',
    'DropHandObject': 'a photo of dropping an object',
    'ThrowObject': 'a photo of throwing an object',
    'BreakObject': 'a photo of breaking an object'
}

def download_youtube_video(url, output_path='temp_video.mp4'):
    """Download YouTube video"""
    print(f"📥 Downloading video from: {url}")
    
    ydl_opts = {
        'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': True,
        'extractor_args': {
            'youtube': {'player_client': ['android', 'web']}
        }
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"   ✅ Downloaded!")
        return output_path
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return None

def extract_frames_from_video(video_path, max_frames=30):
    """Extract frames from video (1 FPS)"""
    print(f"🎬 Extracting frames...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # 1 frame per second
    
    frames = []
    frame_count = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1
    
    cap.release()
    print(f"   ✅ Extracted {len(frames)} frames")
    return frames

def get_embeddings(frames, clip_model, preprocess, device):
    """Convert frames to CLIP embeddings"""
    print(f"🧠 Encoding frames...")
    
    embeddings = []
    batch_size = 16
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        
        # Preprocess
        images = []
        for frame in batch_frames:
            pil_img = Image.fromarray(frame)
            images.append(preprocess(pil_img))
        
        image_tensor = torch.stack(images).to(device)
        
        # Extract features
        with torch.no_grad():
            features = clip_model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        
        embeddings.append(features.cpu())
    
    all_embeddings = torch.cat(embeddings)
    print(f"   ✅ Encoded {len(embeddings)} embeddings")
    return all_embeddings

def predict_action_sequence(embeddings, vla_model, clip_model, tokenizer, device, window_size=30):
    """
    Use VLA to predict next states, match to actions
    Returns: List of (action, confidence) tuples
    """
    print(f"\n🤖 Predicting action sequence...")
    
    # Get action embeddings from CLIP text encoder
    action_texts = [ACTION_DESCRIPTIONS[action] for action in ACTIONS]
    text_tokens = tokenizer(action_texts).to(device)
    
    with torch.no_grad():
        action_embeddings = clip_model.encode_text(text_tokens)
        action_embeddings = action_embeddings / action_embeddings.norm(dim=-1, keepdim=True)
    
    # Predict actions using sliding window
    predicted_actions = []
    
    for i in range(len(embeddings) - window_size):
        # Take window of past frames
        window = embeddings[i:i+window_size].unsqueeze(0).to(device)
        
        # Predict next state
        with torch.no_grad():
            next_state = vla_model(window)
        
        # Match to actions using cosine similarity
        similarities = (next_state @ action_embeddings.T).squeeze()
        best_action_idx = similarities.argmax().item()
        confidence = similarities[best_action_idx].item()
        
        predicted_action = ACTIONS[best_action_idx]
        predicted_actions.append((predicted_action, confidence))
        
        print(f"   Frame {i+window_size}: {predicted_action} (confidence: {confidence:.3f})")
    
    return predicted_actions

def generate_action_plan(predicted_actions, min_confidence=0.3):
    """
    Convert predictions to action plan
    Filter by confidence and remove duplicates
    """
    print(f"\n📋 Generating action plan...")
    
    action_plan = []
    last_action = None
    
    for action, confidence in predicted_actions:
        # Filter low confidence
        if confidence < min_confidence:
            continue
        
        # Remove consecutive duplicates
        if action != last_action:
            action_plan.append(action)
            last_action = action
    
    return action_plan

def main():
    print("="*70)
    print("🎬 YOUTUBE TO ROBOT DEMO")
    print("="*70)
    
    # Get YouTube URL
    if len(sys.argv) > 1:
        youtube_url = sys.argv[1]
    else:
        youtube_url = input("\n📺 Enter YouTube URL: ")
    
    device = torch.device('cpu')
    
    # Load CLIP
    print("\n📥 Loading CLIP...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model.eval()
    print("   ✅ CLIP loaded!")
    
    # Load VLA model
    print("\n📥 Loading VLA Brain...")
    vla_model = TemporalTransformer().to(device)
    checkpoint = torch.load('models/temporal_vla_brain.pt', map_location=device)
    vla_model.load_state_dict(checkpoint['model_state_dict'])
    vla_model.eval()
    print("   ✅ VLA loaded!")
    
    # Download video
    video_path = download_youtube_video(youtube_url)
    if not video_path:
        return
    
    # Extract frames
    frames = extract_frames_from_video(video_path, max_frames=60)
    
    # Get embeddings
    embeddings = get_embeddings(frames, clip_model, preprocess, device)
    
    # Predict actions
    predicted_actions = predict_action_sequence(
        embeddings, vla_model, clip_model, tokenizer, device
    )
    
    # Generate action plan
    action_plan = generate_action_plan(predicted_actions, min_confidence=0.3)
    
    # Display results
    print("\n" + "="*70)
    print("🎯 PREDICTED ACTION SEQUENCE:")
    print("="*70)
    for i, action in enumerate(action_plan, 1):
        print(f"   {i}. {action}")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print("\n💡 Next step: Execute these actions in Project 1 simulator")
    print(f"   Actions: {' → '.join(action_plan)}")
    
    # Cleanup
    Path(video_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
