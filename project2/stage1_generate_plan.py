"""
STAGE 1: YouTube → Action Plan
Runs on login node (no simulator needed)
"""
import torch
import torch.nn as nn
from pathlib import Path
import yt_dlp
import cv2
import open_clip
from PIL import Image
import json
import sys

class TemporalTransformer(nn.Module):
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

ACTIONS = [
    'PickupObject', 'PutObject', 'SliceObject', 'CookObject',
    'OpenObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff',
    'DropHandObject', 'ThrowObject', 'BreakObject'
]

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
    print(f"\n📥 Downloading video...")
    print(f"   URL: {url}")
    
    ydl_opts = {
        'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}}
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            print(f"   ✅ {info['title']}")
        return output_path, info['title']
    except Exception as e:
        print(f"   ❌ {e}")
        return None, None

def extract_frames(video_path, max_frames=60):
    print(f"\n🎬 Extracting frames...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)
    
    frames = []
    frame_count = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1
    
    cap.release()
    print(f"   ✅ {len(frames)} frames")
    return frames

def get_embeddings(frames, clip_model, preprocess, device):
    print(f"\n🧠 Encoding with CLIP...")
    
    embeddings = []
    batch_size = 16
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        
        images = []
        for frame in batch_frames:
            pil_img = Image.fromarray(frame)
            images.append(preprocess(pil_img))
        
        image_tensor = torch.stack(images).to(device)
        
        with torch.no_grad():
            features = clip_model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        
        embeddings.append(features.cpu())
    
    all_embeddings = torch.cat(embeddings)
    print(f"   ✅ Done")
    return all_embeddings

def predict_actions(embeddings, vla_model, clip_model, tokenizer, device, window_size=30):
    print(f"\n🤖 Predicting with Temporal VLA...")
    
    action_texts = [ACTION_DESCRIPTIONS[action] for action in ACTIONS]
    text_tokens = tokenizer(action_texts).to(device)
    
    with torch.no_grad():
        action_embeddings = clip_model.encode_text(text_tokens)
        action_embeddings = action_embeddings / action_embeddings.norm(dim=-1, keepdim=True)
    
    predicted_actions = []
    
    for i in range(len(embeddings) - window_size):
        window = embeddings[i:i+window_size].unsqueeze(0).to(device)
        
        with torch.no_grad():
            next_state = vla_model(window)
        
        similarities = (next_state @ action_embeddings.T).squeeze()
        best_action_idx = similarities.argmax().item()
        confidence = similarities[best_action_idx].item()
        
        predicted_action = ACTIONS[best_action_idx]
        predicted_actions.append((predicted_action, confidence))
    
    print(f"   ✅ Done")
    return predicted_actions

def generate_plan(predicted_actions, min_confidence=0.25):
    print(f"\n📋 Generating action plan...")
    
    action_plan = []
    last_action = None
    
    for action, confidence in predicted_actions:
        if confidence < min_confidence:
            continue
        
        if action != last_action:
            action_plan.append({'action': action, 'confidence': confidence})
            last_action = action
            print(f"   → {action} ({confidence:.3f})")
    
    print(f"   ✅ {len(action_plan)} actions")
    return action_plan

def main():
    print("="*70)
    print("📹 STAGE 1: YOUTUBE → ACTION PLAN")
    print("="*70)
    
    if len(sys.argv) > 1:
        youtube_url = sys.argv[1]
    else:
        youtube_url = input("\n📺 YouTube URL: ")
    
    device = torch.device('cpu')
    
    # Load models
    print("\n📥 Loading models...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model.eval()
    
    vla_model = TemporalTransformer().to(device)
    checkpoint = torch.load('models/temporal_vla_brain.pt', map_location=device, weights_only=False)
    vla_model.load_state_dict(checkpoint['model_state_dict'])
    vla_model.eval()
    print("   ✅ Models loaded")
    
    # Pipeline
    video_path, video_title = download_youtube_video(youtube_url)
    if not video_path:
        return
    
    frames = extract_frames(video_path, max_frames=60)
    embeddings = get_embeddings(frames, clip_model, preprocess, device)
    predicted_actions = predict_actions(embeddings, vla_model, clip_model, tokenizer, device)
    action_plan = generate_plan(predicted_actions, min_confidence=0.25)
    
    # Save plan
    plan_data = {
        'youtube_url': youtube_url,
        'video_title': video_title,
        'actions': action_plan
    }
    
    with open('action_plan.json', 'w') as f:
        json.dump(plan_data, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ ACTION PLAN SAVED: action_plan.json")
    print("="*70)
    print(f"Video: {video_title}")
    print(f"Actions: {' → '.join([a['action'] for a in action_plan])}")
    print("\n💡 Next: Run Stage 2 to execute in AI2-THOR")
    
    # Cleanup
    Path(video_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
