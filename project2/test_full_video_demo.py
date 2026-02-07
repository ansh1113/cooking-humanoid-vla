"""
Full recipe prediction using working download method
"""
import torch
import cv2
import numpy as np
from PIL import Image
import open_clip
import yt_dlp
import os
from phase2_temporal_vla_model import TemporalVLA

# Download function that works
def download_fresh_video(url, filename="test_video.mp4"):
    if os.path.exists(filename):
        if os.path.getsize(filename) < 10000:
            os.remove(filename)
        else:
            print(f"✅ Using existing {filename}")
            return filename
        
    print(f"📥 Downloading from {url}...")
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': filename,
        'quiet': True,
        'overwrites': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}}
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except:
        ydl_opts['format'] = 'worst'
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
    return filename

# Load model
checkpoint = torch.load('models/temporal_vla_phase2_best.pt', map_location='cpu', weights_only=False)
vocab = checkpoint['action_vocab']
idx_to_action = {v: k for k, v in vocab.items()}

print("✅ Model loaded (78% accuracy)")
print(f"📚 Actions: {len(vocab)}")

device = torch.device('cpu')
model = TemporalVLA(512, 512, len(vocab), 8, 6).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("📥 Loading CLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
)

def predict_recipe(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"\n📹 Video: {duration:.1f}s @ {fps:.1f} fps")
    
    clip_duration = 5
    overlap = 2
    stride = clip_duration - overlap
    num_clips = int((duration - clip_duration) / stride) + 1
    
    print(f"📊 Analyzing {num_clips} clips...\n")
    
    predictions = []
    
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
                frames.append(preprocess(Image.fromarray(frame_rgb)))
        
        if len(frames) != 30:
            continue
        
        frames_tensor = torch.stack(frames).to(device)
        
        with torch.no_grad():
            emb = clip_model.encode_image(frames_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            logits = model(emb.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            top_prob, top_idx = torch.max(probs[0], dim=0)
        
        action = idx_to_action[top_idx.item()]
        conf = top_prob.item() * 100
        
        predictions.append((start_time, end_time, action, conf))
        print(f"[{start_time:5.1f}s - {end_time:5.1f}s] {action:30s} ({conf:5.1f}%)")
    
    cap.release()
    
    # Deduplicate
    print("\n" + "="*70)
    print("📋 PREDICTED RECIPE:")
    print("="*70)
    
    deduped = []
    prev = None
    for t1, t2, action, conf in predictions:
        if action != prev:
            deduped.append((t1, t2, action, conf))
            prev = action
    
    for i, (t1, t2, action, conf) in enumerate(deduped, 1):
        print(f"{i:2d}. [{t1:5.1f}s - {t2:5.1f}s] {action:30s} ({conf:5.1f}%)")
    
    print("="*70)

# Test videos
print("\n" + "="*70)
print("🧪 RECIPE PREDICTION TEST")
print("="*70)

test_videos = [
    ("https://www.youtube.com/watch?v=oYZ--rdHL6I", "Paneer Butter Masala"),
    ("https://www.youtube.com/watch?v=a03U45jFxOI", "Butter Chicken"),
]

print("\n📹 Available test videos:")
for i, (url, desc) in enumerate(test_videos, 1):
    print(f"{i}. {desc}")

choice = input("\nEnter video number (1-2) or paste URL: ").strip()

if choice in ['1', '2']:
    video_url = test_videos[int(choice)-1][0]
    desc = test_videos[int(choice)-1][1]
    print(f"\n📹 Testing: {desc}")
else:
    video_url = choice

video_file = download_fresh_video(video_url)
predict_recipe(video_file)

if os.path.exists(video_file):
    os.remove(video_file)

print("\n✅ Done!")
