"""
PHASE 3 FINAL DEMO: The Agentic Chef (Fixed)
Integrates: Whisper (Audio) + Temporal VLA (Vision) + Robot Logic
"""
import torch
import json
import numpy as np
import whisper
import yt_dlp
import cv2
import open_clip
from PIL import Image
from pathlib import Path
import subprocess
import sys
import os

# --- PATH SETUP ---
# Ensure we can import from local directory
sys.path.append(os.getcwd())

from phase2_temporal_vla_model import TemporalVLA

# --- CONFIG ---
MODEL_PATH = 'models/temporal_vla_phase2_best.pt'
VOCAB_PATH = 'data/processed/action_vocab_phase2_fixed.json'
VIDEO_URL = "https://www.youtube.com/watch?v=oYZ--rdHL6I"

def download_video_and_audio(url):
    print(f"📥 Downloading video & audio from {url}...")
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': 'demo_video.mp4',
        'quiet': True,
        'overwrites': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}} 
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"⚠️  Download failed with primary method: {e}")
        print("    Trying fallback method...")
        ydl_opts['format'] = 'worst' 
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    
    print("🔊 Extracting audio...")
    cmd = ['ffmpeg', '-y', '-i', 'demo_video.mp4', '-ab', '160k', '-ac', '2', '-ar', '44100', '-vn', 'demo_audio.mp3']
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 'demo_video.mp4', 'demo_audio.mp3'

def get_audio_plan(audio_path):
    print("👂 Listening to recipe (Whisper)...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    
    recipe_steps = []
    keywords = ['chop', 'slice', 'cut', 'stir', 'mix', 'fry', 'cook', 'pour', 'grate', 'heat', 'add']
    
    print("\n📜 Recipe Steps Detected from Audio:")
    for segment in result['segments']:
        text = segment['text'].lower()
        start = segment['start']
        
        for k in keywords:
            if k in text:
                if not recipe_steps or (start - recipe_steps[-1]['time'] > 10):
                    print(f"   [{start:.0f}s] Chef said: '{text.strip()}'")
                    recipe_steps.append({'time': start, 'text': text, 'keyword': k})
                    break
    return recipe_steps

def extract_clip_embeddings(video_path, start_time, duration=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps: fps = 30
    
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    
    frames = []
    frame_step = int((duration * fps) / 30)
    if frame_step < 1: frame_step = 1
    
    for _ in range(30):
        ret, frame = cap.read()
        if not ret: break
        
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(preprocess(pil_img))
        for _ in range(frame_step): cap.read()
            
    if len(frames) < 10: return None
    
    if len(frames) < 30:
        frames += [frames[-1]] * (30 - len(frames))
        
    img_tensor = torch.stack(frames).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    
    return emb.unsqueeze(0)

def main():
    print("="*70)
    print("🇮🇳 PHASE 3 FINAL DEMO: INDIAN RECIPE AGENT")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n🧠 Loading Temporal VLA Model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    vocab = json.load(open(VOCAB_PATH))
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # --- FIX: Correct Argument Order ---
    model = TemporalVLA(
        embedding_dim=512,
        hidden_dim=512,
        num_actions=len(vocab),
        num_heads=8,
        num_layers=6
    ).to(device)
    # -----------------------------------
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✅ Model Loaded (Vocab size: {len(vocab)})")
    
    video_path, audio_path = download_video_and_audio(VIDEO_URL)
    steps = get_audio_plan(audio_path)
    
    print("\n" + "="*70)
    print("⚡ EXECUTING RECIPE AGENT")
    print("="*70)
    
    for i, step in enumerate(steps, 1):
        print(f"\n👉 Step {i} [{step['time']:.0f}s]: Chef said \"...{step['keyword']}...\"")
        
        embeddings = extract_clip_embeddings(video_path, step['time'])
        
        if embeddings is not None:
            with torch.no_grad():
                logits = model(embeddings)
                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                
                predicted_action = inv_vocab[pred_idx.item()]
                confidence = conf.item()
                
            print(f"   👀 Visual Analysis: I see [{predicted_action}]")
            print(f"   📊 Confidence: {confidence*100:.1f}%")
            
            cmd = "Unknown"
            if "chop" in predicted_action: cmd = "SliceObject (Knife)"
            elif "slice" in predicted_action: cmd = "SliceObject (Knife)"
            elif "stir" in predicted_action: cmd = "CookObject (Stir)"
            elif "mix" in predicted_action: cmd = "CookObject (Mix)"
            elif "fry" in predicted_action: cmd = "CookObject (Fry)"
            elif "pour" in predicted_action: cmd = "PutObject (Pour)"
            elif "grate" in predicted_action: cmd = "SliceObject (Grater)"
            
            print(f"   🤖 ROBOT ACTION: {cmd}")
            
        else:
            print("   ❌ Video buffering error")

    print("\n" + "="*70)
    print("🎉 DEMO COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
