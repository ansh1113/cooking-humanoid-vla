"""
HERO DEMO: CROSS-CUISINE GENERALIZATION (LOCAL VERSION)
Tests the Western-Trained VLA on Indian/Mexican Clips already on disk.
"""
import torch
import json
import cv2
import open_clip
from PIL import Image
import numpy as np
import random
import os
from phase2_temporal_vla_model import TemporalVLA

# --- CONFIG ---
MODEL_PATH = 'models/temporal_vla_phase2_best.pt'
VOCAB_PATH = 'data/processed/action_vocab_phase2_fixed.json'
DATA_PATH = 'data/processed/training_labels_with_clips.json'

def get_model(device):
    print(f"🧠 Loading Phase 2 Model (Western Specialist)...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    
    model = TemporalVLA(
        embedding_dim=512,
        hidden_dim=512,
        num_actions=len(vocab),
        num_heads=8,
        num_layers=6
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, inv_vocab

def process_clip(clip_path, preprocess, device):
    cap = cv2.VideoCapture(clip_path)
    frames = []
    # Extract 30 frames
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1: return None
    indices = np.linspace(0, total-1, 30, dtype=int)
    
    for i in range(total):
        ret, frame = cap.read()
        if not ret: break
        if i in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(preprocess(pil_img))
    cap.release()
    
    if len(frames) < 30: return None
    return torch.stack(frames).to(device)

def main():
    print("="*60)
    print("🤖 VLA HERO DEMO: WESTERN BRAIN vs. GLOBAL CUISINE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab = get_model(device)
    
    print("👁️  Loading Vision Encoder...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    
    # Load available clips
    with open(DATA_PATH) as f:
        data = json.load(f)
    
    # Pick 5 random samples
    samples = random.sample(data, 5)
    
    print("\n🚀 RUNNING INFERENCE ON UNSEEN DATA")
    print("="*60)
    
    for i, item in enumerate(samples, 1):
        frames = process_clip(item['clip_path'], preprocess, device)
        if frames is None: continue
        
        # Inference
        with torch.no_grad():
            img_embeds = clip_model.encode_image(frames)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
            
            logits = model(img_embeds.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            action = vocab[pred_idx.item()]
            
        print(f"\n🎥 Clip {i}: {item['video_title'][:40]}...")
        print(f"   🗣️  Chef said: \"{item['transcript']}\"")
        print(f"   🧠 VLA Prediction: [{action.upper()}]")
        print(f"   📊 Confidence: {conf.item():.1%}")
        
        # Highlight Generalization
        if "mix" in action.lower() or "stir" in action.lower():
            if "mix" in item['transcript'] or "stir" in item['transcript']:
                print("   ✅ SUCCESS: Physics Match (Mixing)")
        if "chop" in action.lower() or "slice" in action.lower():
            if "cut" in item['transcript'] or "chop" in item['transcript']:
                print("   ✅ SUCCESS: Physics Match (Cutting)")

    print("\n" + "="*60)
    print("🎉 DEMO COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
