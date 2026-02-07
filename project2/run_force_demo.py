"""
FORCE DEMO: Manual Override
Finds a real video file and forces the model to predict on it.
"""
import torch
import json
import cv2
import open_clip
from PIL import Image
import numpy as np
import os
import glob
from phase2_temporal_vla_model import TemporalVLA

# --- CONFIG ---
MODEL_PATH = 'models/temporal_vla_phase2_best.pt'
VOCAB_PATH = 'data/processed/action_vocab_phase2_fixed.json'
CLIP_DIR = 'data/training_clips'

def get_model(device):
    print(f"🧠 Loading 78% Western Model...")
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

def main():
    print("="*60)
    print("🚨 FORCE DEMO: RUNNING ON REAL DATA")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load CLIP (The Fix: Unpack 3 values)
    print("👁️  Loading Vision Encoder...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    
    # 2. Find a Real Video
    mp4s = glob.glob(os.path.join(CLIP_DIR, "*.mp4"))
    if not mp4s:
        print("❌ ERROR: No MP4 files found in", CLIP_DIR)
        return
    
    # Pick a random one
    import random
    target_video = random.choice(mp4s)
    print(f"📂 Found Video: {os.path.basename(target_video)}")
    
    # 3. Process Video
    cap = cv2.VideoCapture(target_video)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   Frames: {total}")
    
    if total < 1:
        print("❌ Video is empty/corrupt.")
        return

    # Sample 30 frames
    indices = np.linspace(0, total-1, 30, dtype=int)
    
    for i in range(total):
        ret, frame = cap.read()
        if not ret: break
        if i in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(preprocess(pil_img))
    cap.release()
    
    if len(frames) < 30:
        frames += [frames[-1]] * (30 - len(frames))
        
    img_tensor = torch.stack(frames).to(device)
    
    # 4. Encode Video
    with torch.no_grad():
        img_embeds = clip_model.encode_image(img_tensor)
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
    
    # 5. Run Inference
    model, vocab = get_model(device)
    with torch.no_grad():
        logits = model(img_embeds.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        action = vocab[pred_idx.item()]
    
    print("\n" + "="*60)
    print(f"🎯 PREDICTION RESULT")
    print("="*60)
    print(f"🎥 Video:      {os.path.basename(target_video)}")
    print(f"🧠 Prediction: [{action.upper()}]")
    print(f"📊 Confidence: {conf.item():.1%}")
    print("="*60)
    
    # Interpretation
    if "mix" in action.lower() or "stir" in action.lower():
        print("💡 Insight: Model detected circular motion (Mixing/Stirring).")
    elif "chop" in action.lower() or "slice" in action.lower():
        print("💡 Insight: Model detected vertical motion (Cutting).")
    elif "pour" in action.lower():
        print("💡 Insight: Model detected object transfer (Pouring).")

if __name__ == "__main__":
    main()
