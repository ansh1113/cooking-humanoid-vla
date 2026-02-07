"""
PHASE 1: FEATURE EXTRACTION 🧠
Converts Raw Video Segments -> CLIP Embeddings (30x512)
"""
import torch
import json
import cv2
import open_clip
from PIL import Image
import numpy as np
import os
import sys

# CONFIG
DATA_FILE = 'data/processed/golden_40_dataset.json'
FEATURE_DIR = 'data/features_golden40'
os.makedirs(FEATURE_DIR, exist_ok=True)

def main():
    print(f"🚀 Processing Golden Dataset Features...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Load CLIP (The Frozen Visual Brain)
    print("👁️  Loading CLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    
    with open(DATA_FILE) as f:
        data = json.load(f)
        
    print(f"📊 Total samples to process: {len(data)}")
    
    success = 0
    skipped = 0
    missing = 0
    
    for i, item in enumerate(data):
        # 1. Resolve Video Path
        # Handles both cases: URL-based filenames or explicit filenames
        if 'video_file' in item:
             vid_path = item['video_file']
             vid_id = vid_path.replace('temp_golden_', '').replace('.mp4', '')
        else:
             vid_id = item['video_url'].split('=')[-1]
             vid_path = f"temp_golden_{vid_id}.mp4"
        
        # 2. Define Output Path
        # Naming convention: <VIDEO_ID>_<START_TIME>.pt
        start_str = f"{item['start']:.2f}"
        save_name = f"{vid_id}_{start_str}.pt"
        save_path = os.path.join(FEATURE_DIR, save_name)
        
        # 3. Skip if already done
        if os.path.exists(save_path):
            skipped += 1
            if i % 50 == 0: print(f"   Skipped {i}/{len(data)} (Exists)")
            continue
            
        # 4. Check if video exists
        if not os.path.exists(vid_path):
            # Try searching current directory for the ID
            potential_file = f"temp_golden_{vid_id}.mp4"
            if os.path.exists(potential_file):
                vid_path = potential_file
            else:
                missing += 1
                if missing <= 5: print(f"⚠️  Missing video: {vid_path}")
                continue
            
        # 5. Extract 30 Frames
        cap = cv2.VideoCapture(vid_path)
        frames = []
        # Evenly space 30 frames between start and end time
        indices = np.linspace(item['start'], item['end'], 30)
        
        for t in indices:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(preprocess(Image.fromarray(frame)))
        cap.release()
        
        # 6. Validation & Padding
        if len(frames) < 5:
            print(f"   ⚠️  Corrupt/Too short: {vid_path} segment {item['start']}-{item['end']}")
            continue
            
        # If we got <30 frames (e.g. video ended early), pad with the last frame
        while len(frames) < 30:
            frames.append(frames[-1])
            
        # 7. CLIP Encoding (The Heavy Lifting)
        with torch.no_grad():
            img_tensor = torch.stack(frames).to(device)
            features = model.encode_image(img_tensor)
            # Normalize vector to unit length (Critical for Transformer stability)
            features = features / features.norm(dim=-1, keepdim=True)
            
        # 8. Save
        torch.save(features.cpu(), save_path)
        success += 1
        
        if success % 20 == 0:
            print(f"   Processed {i}/{len(data)} samples...")

    print(f"\n✅ COMPLETE.")
    print(f"   Cached:  {success}")
    print(f"   Skipped: {skipped}")
    print(f"   Missing: {missing}")
    print(f"   Total Ready: {success + skipped}")

if __name__ == "__main__":
    main()
