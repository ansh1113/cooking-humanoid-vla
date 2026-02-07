"""
THE INDIAN SOUS-CHEF BRAIN (TRANSFER LEARNING EDITION) 🧠
Uses the robust Western Model (78% acc) to drive Indian Cooking logic.
"""
import torch
import json
import cv2
import open_clip
from PIL import Image
import numpy as np
import os
from phase2_temporal_vla_model import TemporalVLA
from process_8_indian_veg import download_fresh_video, extract_audio, get_transcript, detect_chapters

# --- CONFIG ---
# Use the winner model (Phase 2)
MODEL_PATH = 'models/temporal_vla_phase2_best.pt'
VOCAB_PATH = 'data/processed/action_vocab_phase2_fixed.json'

# --- THE TRANSLATOR LAYER ---
# Maps specific Western labels to generic Robot Primitives
ACTION_MAPPING = {
    # CUTTING
    'chopping carrot': 'CUT', 'chopping herbs': 'CUT', 'chopping onion': 'CUT', 
    'chopping vegetables': 'CUT', 'slicing onion': 'CUT', 'slicing potato': 'CUT', 
    'slicing tomato': 'CUT', 'slicing vegetables': 'CUT',
    # MIXING
    'mixing batter': 'MIX', 'mixing ingredients': 'MIX', 'mixing salad': 'MIX',
    'stirring chicken': 'MIX', 'stirring eggs': 'MIX', 'stirring meat': 'MIX',
    'stirring mixture': 'MIX', 'stirring pasta': 'MIX', 'stirring rice': 'MIX',
    # ADDING / POURING
    'pouring sauce': 'ADD', 'grating cheese': 'ADD',
    # COOKING
    'cooking steak': 'COOK', 'frying egg': 'COOK'
}

def load_western_brain(device):
    print("🧠 Loading Western Specialist (Phase 2 Model)...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    
    model = TemporalVLA(512, 512, len(vocab), 8, 6).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, inv_vocab

def get_visual_action(model, clip_model, preprocess, video_path, start, end, vocab, device):
    cap = cv2.VideoCapture(video_path)
    frames = []
    duration = end - start
    if duration <= 0: return None
    indices = np.linspace(start, end, 30)
    
    for t in indices:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(preprocess(pil_img))
    cap.release()
    
    if len(frames) < 30: return None
    
    img_tensor = torch.stack(frames).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(img_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = model(feats.unsqueeze(0))
        pred_idx = torch.argmax(logits, dim=1).item()
        
    raw_label = vocab[pred_idx]
    # Apply Transfer Learning Mapping
    return ACTION_MAPPING.get(raw_label, 'OTHER')

def main():
    # The Paneer Video
    video_url = "https://www.youtube.com/watch?v=oYZ--rdHL6I" 
    print(f"🎥 PROCESSING RECIPE: {video_url}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Brains
    vla_model, vocab = load_western_brain(device)
    print("👁️  Loading CLIP Vision...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    
    # 2. Download
    vid_file = "brain_input_vid.mp4"
    aud_file = "brain_input_aud.mp3"
    
    try:
        download_fresh_video(video_url, vid_file)
        extract_audio(vid_file, aud_file)
        
        # 3. Listen (Whisper)
        print("👂 Listening to Chef...")
        transcript = get_transcript(aud_file)
        chapters = detect_chapters(transcript)
        
        print(f"\n📝 GENERATING ROBOT ACTION PLAN:")
        print("=" * 70)
        
        final_plan = []
        
        for i, ch in enumerate(chapters, 1):
            if (ch['end'] - ch['start']) < 4: continue
            
            # A. Visual Recognition (Transfer Learning)
            primitive = get_visual_action(vla_model, clip_model, preprocess, vid_file, ch['start'], ch['end'], vocab, device)
            if not primitive or primitive == 'OTHER': continue
            
            # B. Audio Context (Noun Extraction)
            text = ch['text'].lower()
            obj = "Ingredients" # Default
            if "onion" in text: obj = "Onion"
            elif "paneer" in text: obj = "Paneer"
            elif "tomato" in text: obj = "Tomato"
            elif "garlic" in text: obj = "GarlicPaste"
            elif "water" in text: obj = "Water"
            elif "salt" in text: obj = "Salt"
            elif "masala" in text: obj = "Masala"
            elif "oil" in text or "ghee" in text: obj = "Ghee"
            
            # C. Generate Command
            cmd = "Wait()"
            if primitive == "CUT": cmd = f"SliceObject({obj})"
            elif primitive == "MIX": cmd = f"Stir(Pan)"
            elif primitive == "ADD": cmd = f"Transfer({obj}, Pan)"
            elif primitive == "COOK": cmd = f"Cook(Pan)"
            
            print(f"STEP {i} [{ch['start']:.0f}s]:")
            print(f"  🗣️  Context: \"{ch['text'][:60]}...\"")
            print(f"  🧠  Visual:  Detected '{primitive}' motion")
            print(f"  🤖  PLAN:    {cmd}")
            print("-" * 70)
            
            final_plan.append({"step": i, "command": cmd, "visual_primitive": primitive, "object": obj})
            
        # Save Plan
        with open('final_robot_action_plan.json', 'w') as f:
            json.dump(final_plan, f, indent=2)
        print(f"\n✅ Plan saved to final_robot_action_plan.json")
            
    finally:
        if os.path.exists(vid_file): os.remove(vid_file)
        if os.path.exists(aud_file): os.remove(aud_file)

if __name__ == "__main__":
    main()
