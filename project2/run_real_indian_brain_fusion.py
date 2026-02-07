"""
THE INDIAN SOUS-CHEF BRAIN (FUSION EDITION) 🧠
Logic: Audio (Intent) + Video (Verification) = Robust Plan
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
MODEL_PATH = 'models/temporal_vla_phase2_best.pt'
VOCAB_PATH = 'data/processed/action_vocab_phase2_fixed.json'

ACTION_MAPPING = {
    'chopping carrot': 'CUT', 'chopping herbs': 'CUT', 'chopping onion': 'CUT', 
    'chopping vegetables': 'CUT', 'slicing onion': 'CUT', 'slicing potato': 'CUT', 
    'slicing tomato': 'CUT', 'slicing vegetables': 'CUT',
    'mixing batter': 'MIX', 'mixing ingredients': 'MIX', 'mixing salad': 'MIX',
    'stirring chicken': 'MIX', 'stirring eggs': 'MIX', 'stirring meat': 'MIX',
    'stirring mixture': 'MIX', 'stirring pasta': 'MIX', 'stirring rice': 'MIX',
    'pouring sauce': 'ADD', 'grating cheese': 'ADD',
    'cooking steak': 'COOK', 'frying egg': 'COOK'
}

def load_brain(device):
    print("🧠 Loading Visual Brain...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    model = TemporalVLA(512, 512, len(vocab), 8, 6).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, vocab

def get_visual_primitive(model, clip_model, preprocess, video_path, start, end, vocab, device):
    cap = cv2.VideoCapture(video_path)
    frames = []
    duration = end - start
    if duration <= 0: return "UNKNOWN"
    indices = np.linspace(start, end, 30)
    for t in indices:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(preprocess(pil_img))
    cap.release()
    if len(frames) < 30: return "UNKNOWN"
    
    img_tensor = torch.stack(frames).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(img_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = model(feats.unsqueeze(0))
        pred_idx = torch.argmax(logits, dim=1).item()
    
    raw = vocab[pred_idx]
    return ACTION_MAPPING.get(raw, 'MIX') # Default to MIX if uncertain

def get_audio_intent(text):
    t = text.lower()
    if any(x in t for x in ['cut', 'chop', 'slice', 'dice', 'mince']): return 'CUT'
    if any(x in t for x in ['add', 'pour', 'put', 'transfer', 'garnish']): return 'ADD'
    if any(x in t for x in ['stir', 'mix', 'saute', 'fry', 'whisk']): return 'MIX'
    if any(x in t for x in ['knead', 'roll']): return 'KNEAD'
    return None # Audio is vague ("Cook for 10 mins")

def main():
    video_url = "https://www.youtube.com/watch?v=oYZ--rdHL6I"
    print(f"🎥 PROCESSING RECIPE: {video_url}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Models
    model, raw_vocab_dict = load_brain(device)
    vocab = {v: k for k, v in raw_vocab_dict.items()} # Invert dict
    print("👁️  Loading CLIP...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    
    # 2. Download
    vid_file = "fusion_vid.mp4"
    aud_file = "fusion_aud.mp3"
    
    try:
        download_fresh_video(video_url, vid_file)
        extract_audio(vid_file, aud_file)
        print("👂 Listening (Whisper)...")
        transcript = get_transcript(aud_file)
        chapters = detect_chapters(transcript)
        
        print(f"\n📝 GENERATING INTELLIGENT ROBOT PLAN:")
        print("=" * 70)
        
        plan = []
        for i, ch in enumerate(chapters, 1):
            if (ch['end'] - ch['start']) < 4: continue
            
            # 1. AUDIO ANALYSIS (The Intent)
            audio_intent = get_audio_intent(ch['text'])
            
            # 2. VISUAL ANALYSIS (The Verification)
            visual_intent = get_visual_primitive(model, clip_model, preprocess, vid_file, ch['start'], ch['end'], vocab, device)
            
            # 3. FUSION LOGIC
            # If Audio is specific, it overrides Video (Speech is precise)
            # If Audio is vague, rely on Video (Eyes see what's happening)
            final_action = audio_intent if audio_intent else visual_intent
            
            # 4. OBJECT DETECTION (From Text)
            text = ch['text'].lower()
            obj = "Ingredients"
            if "onion" in text: obj = "Onion"
            elif "paneer" in text: obj = "Paneer"
            elif "tomato" in text: obj = "Tomato"
            elif "salt" in text: obj = "Salt"
            elif "oil" in text or "ghee" in text: obj = "Ghee"
            elif "water" in text: obj = "Water"
            
            # 5. COMMAND GENERATION
            cmd = "Wait()"
            if final_action == "CUT": cmd = f"SliceObject({obj})"
            elif final_action == "MIX": cmd = f"Stir(Pan)"
            elif final_action == "ADD": cmd = f"Transfer({obj}, Pan)"
            elif final_action == "KNEAD": cmd = f"Knead(Dough)"
            elif final_action == "COOK": cmd = f"Cook(Pan)"
            
            print(f"STEP {i} [{ch['start']:.0f}s]:")
            print(f"  🗣️  Audio: \"{ch['text'][:40]}...\" -> Intent: {audio_intent}")
            print(f"  👁️  Video: Detected '{visual_intent}' motion")
            print(f"  🤖  PLAN:  {cmd}")
            print("-" * 70)
            
            plan.append({"step": i, "command": cmd, "reasoning": f"Audio={audio_intent}, Video={visual_intent}"})
            
        with open('final_fusion_plan.json', 'w') as f:
            json.dump(plan, f, indent=2)
            
    finally:
        if os.path.exists(vid_file): os.remove(vid_file)
        if os.path.exists(aud_file): os.remove(aud_file)

if __name__ == "__main__":
    main()
