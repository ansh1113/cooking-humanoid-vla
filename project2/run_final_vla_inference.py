"""
THE FINAL EXAM: FINE-TUNED VLA INFERENCE (FIXED) 🎓
Replicates the training architecture to load weights correctly.
"""
import torch
import torch.nn as nn
import json
import cv2
import open_clip
from PIL import Image
import numpy as np
import os
from phase2_temporal_vla_model import TemporalVLA
from process_8_indian_veg import download_fresh_video, extract_audio, get_transcript, detect_chapters

# --- CONFIG ---
MODEL_PATH = 'models/indian_finetuned_vla.pt'
VOCAB_PATH = 'data/processed/indian_verb_vocab.json'

def load_finetuned_brain(device):
    print("🧠 Loading Fine-Tuned Indian VLA...")
    
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # 1. Initialize Base Model (Parameters don't matter for the head, we replace it)
    model = TemporalVLA(512, 512, 21, 8, 6).to(device)
    
    # 2. REPLICATE THE SURGERY
    # We must define the head EXACTLY as we did during training
    new_head = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 4) 
    ).to(device)
    
    # Replace the layer. The error message told us it's named 'classifier'
    if hasattr(model, 'classifier'):
        model.classifier = new_head
    elif hasattr(model, 'head'):
        model.head = new_head
    elif hasattr(model, 'action_head'):
        model.action_head = new_head
    
    # 3. Load Weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, inv_vocab

def get_visual_prediction(model, clip_model, preprocess, video_path, start, end, vocab, device):
    cap = cv2.VideoCapture(video_path)
    frames = []
    duration = end - start
    if duration <= 0: return "UNKNOWN", 0.0
    
    indices = np.linspace(start, end, 30)
    for t in indices:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(preprocess(pil_img))
    cap.release()
    
    if len(frames) < 30: return "UNKNOWN", 0.0
    
    img_tensor = torch.stack(frames).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(img_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = model(feats.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    
    action = vocab[pred_idx.item()]
    return action, conf.item()

def main():
    video_url = "https://www.youtube.com/watch?v=oYZ--rdHL6I"
    print(f"🎥 PROCESSING RECIPE: {video_url}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Brains
    model, vocab = load_finetuned_brain(device)
    print(f"   Vocabulary: {list(vocab.values())}")
    
    print("👁️  Loading CLIP...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    
    # 2. Prep Video
    vid_file = "final_exam_vid.mp4"
    aud_file = "final_exam_aud.mp3"
    
    try:
        download_fresh_video(video_url, vid_file)
        extract_audio(vid_file, aud_file)
        print("👂 Getting Chapters (Whisper)...")
        transcript = get_transcript(aud_file)
        chapters = detect_chapters(transcript)
        
        print(f"\n🧪 THE FINAL TEST: VISUAL DISCRIMINATION")
        print("Can the model distinguish ADD vs MIX without help?")
        print("=" * 70)
        
        mix_count = 0
        add_count = 0
        cut_count = 0
        
        for i, ch in enumerate(chapters, 1):
            if (ch['end'] - ch['start']) < 4: continue
            
            # PURE VISUAL INFERENCE
            action, conf = get_visual_prediction(model, clip_model, preprocess, vid_file, ch['start'], ch['end'], vocab, device)
            
            # Stats
            if action == 'MIX': mix_count += 1
            elif action == 'ADD': add_count += 1
            elif action == 'CUT': cut_count += 1
            
            # Color code output
            icon = "❓"
            if action == "MIX": icon = "🥣"
            elif action == "ADD": icon = "🧂"
            elif action == "CUT": icon = "🔪"
            
            print(f"STEP {i} [{ch['start']:.0f}s]: \"{ch['text'][:40]}...\"")
            print(f"   🧠 Visual: {icon} {action} ({conf:.1%})")
            
            # Reality Check
            text_intent = "UNKNOWN"
            if "add" in ch['text'].lower() or "pour" in ch['text'].lower(): text_intent = "ADD"
            elif "stir" in ch['text'].lower() or "fry" in ch['text'].lower(): text_intent = "MIX"
            elif "cut" in ch['text'].lower() or "chop" in ch['text'].lower(): text_intent = "CUT"
            
            if text_intent != "UNKNOWN":
                 if text_intent == action:
                     print(f"   ✅ MATCH! (Vision saw '{action}' correctly)")
                 else:
                     print(f"   ❌ Disagreement (Text says {text_intent})")
            print("-" * 70)
            
        print("\n📊 FINAL STATS:")
        print(f"   MIX Detected: {mix_count}")
        print(f"   ADD Detected: {add_count}")
        print(f"   CUT Detected: {cut_count}")
        
        if add_count > 0 and mix_count > 0:
            print("\n🎉 SUCCESS: Model is NOT collapsing to a single class!")
            print("   It can distinguish Stirring from Adding!")
        else:
            print("\n⚠️  WARNING: Model might still be biased.")

    finally:
        if os.path.exists(vid_file): os.remove(vid_file)
        if os.path.exists(aud_file): os.remove(aud_file)

if __name__ == "__main__":
    main()
