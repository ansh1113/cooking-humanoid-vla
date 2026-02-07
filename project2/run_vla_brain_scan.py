"""
VLA BRAIN SCAN 🧠
Reveals the Top-2 Probabilities to diagnose "Model Confusion" vs "Model Bias".
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

# CONFIG
MODEL_PATH = 'models/indian_finetuned_vla.pt'
VOCAB_PATH = 'data/processed/indian_verb_vocab.json'

def load_brain(device):
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # Replicate Architecture
    model = TemporalVLA(512, 512, 21, 8, 6).to(device)
    new_head = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 4) 
    ).to(device)
    if hasattr(model, 'classifier'): model.classifier = new_head
    elif hasattr(model, 'head'): model.head = new_head
    elif hasattr(model, 'action_head'): model.action_head = new_head
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, inv_vocab

def get_brain_activity(model, clip_model, preprocess, video_path, start, end, vocab, device):
    cap = cv2.VideoCapture(video_path)
    frames = []
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
        probs = torch.softmax(logits, dim=1) # Get % probabilities
        
    # Get Top 3
    top_probs, top_idxs = torch.topk(probs, 3)
    
    results = []
    for i in range(3):
        results.append((vocab[top_idxs[0][i].item()], top_probs[0][i].item()))
    return results

def main():
    video_url = "https://www.youtube.com/watch?v=oYZ--rdHL6I"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab = load_brain(device)
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    
    vid_file = "brain_scan.mp4"
    aud_file = "brain_scan.mp3"
    try:
        download_fresh_video(video_url, vid_file)
        extract_audio(vid_file, aud_file)
        transcript = get_transcript(aud_file)
        chapters = detect_chapters(transcript)
        
        print(f"\n🧠 VLA BRAIN SCAN RESULTS")
        print("=" * 70)
        
        for i, ch in enumerate(chapters, 1):
            if (ch['end'] - ch['start']) < 4: continue
            
            activity = get_brain_activity(model, clip_model, preprocess, vid_file, ch['start'], ch['end'], vocab, device)
            
            p1_act, p1_conf = activity[0]
            p2_act, p2_conf = activity[1]
            
            # Formatting
            bar = "█" * int(p1_conf * 20)
            bar2 = "░" * int(p2_conf * 20)
            
            print(f"STEP {i}: \"{ch['text'][:40]}...\"")
            print(f"   1️⃣  {p1_act}: {p1_conf:.1%} {bar}")
            print(f"   2️⃣  {p2_act}: {p2_conf:.1%} {bar2}")
            
            if p1_act == "ADD" and p2_act == "MIX" and abs(p1_conf - p2_conf) < 0.10:
                print("   🤔 INSIGHT: Visual Ambiguity (Could be Mixing OR Adding)")
            
            # Check for the Missing "CUT"
            if "cut" in ch['text'].lower() and p1_act != "CUT":
                cut_prob = next((p for a,p in activity if a=="CUT"), 0.0)
                print(f"   🔪 CUT Probability: {cut_prob:.1%} (Model does not see cutting)")
                
            print("-" * 70)

    finally:
        if os.path.exists(vid_file): os.remove(vid_file)
        if os.path.exists(aud_file): os.remove(aud_file)

if __name__ == "__main__":
    main()
