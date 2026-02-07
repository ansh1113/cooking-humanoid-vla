"""
🇮🇳 INDIAN VLA DEMO: THE GRAND UNIFIER
Target Video: Dal Chawal (Chef Varun / Rajshri Food)
"""
import torch
import json
import cv2
import open_clip
from PIL import Image
import numpy as np
import os
import sys
import subprocess

# CONFIG
MODEL_PATH = 'models/full_fidelity_vla.pt'
VOCAB_PATH = 'models/vocab_full.json'
TEST_URL = "https://www.youtube.com/watch?v=gQtZXZvS-lc" 

# ROBOT TRANSLATOR (The "Body" that executes the "Brain's" command)
ROBOT_ACTIONS = {
    # STIR group
    'stirring curry': 'Stir(Pan, Circular)',
    'stirring gently': 'Stir(Pan, Slow)',
    'sautéing': 'Stir(Pan, Fast)',
    'frying onions': 'Stir(Pan, Medium)',
    'simmering': 'Stir(Pan, Occasional)',
    'boiling': 'Stir(Pan, Caution)',
    'roasting': 'Stir(Pan, Dry)',
    'mixing thoroughly': 'Stir(Bowl, Fold)',
    'shallow frying': 'Stir(Pan, Flip)',
    'deep frying': 'Stir(Wok, Deep)',
    
    # POUR group
    'adding water': 'Pour(Water)',
    'adding oil': 'Pour(Oil)',
    'adding ghee': 'Pour(Ghee)',
    'adding butter': 'Place(Butter)',
    'adding ginger garlic paste': 'Scoop(Paste)',
    
    # TRANSFER group
    'adding vegetables': 'Transfer(Bowl->Pan)',
    'adding tomatoes': 'Transfer(Bowl->Pan)',
    'adding paneer': 'Transfer(Bowl->Pan)',
    'adding chicken': 'Transfer(Bowl->Pan)',
    'adding rice': 'Transfer(Bowl->Pan)',
    'plating': 'Serve(Plate)',
    'serving': 'Serve(Plate)',
    
    # SPRINKLE group
    'adding masala': 'Sprinkle(SpiceBox)',
    'adding turmeric': 'Sprinkle(Turmeric)',
    'adding chili powder': 'Sprinkle(Chili)',
    'tempering spices': 'Sprinkle(WholeSpices)',
    'garnishing with coriander': 'Sprinkle(Herbs)',
    'garnishing with cream': 'Drizzle(Cream)',
    'adding coriander': 'Sprinkle(Herbs)',
    
    # MANIPULATION group
    'kneading dough': 'Press(Dough)',
    'grinding paste': 'Process(Grinder)',
    'chopping onion': 'Slice(Onion)',
    'chopping vegetables': 'Slice(Veg)',
    'mincing ginger garlic': 'Mince(GingerGarlic)',
    'peeling': 'Peel(Veg)',
    'washing ingredients': 'Wash(Sink)',
    
    # PASSIVE group
    'pressure cooking': 'Wait(Whistle)',
    'steaming': 'Wait(Steam)',
    'grilling': 'Wait(Grill)'
}

class FullFidelityVLA(torch.nn.Module):
    def __init__(self, num_classes, input_dim=512, hidden_dim=512, num_layers=6, nhead=8):
        super().__init__()
        self.pos_enc = torch.nn.Parameter(torch.randn(1, 30, input_dim) * 0.02)
        encoder = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=2048, dropout=0.4, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, num_classes)
        )
    def forward(self, x):
        x = x + self.pos_enc
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)

def load_brain(device):
    print("🧠 Loading Indian VLA Brain...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    with open(VOCAB_PATH) as f: vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    model = FullFidelityVLA(len(vocab)).to(device)
    if 'model' in checkpoint: model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
    else: model.load_state_dict(checkpoint)
    model.eval()
    return model, vocab, inv_vocab

def get_transcript(audio_path):
    import whisper
    print("👂 Listening (Whisper)...")
    model = whisper.load_model("base")
    return model.transcribe(audio_path, task="translate")

def analyze_clip(model, clip_model, preprocess, video_path, start, end, device, inv_vocab):
    cap = cv2.VideoCapture(video_path)
    frames = []
    indices = np.linspace(start, end, 30)
    for t in indices:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess(Image.fromarray(frame)))
    cap.release()
    if len(frames) < 30: return None
    with torch.no_grad():
        img_tensor = torch.stack(frames).to(device)
        feats = clip_model.encode_image(img_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = model(feats.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
    top_probs, top_idxs = torch.topk(probs, 3)
    results = []
    for i in range(3):
        act = inv_vocab[top_idxs[0][i].item()]
        conf = top_probs[0][i].item()
        results.append((act, conf))
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vla_model, vocab, inv_vocab = load_brain(device)
    print("👁️  Loading Vision (CLIP)...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    
    print(f"\n🎥 Downloading Dal Chawal Video...")
    vid_file = "test_demo.mp4"
    aud_file = "test_demo.mp3"
    if os.path.exists(vid_file): os.remove(vid_file)
    if os.path.exists(aud_file): os.remove(aud_file)

    # Use format 18 (360p MP4) for reliability
    os.system(f"yt-dlp -f 18 -o {vid_file} {TEST_URL}")
    
    if not os.path.exists(vid_file):
        print("❌ Download failed.")
        return

    os.system(f"ffmpeg -y -i {vid_file} -ar 16000 -ac 1 {aud_file} >/dev/null 2>&1")
    transcript = get_transcript(aud_file)
    
    print("\n🤖 GENERATING ROBOT ACTION PLAN")
    print("="*80)
    print(f"{'TIME':<10} | {'ROBOT COMMAND':<25} | {'CONFIDENCE':<10} | {'AUDIO CONTEXT'}")
    print("-" * 80)
    
    prev_cmd = ""
    for segment in transcript['segments']:
        start, end = segment['start'], segment['end']
        text = segment['text']
        duration = end - start
        
        # Analyze up to 10 mins (600s) to catch the end tempering
        if start > 600: break 
        if duration < 3: continue
        
        preds = analyze_clip(vla_model, clip_model, preprocess, vid_file, start, end, device, inv_vocab)
        if not preds: continue
        top1_act, top1_conf = preds[0]
        robot_cmd = ROBOT_ACTIONS.get(top1_act, "Wait()")
        
        if robot_cmd != prev_cmd or top1_conf > 0.6:
            conf_str = f"{top1_conf:.0%}"
            if top1_conf > 0.7: conf_str = f"\033[92m{conf_str}\033[0m"
            elif top1_conf < 0.4: conf_str = f"\033[91m{conf_str}\033[0m"
            clean_text = text.replace('\n', ' ').strip()[:35]
            print(f"{start:3.0f}s-{end:3.0f}s | {robot_cmd:<25} | {conf_str:<10} | \"{clean_text}...\"")
            prev_cmd = robot_cmd

if __name__ == "__main__":
    main()
