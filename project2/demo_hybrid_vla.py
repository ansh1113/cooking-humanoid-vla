"""
🤖 HYBRID VLA: VISION + OBJECT DETECTION
Target: Jeera Rice (Chef Ranveer Brar)
"""
import torch
import json
import cv2
import open_clip
from PIL import Image
import numpy as np
import os
import sys
from ultralytics import YOLO 

# CONFIG
MODEL_PATH = 'models/full_fidelity_vla.pt'
VOCAB_PATH = 'models/vocab_full.json'
TEST_URL = "https://www.youtube.com/watch?v=3c4Kxtryx1w" # Chef Ranveer

# --- 1. THE EXPERT (VLA) ---
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
    print("🧠 Loading VLA Brain...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    with open(VOCAB_PATH) as f: vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    model = FullFidelityVLA(len(vocab)).to(device)
    if 'model' in checkpoint: model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
    else: model.load_state_dict(checkpoint)
    model.eval()
    return model, vocab, inv_vocab

# --- 2. THE LOGIC LAYER ---
def verify_action(vla_action, vla_conf, detected_objects):
    final_action = vla_action
    final_conf = vla_conf
    note = ""

    # Rule 1: Pouring needs a Bottle/Cup
    if "adding oil" in vla_action or "adding water" in vla_action:
        if 'bottle' in detected_objects or 'cup' in detected_objects:
            final_conf += 0.2
            note = "✅ Bottle Found (+Conf)"
        elif 'spoon' in detected_objects:
            if vla_conf < 0.6: 
                final_action = "adding ghee" 
                note = "⚠️ Spoon detected -> Switched to Ghee"

    # Rule 2: Cutting needs a Knife
    if "chopping" in vla_action:
        if 'knife' in detected_objects:
            final_conf += 0.3
            note = "✅ Knife Found (+Conf)"
        else:
            final_conf -= 0.1

    # Rule 3: Stirring implies NO Knife
    if "stirring" in vla_action:
        if 'knife' in detected_objects:
            final_action = "chopping vegetables"
            note = "⚠️ Knife detected -> Switched to Chopping"
            
    final_conf = min(final_conf, 1.0)
    return final_action, final_conf, note

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vla_model, vocab, inv_vocab = load_brain(device)
    
    print("🕵️  Loading Scout (YOLOv8)...")
    yolo_model = YOLO('yolov8n.pt') 

    print("👁️  Loading Vision (CLIP)...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    
    print(f"\n🎥 Downloading Test Video...")
    vid_file = "test_hybrid.mp4"
    if os.path.exists(vid_file): os.remove(vid_file)
    os.system(f"yt-dlp -f 18 -o {vid_file} {TEST_URL}")
    
    if not os.path.exists(vid_file):
        print("❌ Download failed.")
        return
    
    cap = cv2.VideoCapture(vid_file)
    
    print("\n🤖 HYBRID VLA INFERENCE (VLA + YOLO)")
    print("="*110)
    print(f"{'TIME':<8} | {'VLA PREDICTION':<25} | {'CONF':<6} | {'OBJECTS (YOLO)':<20} | {'FINAL DECISION'}")
    print("-" * 110)

    step_sec = 5
    curr_sec = 0 # Start from beginning
    
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, curr_sec * 1000)
        ret, frame = cap.read()
        if not ret or curr_sec > 300: break # Analyze first 5 mins
        
        # VLA
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        tensor = preprocess(pil_img).unsqueeze(0).to(device)
        clip_tensor = torch.stack([tensor.squeeze(0)] * 30).to(device) 
        
        with torch.no_grad():
            feats = clip_model.encode_image(clip_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = vla_model(feats.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            
        top1_conf, top1_idx = torch.topk(probs, 1)
        vla_action = inv_vocab[top1_idx.item()]
        vla_conf = top1_conf.item()

        # YOLO
        results = yolo_model(frame, verbose=False)
        detected_classes = [results[0].names[int(c)] for c in results[0].boxes.cls]
        kitchen_objs = [obj for obj in detected_classes if obj in ['bottle', 'cup', 'bowl', 'spoon', 'knife', 'fork']]
        obj_str = ", ".join(list(set(kitchen_objs)))
        if not obj_str: obj_str = "-"
        
        # VERIFY
        final_act, final_conf, note = verify_action(vla_action, vla_conf, kitchen_objs)
        
        # Color
        conf_str = f"{final_conf:.0%}"
        if final_conf > 0.7: conf_str = f"\033[92m{conf_str}\033[0m"
        elif final_conf < 0.4: conf_str = f"\033[91m{conf_str}\033[0m"
        if note: note = f" -> \033[93m{note}\033[0m"
        
        print(f"{curr_sec}s    | {vla_action:<25} | {conf_str:<6} | {obj_str:<20} | {final_act} {note}")
        curr_sec += step_sec

if __name__ == "__main__":
    main()
