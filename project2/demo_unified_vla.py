"""
🤖 UNIFIED VLA: WHISPER + VISION + YOLO
Target: Masala Khichdi (Chef Sanjyot Keer)
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
import warnings
warnings.filterwarnings("ignore")

# CONFIG
MODEL_PATH = 'models/full_fidelity_vla.pt'
VOCAB_PATH = 'models/vocab_full.json'
TEST_URL = "https://www.youtube.com/watch?v=_JUcqjCKhHc" # Masala Khichdi

# ROBOT TRANSLATOR
ROBOT_ACTIONS = {
    # STIR group
    'stirring curry': 'Stir(Pan, Circular)', 'stirring gently': 'Stir(Pan, Slow)',
    'sautéing': 'Stir(Pan, Fast)', 'frying onions': 'Stir(Pan, Medium)',
    'simmering': 'Stir(Pan, Occasional)', 'boiling': 'Stir(Pan, Caution)',
    'roasting': 'Stir(Pan, Dry)', 'mixing thoroughly': 'Stir(Bowl, Fold)',
    'shallow frying': 'Stir(Pan, Flip)', 'deep frying': 'Stir(Wok, Deep)',
    # POUR group
    'adding water': 'Pour(Water)', 'adding oil': 'Pour(Oil)', 'adding ghee': 'Pour(Ghee)',
    'adding butter': 'Place(Butter)', 'adding ginger garlic paste': 'Scoop(Paste)',
    # TRANSFER group
    'adding vegetables': 'Transfer(Bowl->Pan)', 'adding tomatoes': 'Transfer(Bowl->Pan)',
    'adding paneer': 'Transfer(Bowl->Pan)', 'adding chicken': 'Transfer(Bowl->Pan)',
    'adding rice': 'Transfer(Bowl->Pan)', 'plating': 'Serve(Plate)', 'serving': 'Serve(Plate)',
    # SPRINKLE group
    'adding masala': 'Sprinkle(SpiceBox)', 'adding turmeric': 'Sprinkle(Turmeric)',
    'adding chili powder': 'Sprinkle(Chili)', 'tempering spices': 'Sprinkle(WholeSpices)',
    'garnishing with coriander': 'Sprinkle(Herbs)', 'garnishing with cream': 'Drizzle(Cream)',
    'adding coriander': 'Sprinkle(Herbs)',
    # MANIPULATION group
    'kneading dough': 'Press(Dough)', 'grinding paste': 'Process(Grinder)',
    'chopping onion': 'Slice(Onion)', 'chopping vegetables': 'Slice(Veg)',
    'mincing ginger garlic': 'Mince(GingerGarlic)', 'peeling': 'Peel(Veg)',
    'washing ingredients': 'Wash(Sink)',
    # PASSIVE group
    'pressure cooking': 'Wait(Whistle)', 'steaming': 'Wait(Steam)', 'grilling': 'Wait(Grill)'
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

def get_transcript(audio_path):
    import whisper
    print("👂 Listening (Whisper)...")
    model = whisper.load_model("base")
    return model.transcribe(audio_path, task="translate")

# --- YOLO VERIFICATION LOGIC ---
def verify_action(vla_action, vla_conf, detected_objects):
    final_conf = vla_conf
    note = ""
    
    # Logic: Pouring needs Bottle/Cup
    if "adding oil" in vla_action or "adding water" in vla_action:
        if 'bottle' in detected_objects or 'cup' in detected_objects:
            final_conf += 0.2
            note = "+"
    
    # Logic: Chopping needs Knife
    if "chopping" in vla_action:
        if 'knife' in detected_objects:
            final_conf += 0.3
            note = "+"
    
    # Logic: Stirring should NOT have Knife
    if "stirring" in vla_action and 'knife' in detected_objects:
        final_conf -= 0.1
        note = "-"

    return min(final_conf, 1.0), note

def analyze_clip(model, clip_model, preprocess, video_path, start, end, device, inv_vocab):
    cap = cv2.VideoCapture(video_path)
    frames = []
    yolo_frame = None 
    
    indices = np.linspace(start, end, 30)
    idx_count = 0
    
    for t in indices:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            if idx_count == 15: yolo_frame = frame 
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess(Image.fromarray(frame_rgb)))
            idx_count += 1
            
    cap.release()
    if len(frames) < 30: return None, None
    
    # VLA Inference
    with torch.no_grad():
        img_tensor = torch.stack(frames).to(device)
        feats = clip_model.encode_image(img_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = model(feats.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
        
    top_prob, top_idx = torch.topk(probs, 1)
    action = inv_vocab[top_idx.item()]
    conf = top_prob.item()
    
    return (action, conf), yolo_frame

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vla_model, vocab, inv_vocab = load_brain(device)
    
    print("🕵️  Loading Scout (YOLOv8)...")
    yolo_model = YOLO('yolov8n.pt') 

    print("👁️  Loading Vision (CLIP)...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    
    print(f"\n🎥 Downloading Masala Khichdi Video...")
    vid_file = "test_unified.mp4"
    aud_file = "test_unified.mp3"
    if os.path.exists(vid_file): os.remove(vid_file)
    os.system(f"yt-dlp -f 18 -o {vid_file} {TEST_URL}")
    
    if not os.path.exists(vid_file):
        print("❌ Download failed.")
        return

    os.system(f"ffmpeg -y -i {vid_file} -ar 16000 -ac 1 {aud_file} >/dev/null 2>&1")
    transcript = get_transcript(aud_file)
    
    print("\n🤖 UNIFIED ROBOT PERCEPTION (WHISPER + VLA + YOLO)")
    print("="*120)
    print(f"{'TIME':<9} | {'ROBOT COMMAND':<22} | {'CONF':<6} | {'OBJECTS (YOLO)':<18} | {'AUDIO CONTEXT'}")
    print("-" * 120)
    
    for segment in transcript['segments']:
        start, end = segment['start'], segment['end']
        text = segment['text']
        duration = end - start
        
        # Analyze first 5 mins (Video is short)
        if start > 300: break 
        if duration < 2.5: continue 
        
        vla_res, yolo_img = analyze_clip(vla_model, clip_model, preprocess, vid_file, start, end, device, inv_vocab)
        if not vla_res: continue
        
        raw_action, raw_conf = vla_res
        
        if yolo_img is not None:
            results = yolo_model(yolo_img, verbose=False)
            detected = [results[0].names[int(c)] for c in results[0].boxes.cls]
            kitchen_objs = [o for o in detected if o in ['bottle','cup','bowl','spoon','knife','fork']]
            obj_str = ",".join(list(set(kitchen_objs)))
        else:
            obj_str = ""
            
        if not obj_str: obj_str = "-"
        
        final_conf, note = verify_action(raw_action, raw_conf, kitchen_objs)
        robot_cmd = ROBOT_ACTIONS.get(raw_action, "Wait()")
        
        if final_conf > 0.4:
            conf_str = f"{final_conf:.0%}"
            if note == "+": conf_str = f"\033[92m{conf_str}↑\033[0m"
            elif final_conf > 0.7: conf_str = f"\033[92m{conf_str}\033[0m"
            elif final_conf < 0.5: conf_str = f"\033[91m{conf_str}\033[0m"
            
            clean_text = text.replace('\n', ' ').strip()[:40]
            print(f"{start:3.0f}s-{end:3.0f}s | {robot_cmd:<22} | {conf_str:<6} | {obj_str:<18} | \"{clean_text}...\"")

if __name__ == "__main__":
    main()
