"""
🎬 FULL-LENGTH VLA DEMO VIDEO
Target: Masala Khichdi (Chef Sanjyot) - ENTIRE VIDEO
Side-by-side: Original video + Real-time VLA predictions
"""
import torch
import json
import cv2
import open_clip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import subprocess

# CONFIG
MODEL_PATH = 'models/full_fidelity_vla.pt'
VOCAB_PATH = 'models/vocab_full.json'
VIDEO_URL = "https://www.youtube.com/watch?v=_JUcqjCKhHc"  # Masala Khichdi
OUTPUT_VIDEO = "vla_demo_full.mp4"

# Robot command mapping with emojis
ROBOT_COMMANDS = {
    'stirring curry': '🥘 Stir(Pan, Circular)',
    'stirring gently': '🥘 Stir(Pan, Slow)',
    'sautéing': '🔥 Stir(Pan, Fast)',
    'frying onions': '🧅 Stir(Pan, Medium)',
    'simmering': '♨️ Stir(Pan, Occasional)',
    'boiling': '💨 Stir(Pan, Caution)',
    'roasting': '🔥 Stir(Pan, Dry)',
    'mixing thoroughly': '🥣 Stir(Bowl, Fold)',
    'adding water': '💧 Pour(Water)',
    'adding oil': '🫗 Pour(Oil)',
    'adding ghee': '🧈 Pour(Ghee)',
    'adding vegetables': '🥬 Transfer(Veg→Pan)',
    'adding tomatoes': '🍅 Transfer(Tomato→Pan)',
    'adding paneer': '🧀 Transfer(Paneer→Pan)',
    'adding rice': '🍚 Transfer(Rice→Pan)',
    'adding masala': '🌶️ Sprinkle(Masala)',
    'adding turmeric': '🟡 Sprinkle(Turmeric)',
    'tempering spices': '✨ Sprinkle(WholeSpices)',
    'garnishing with coriander': '🌿 Sprinkle(Herbs)',
    'grinding paste': '⚙️ Process(Grinder)',
    'pressure cooking': '⏱️ Wait(Whistle)',
    'kneading dough': '👐 Press(Dough)',
    'chopping vegetables': '🔪 Slice(Veg)',
    'plating': '🍽️ Serve(Plate)',
}

class FullFidelityVLA(torch.nn.Module):
    def __init__(self, num_classes, input_dim=512, hidden_dim=512, num_layers=6, nhead=8):
        super().__init__()
        self.pos_enc = torch.nn.Parameter(torch.randn(1, 30, input_dim) * 0.02)
        encoder = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, 
            dim_feedforward=2048, dropout=0.4, batch_first=True
        )
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

print("🧠 Loading VLA model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
with open(VOCAB_PATH) as f:
    vocab = json.load(f)
inv_vocab = {v: k for k, v in vocab.items()}

model = FullFidelityVLA(len(vocab)).to(device)
if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
elif 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

print("👁️ Loading CLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
)

# Download video
print("🎥 Downloading Masala Khichdi video (Chef Sanjyot)...")
if os.path.exists('demo_full_input.mp4'):
    os.remove('demo_full_input.mp4')
subprocess.run([
    'yt-dlp', '-f', '18', '-o', 'demo_full_input.mp4', VIDEO_URL
], check=True)

# Process video
print("🎬 Creating full-length demo video...")
print("⏱️  This will take 5-10 minutes...")

cap = cv2.VideoCapture('demo_full_input.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps

print(f"📹 Video info: {width}x{height}, {fps} fps, {duration_sec:.1f} seconds")

# Output video (2x width for side-by-side)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('temp_full_video.mp4', fourcc, fps, (width * 2, height))

# Process every 5 seconds
frame_count = 0
segment_duration = 5 * fps  # 5 seconds
current_prediction = "Processing..."
current_confidence = 0.0
segments_processed = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Every 5 seconds, run VLA prediction
        if frame_count % segment_duration == 0:
            current_time = frame_count / fps
            print(f"⏳ Processing {current_time:.0f}s / {duration_sec:.0f}s ({current_time/duration_sec*100:.0f}%)")
            
            # Extract 30 frames for VLA
            frames = []
            temp_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            for i in range(30):
                ret_temp, frame_temp = cap.read()
                if ret_temp:
                    rgb = cv2.cvtColor(frame_temp, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    tensor = preprocess(pil_img)
                    frames.append(tensor)
            
            # Reset position
            cap.set(cv2.CAP_PROP_POS_FRAMES, temp_pos)
            
            if len(frames) == 30:
                # Run VLA
                with torch.no_grad():
                    clip_tensor = torch.stack(frames).to(device)
                    feats = clip_model.encode_image(clip_tensor)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    logits = model(feats.unsqueeze(0))
                    probs = torch.softmax(logits, dim=1)
                    
                    top_conf, top_idx = torch.topk(probs, 1)
                    current_prediction = inv_vocab[top_idx.item()]
                    current_confidence = top_conf.item()
                    segments_processed += 1
        
        # Create visualization frame
        left_frame = frame.copy()
        right_frame = np.zeros_like(frame)
        
        # Convert to PIL for text rendering
        pil_right = Image.fromarray(right_frame)
        draw = ImageDraw.Draw(pil_right)
        
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
            font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            font_title = ImageFont.load_default()
            font_large = ImageFont.load_default()
            font_med = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Title
        draw.text((width//2 - 180, 30), "🤖 Indian Cooking VLA", fill=(255, 255, 255), font=font_title)
        draw.text((width//2 - 100, 70), "Real-Time Prediction", fill=(200, 200, 200), font=font_small)
        
        # Current action
        draw.text((30, 150), "Current Action:", fill=(200, 200, 200), font=font_med)
        robot_cmd = ROBOT_COMMANDS.get(current_prediction, f"🎯 {current_prediction}")
        draw.text((30, 190), robot_cmd, fill=(100, 255, 100), font=font_large)
        
        # Confidence
        draw.text((30, 270), "Confidence:", fill=(200, 200, 200), font=font_med)
        conf_text = f"{current_confidence * 100:.1f}%"
        conf_color = (100, 255, 100) if current_confidence > 0.7 else (255, 200, 100) if current_confidence > 0.5 else (255, 100, 100)
        draw.text((30, 310), conf_text, fill=conf_color, font=font_large)
        
        # Confidence bar
        bar_width = int((width - 60) * current_confidence)
        draw.rectangle([(30, 370), (30 + bar_width, 405)], fill=conf_color)
        draw.rectangle([(30, 370), (width - 30, 405)], outline=(100, 100, 100), width=2)
        
        # Progress
        progress = frame_count / total_frames
        draw.text((30, height - 120), f"Progress: {progress*100:.0f}%", fill=(150, 150, 150), font=font_small)
        prog_width = int((width - 60) * progress)
        draw.rectangle([(30, height - 90), (30 + prog_width, height - 70)], fill=(100, 150, 255))
        draw.rectangle([(30, height - 90), (width - 30, height - 70)], outline=(100, 100, 100), width=1)
        
        # Stats (bottom)
        draw.text((30, height - 50), "Model: VLA Transformer (19M)", fill=(150, 150, 150), font=font_small)
        draw.text((30, height - 30), "Dataset: 1,607 Indian clips | Accuracy: 77%", fill=(150, 150, 150), font=font_small)
        
        right_frame = np.array(pil_right)
        
        # Combine side-by-side
        combined = np.hstack([left_frame, right_frame])
        out.write(combined)
        
        frame_count += 1

finally:
    cap.release()
    out.release()

print(f"✅ Processed {segments_processed} segments")
print("🎵 Adding audio...")

subprocess.run([
    'ffmpeg', '-y', '-i', 'temp_full_video.mp4', '-i', 'demo_full_input.mp4',
    '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac',
    '-shortest', OUTPUT_VIDEO
], check=True, capture_output=True)

os.remove('temp_full_video.mp4')

print(f"\n✅ FULL DEMO VIDEO CREATED: {OUTPUT_VIDEO}")
print(f"📹 Duration: {duration_sec:.1f} seconds ({duration_sec/60:.1f} minutes)")
print(f"🎬 Resolution: {width*2}x{height}")
print(f"🎯 Segments analyzed: {segments_processed}")
print(f"\n📥 Download with:")
print(f"scp anshb3@cc-login.campuscluster.illinois.edu:~/vla_project/project2/{OUTPUT_VIDEO} .")
