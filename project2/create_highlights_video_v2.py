"""
Create highlights montage - SIMPLER VERSION
Just extract clips from already-downloaded videos
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess
import os

# High-confidence timestamps from demo videos you already have
HIGHLIGHTS = [
    {'video': 'demo_input.mp4', 'start': 63, 'action': 'Pour(Oil)', 'conf': 0.87, 'emoji': '🫗'},
    {'video': 'demo_input.mp4', 'start': 106, 'action': 'Process(Grinder)', 'conf': 0.90, 'emoji': '⚙️'},
    {'video': 'demo_input.mp4', 'start': 121, 'action': 'Sprinkle(Masala)', 'conf': 0.91, 'emoji': '🌶️'},
    {'video': 'demo_input.mp4', 'start': 190, 'action': 'Pour(Oil)', 'conf': 0.87, 'emoji': '🫗'},
    {'video': 'demo_input.mp4', 'start': 255, 'action': 'Sprinkle(Masala)', 'conf': 0.92, 'emoji': '🌶️'},
]

print("🎬 Creating highlights from existing video...")

if not os.path.exists('demo_input.mp4'):
    print("❌ demo_input.mp4 not found!")
    exit(1)

cap = cv2.VideoCapture('demo_input.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('temp_highlights.mp4', fourcc, fps, (width, height))

for i, highlight in enumerate(HIGHLIGHTS):
    print(f"📹 Extracting highlight {i+1}/{len(HIGHLIGHTS)}: {highlight['action']}")
    
    # Jump to timestamp
    start_frame = int(highlight['start'] * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract 3 seconds (72 frames at 24fps)
    for frame_num in range(int(3 * fps)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add overlay
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Create semi-transparent overlay
        overlay = Image.new('RGBA', pil_frame.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.rectangle([(0, height-120), (width, height)], fill=(0, 0, 0, 200))
        
        pil_frame = pil_frame.convert('RGBA')
        pil_frame = Image.alpha_composite(pil_frame, overlay)
        pil_frame = pil_frame.convert('RGB')
        
        draw = ImageDraw.Draw(pil_frame)
        
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
            font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        except:
            font_large = ImageFont.load_default()
            font_med = ImageFont.load_default()
        
        # Draw text
        text = f"{highlight['emoji']} {highlight['action']}"
        draw.text((30, height-95), text, fill=(100, 255, 100), font=font_large)
        
        conf_text = f"{highlight['conf']*100:.0f}% Confidence"
        draw.text((30, height-45), conf_text, fill=(255, 255, 100), font=font_med)
        
        frame_out = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
        out.write(frame_out)
    
    # Add 0.5 second black gap
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(int(0.5 * fps)):
        out.write(black_frame)

cap.release()
out.release()

# Add audio
print("🎵 Adding audio...")
subprocess.run([
    'ffmpeg', '-y', '-i', 'temp_highlights.mp4', '-i', 'demo_input.mp4',
    '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-c:a', 'aac',
    '-shortest', 'vla_highlights.mp4'
], check=True, capture_output=True)

os.remove('temp_highlights.mp4')
print("✅ Highlights video created: vla_highlights.mp4")
print(f"📹 Duration: ~{len(HIGHLIGHTS) * 3.5:.0f} seconds")
