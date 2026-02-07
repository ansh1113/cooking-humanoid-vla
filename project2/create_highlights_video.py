"""
Create a montage of HIGH-CONFIDENCE predictions
Shows the best moments from multiple recipes
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess
import os  # ← THIS WAS MISSING!

# High-confidence moments from your logs
HIGHLIGHTS = [
    {
        'video': 'Paneer Butter Masala',
        'url': 'https://www.youtube.com/watch?v=CXvznff0cMs',
        'timestamp': 63,  # seconds
        'action': 'Pour(Oil)',
        'confidence': 0.87,
        'emoji': '🫗'
    },
    {
        'video': 'Paneer Butter Masala',
        'url': 'https://www.youtube.com/watch?v=CXvznff0cMs',
        'timestamp': 106,
        'action': 'Process(Grinder)',
        'confidence': 0.90,
        'emoji': '⚙️'
    },
    {
        'video': 'Paneer Butter Masala',
        'url': 'https://www.youtube.com/watch?v=CXvznff0cMs',
        'timestamp': 255,
        'action': 'Sprinkle(Masala)',
        'confidence': 0.92,
        'emoji': '🌶️'
    },
    {
        'video': 'Masala Khichdi',
        'url': 'https://www.youtube.com/watch?v=_JUcqjCKhHc',
        'timestamp': 46,
        'action': 'Pour(Water)',
        'confidence': 0.96,
        'emoji': '💧'
    },
]

print("🎬 Creating highlights montage...")

clips = []

for i, highlight in enumerate(HIGHLIGHTS):
    print(f"📹 Processing highlight {i+1}/{len(HIGHLIGHTS)}: {highlight['action']}")
    
    # Download 3-second clip around timestamp
    video_file = f"highlight_{i}.mp4"
    if not os.path.exists(video_file):
        start_time = max(0, highlight['timestamp'] - 1)
        duration = 3
        
        subprocess.run([
            'yt-dlp', '-f', '18', 
            '--download-sections', f"*{start_time}-{start_time + duration}",
            '-o', video_file, 
            highlight['url']
        ], check=True, stderr=subprocess.DEVNULL)
    
    # Read video
    cap = cv2.VideoCapture(video_file)
    frames = []
    
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add text overlay
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_frame)
        
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
            font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except:
            font_large = ImageFont.load_default()
            font_med = ImageFont.load_default()
        
        # Draw semi-transparent background for text
        overlay = Image.new('RGBA', pil_frame.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.rectangle([(0, frame_height-100), (frame_width, frame_height)], 
                              fill=(0, 0, 0, 180))
        pil_frame = pil_frame.convert('RGBA')
        pil_frame = Image.alpha_composite(pil_frame, overlay)
        pil_frame = pil_frame.convert('RGB')
        draw = ImageDraw.Draw(pil_frame)
        
        # Action text
        text = f"{highlight['emoji']} {highlight['action']}"
        draw.text((20, frame_height-80), text, fill=(100, 255, 100), font=font_large)
        
        # Confidence
        conf_text = f"{highlight['confidence']*100:.0f}% confidence"
        draw.text((20, frame_height-40), conf_text, fill=(255, 255, 100), font=font_med)
        
        frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
        frames.append(frame)
    
    cap.release()
    clips.append(frames)

# Combine all clips
print("🎞️ Combining clips...")
if clips:
    height, width = clips[0][0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('temp_highlights.mp4', fourcc, 24, (width, height))
    
    for frames in clips:
        for frame in frames:
            out.write(frame)
        
        # Add 0.3 second black transition
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(7):
            out.write(black_frame)
    
    out.release()
    
    # Add audio from first clip
    print("🎵 Adding audio...")
    subprocess.run([
        'ffmpeg', '-y', '-i', 'temp_highlights.mp4', '-i', 'highlight_0.mp4',
        '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac',
        '-shortest', 'vla_highlights.mp4'
    ], check=True, stderr=subprocess.DEVNULL)
    
    os.remove('temp_highlights.mp4')
    print("✅ Highlights video created: vla_highlights.mp4")
else:
    print("❌ No clips processed")
