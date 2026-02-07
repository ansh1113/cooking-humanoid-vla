"""
convert_videos.py - Convert AV1 videos to H264 for OpenCV compatibility
"""
import subprocess
from pathlib import Path
import json

def convert_video(input_path, output_path):
    """Convert video to H264 codec using ffmpeg"""
    cmd = [
        '/u/anshb3/bin/ffmpeg',
        '-i', str(input_path),
        '-c:v', 'libx264',  # H264 codec
        '-c:a', 'aac',      # AAC audio
        '-crf', '23',       # Quality (23 is good)
        '-preset', 'fast',  # Encoding speed
        '-y',               # Overwrite
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("="*70)
    print("🎬 CONVERTING VIDEOS TO H264")
    print("="*70)
    
    input_dir = Path('data/raw/videos')
    output_dir = Path('data/processed/videos_h264')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = list(input_dir.glob('*.mp4'))
    
    print(f"Found {len(video_files)} videos to convert")
    
    converted = []
    
    for video_path in video_files:
        output_path = output_dir / video_path.name
        
        if output_path.exists():
            print(f"⏭️  Skipping {video_path.name} (already converted)")
            converted.append(str(output_path))
            continue
        
        print(f"🔄 Converting: {video_path.name}")
        
        if convert_video(video_path, output_path):
            print(f"   ✅ Success!")
            converted.append(str(output_path))
        else:
            print(f"   ❌ Failed")
    
    print("\n" + "="*70)
    print(f"✅ Converted {len(converted)}/{len(video_files)} videos")
    print(f"📁 Saved to: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
