"""
Convert ALL videos from all directories to H264
"""
import subprocess
from pathlib import Path

def convert_video(input_path, output_path):
    cmd = [
        '/u/anshb3/bin/ffmpeg',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-crf', '23',
        '-preset', 'fast',
        '-y',
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except:
        return False

def main():
    print("="*70)
    print("🎬 CONVERTING ALL 182 VIDEOS TO H264")
    print("="*70)
    
    # Find all videos from all directories
    video_dirs = [
        'data/raw/videos',
        'data/raw/videos_high_quality',
        'data/raw/videos_extended'
    ]
    
    output_dir = Path('data/processed/videos_h264_all')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_videos = []
    for video_dir in video_dirs:
        vdir = Path(video_dir)
        if vdir.exists():
            all_videos.extend(list(vdir.glob('*.mp4')))
    
    print(f"Found {len(all_videos)} total videos\n")
    
    converted = 0
    skipped = 0
    failed = 0
    
    for i, video_path in enumerate(all_videos, 1):
        output_path = output_dir / video_path.name
        
        if output_path.exists():
            print(f"[{i}/{len(all_videos)}] ⏭️  Skip: {video_path.name}")
            skipped += 1
            continue
        
        print(f"[{i}/{len(all_videos)}] 🔄 Converting: {video_path.name}")
        if convert_video(video_path, output_path):
            print(f"             ✅ Success!")
            converted += 1
        else:
            print(f"             ❌ Failed")
            failed += 1
    
    print("\n" + "="*70)
    print(f"✅ Converted: {converted}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {converted + skipped}/{len(all_videos)}")
    print("="*70)

if __name__ == "__main__":
    main()
