"""
HYDRATION SCRIPT - Extract video clips for training
Downloads videos, extracts labeled clips, deletes full videos
"""
import json
import subprocess
import os
from tqdm import tqdm
import yt_dlp

CLIPS_DIR = 'data/training_clips'
os.makedirs(CLIPS_DIR, exist_ok=True)

def download_video(url, output_path):
    """Download video temporarily"""
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}}
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False

def extract_clip(video_path, start, end, output_path):
    """Extract clip using ffmpeg"""
    duration = end - start
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start),
        '-i', video_path,
        '-t', str(duration),
        '-c', 'copy',
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.exists(output_path)

def main():
    # Load labels
    with open('data/processed/all_diverse_cooking_labels.json') as f:
        labels = json.load(f)
    
    print(f"📊 Total labels: {len(labels)}")
    
    # First, check the format
    print("\n🔍 Checking label format...")
    sample = labels[0]
    print(f"Sample keys: {list(sample.keys())}")
    
    # Filter labels that have required fields
    valid_labels = []
    for i, label in enumerate(labels):
        if 'video_url' in label and 'start' in label and 'end' in label:
            label['clip_id'] = len(valid_labels)
            valid_labels.append(label)
        else:
            print(f"⚠️  Skipping label {i}: missing video_url or timestamps")
    
    print(f"\n✅ Valid labels with timestamps: {len(valid_labels)}/{len(labels)}")
    print(f"📂 Clips directory: {CLIPS_DIR}")
    
    # Group by video URL
    from collections import defaultdict
    videos = defaultdict(list)
    for label in valid_labels:
        videos[label['video_url']].append(label)
    
    print(f"📹 Unique videos: {len(videos)}")
    print(f"\n🔧 Starting hydration...")
    
    success_count = 0
    fail_count = 0
    
    for video_url, clips_data in tqdm(videos.items(), desc="Videos"):
        temp_video = 'temp_hydrate.mp4'
        
        # Download video
        if not download_video(video_url, temp_video):
            print(f"\n  ❌ Failed to download: {video_url[:50]}...")
            fail_count += len(clips_data)
            continue
        
        # Extract each clip
        for clip_data in clips_data:
            clip_id = clip_data['clip_id']
            clip_path = f"{CLIPS_DIR}/clip_{clip_id:04d}.mp4"
            
            if extract_clip(temp_video, clip_data['start'], clip_data['end'], clip_path):
                clip_data['clip_path'] = clip_path
                success_count += 1
            else:
                fail_count += 1
        
        # Delete temp video
        if os.path.exists(temp_video):
            os.remove(temp_video)
    
    # Save updated labels with clip paths
    with open('data/processed/training_labels_with_clips.json', 'w') as f:
        json.dump(valid_labels, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ HYDRATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Success: {success_count}/{len(valid_labels)}")
    print(f"Failed:  {fail_count}/{len(valid_labels)}")
    print(f"📂 Clips saved to: {CLIPS_DIR}")
    print(f"📄 Labels updated: data/processed/training_labels_with_clips.json")
    
    # Check total size
    if os.path.exists(CLIPS_DIR):
        clips = [f for f in os.listdir(CLIPS_DIR) if f.endswith('.mp4')]
        if clips:
            total_size = sum(os.path.getsize(f"{CLIPS_DIR}/{f}") for f in clips)
            print(f"💾 Total clip size: {total_size / 1024**3:.2f} GB")
            print(f"📊 Average clip size: {total_size / len(clips) / 1024**2:.1f} MB")

if __name__ == "__main__":
    main()
