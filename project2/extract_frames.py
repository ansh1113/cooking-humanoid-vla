"""
extract_frames.py - Extract frames from videos for CLIP processing
"""
import cv2
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

def extract_frames(video_path, output_dir, fps=1):
    """
    Extract frames from video at specified FPS
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: Frames per second to extract (1 = 1 frame/second)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frames_extracted = []
    frame_count = 0
    saved_count = 0
    
    print(f"📹 Processing: {video_path.name}")
    print(f"   Video FPS: {video_fps:.1f}, Extracting every {frame_interval} frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps
            frame_filename = f"frame_{saved_count:06d}_t{timestamp:.2f}s.jpg"
            frame_path = output_dir / frame_filename
            
            cv2.imwrite(str(frame_path), frame)
            frames_extracted.append({
                'frame_id': saved_count,
                'timestamp': timestamp,
                'path': str(frame_path.relative_to(output_dir.parent.parent))
            })
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"   ✅ Extracted {saved_count} frames")
    return frames_extracted

def process_all_videos(video_dir='data/processed/videos_h264', output_dir='data/processed/frames'):
    """Process all downloaded videos"""
    
    video_dir = Path(video_dir)
    video_files = list(video_dir.glob('*.mp4'))
    
    print("="*70)
    print(f"🎬 FRAME EXTRACTION")
    print("="*70)
    print(f"Found {len(video_files)} videos")
    
    all_frames = {}
    
    for video_path in video_files:
        video_id = video_path.stem
        video_output_dir = Path(output_dir) / video_id
        
        frames = extract_frames(video_path, video_output_dir, fps=1)
        all_frames[video_id] = {
            'video_path': str(video_path),
            'num_frames': len(frames),
            'frames': frames
        }
    
    # Save frame metadata
    metadata_file = Path(output_dir) / 'frame_metadata.json'
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(all_frames, f, indent=2)
    
    total_frames = sum(v['num_frames'] for v in all_frames.values())
    
    print("\n" + "="*70)
    print(f"✅ Extracted {total_frames} total frames from {len(video_files)} videos")
    print(f"💾 Metadata saved to: {metadata_file}")
    print("="*70)

if __name__ == "__main__":
    process_all_videos()
