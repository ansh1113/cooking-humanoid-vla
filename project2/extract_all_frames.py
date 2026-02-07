"""
Extract frames from ALL 156 H264 videos
"""
import cv2
import json
from pathlib import Path

def extract_frames(video_path, output_dir, fps=1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frames_extracted = []
    frame_count = 0
    saved_count = 0
    
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
    return frames_extracted

def main():
    print("="*70)
    print("🎬 EXTRACTING FRAMES FROM 156 VIDEOS")
    print("="*70)
    
    video_dir = Path('data/processed/videos_h264_all')
    video_files = sorted(video_dir.glob('*.mp4'))
    
    print(f"Found {len(video_files)} videos\n")
    
    all_frames = {}
    
    for i, video_path in enumerate(video_files, 1):
        video_id = video_path.stem
        video_output_dir = Path('data/processed/frames_all') / video_id
        
        print(f"[{i}/{len(video_files)}] 📹 {video_path.name}")
        
        frames = extract_frames(video_path, video_output_dir, fps=1)
        all_frames[video_id] = {
            'video_path': str(video_path),
            'num_frames': len(frames),
            'frames': frames
        }
        
        print(f"             ✅ {len(frames)} frames")
    
    # Save metadata
    metadata_file = Path('data/processed/frames_all/frame_metadata.json')
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(all_frames, f, indent=2)
    
    total_frames = sum(v['num_frames'] for v in all_frames.values())
    
    print("\n" + "="*70)
    print(f"✅ Extracted {total_frames} frames from {len(video_files)} videos")
    print(f"💾 Saved to: data/processed/frames_all/")
    print("="*70)

if __name__ == "__main__":
    main()
