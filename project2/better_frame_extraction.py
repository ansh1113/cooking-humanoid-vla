"""
Better frame extraction: Skip intro, sample from cooking portion
"""
import cv2
import numpy as np

def extract_frames_smart(video_path, max_frames=60):
    """
    Extract frames AFTER skipping intro
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return []
    
    # Get video stats
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"   Video: {duration:.1f}s, {total_frames} frames, {fps:.1f} fps")
    
    # SKIP FIRST 20% (intro/titles)
    # SAMPLE FROM MIDDLE 60% (actual cooking)
    start_frame = int(total_frames * 0.2)  # Skip first 20%
    end_frame = int(total_frames * 0.8)    # Skip last 20% (outro)
    
    useful_frames = end_frame - start_frame
    
    # Sample uniformly from cooking portion
    if useful_frames < max_frames:
        indices = range(start_frame, end_frame)
    else:
        step = useful_frames / max_frames
        indices = [int(start_frame + i * step) for i in range(max_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    
    print(f"   Extracted {len(frames)} frames from cooking portion ({start_frame}-{end_frame})")
    
    return frames

# Test on Butter Chicken video
if __name__ == "__main__":
    from stage1_generate_plan import download_youtube_video
    
    url = "https://www.youtube.com/watch?v=a03U45jFxOI"
    
    print("Downloading...")
    result = download_youtube_video(url)
    if isinstance(result, tuple):
        video_path, title = result
    else:
        video_path = result
    
    print("Extracting frames WITH SMART SAMPLING...")
    frames = extract_frames_smart(video_path, max_frames=60)
    
    # Sample 8 for GPT-4V
    n = len(frames)
    indices = [0, n//7, 2*n//7, 3*n//7, 4*n//7, 5*n//7, 6*n//7, n-1]
    sampled = [frames[i] for i in indices]
    
    # Save
    import os
    os.makedirs('debug_frames_smart', exist_ok=True)
    for i, frame in enumerate(sampled):
        cv2.imwrite(f'debug_frames_smart/frame_{i}.jpg', frame)
    
    print(f"\nSaved {len(sampled)} frames to debug_frames_smart/")
    print("Check these - they should show ACTUAL COOKING!")
