"""
STAGE 1: FAST DOWNLOADER 🚀
Downloads all videos on the high-speed login node.
"""
import json
import os
import yt_dlp
import concurrent.futures

# CONFIG
VIDEO_LIST = 'golden_40_indian_videos.json'

def download_video(item):
    url = item['url']
    vid_id = url.split('=')[-1]
    filename = f"temp_golden_{vid_id}.mp4"
    
    if os.path.exists(filename) and os.path.getsize(filename) > 10000:
        print(f"✅ Already exists: {filename}")
        return
        
    print(f"📥 Downloading: {item['title']}...")
    
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': filename,
        'quiet': True,
        'overwrites': True,
        # Use Android client for stability
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}}
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"✨ Downloaded: {filename}")
    except Exception as e:
        print(f"❌ Failed {url}: {str(e)}")

def main():
    print("="*60)
    print("🚀 HIGH-SPEED VIDEO DOWNLOADER")
    print("="*60)
    
    with open(VIDEO_LIST) as f:
        data = json.load(f)
        
    all_videos = []
    for cat in data.values():
        all_videos.extend(cat)
        
    print(f"📊 Queue: {len(all_videos)} videos")
    
    # Download in parallel (4 threads) to max out bandwidth
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(download_video, all_videos)
        
    print("\n✅ All downloads complete. Ready for Stage 2 (Processing).")

if __name__ == "__main__":
    main()
