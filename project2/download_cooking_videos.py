"""
Download 50 cooking videos for full training
"""
import yt_dlp
import json
from pathlib import Path

COOKING_KEYWORDS = [
    "how to chop vegetables",
    "knife skills cooking", 
    "basic cooking techniques",
    "how to slice tomato",
    "cooking basics tutorial",
    "simple salad recipe",
    "easy pasta recipe",
    "how to cook eggs",
    "basic knife skills",
    "vegetable prep",
    "simple cooking",
    "beginner cooking",
    "quick recipe",
    "easy meal prep",
    "cooking fundamentals"
]

def download_video(url, output_dir):
    ydl_opts = {
        'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'skip_download': False,
        'quiet': False,
        'match_filter': lambda info: None if info.get('duration', 0) <= 600 else 'Video too long',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                return {'success': False}
            return {
                'id': info['id'],
                'title': info['title'],
                'duration': info['duration'],
                'success': True
            }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def search_and_download(query, max_results=5, output_dir='videos'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    search_opts = {
        'default_search': 'ytsearch',
        'max_downloads': max_results,
        'quiet': True,
    }
    
    print(f"\n🔍 {query}")
    
    with yt_dlp.YoutubeDL(search_opts) as ydl:
        try:
            search_results = ydl.extract_info(f"ytsearch{max_results*2}:{query}", download=False)
            
            videos = []
            for entry in search_results['entries']:
                if len(videos) >= max_results:
                    break
                if entry and entry.get('duration', 0) <= 600:
                    url = f"https://www.youtube.com/watch?v={entry['id']}"
                    result = download_video(url, output_dir)
                    if result['success']:
                        videos.append(result)
                        print(f"   ✅ {result['title'][:50]}")
            return videos
        except Exception as e:
            print(f"   ❌ {e}")
            return []

def main():
    print("="*70)
    print("📺 DOWNLOADING 50 VIDEOS FOR FULL TRAINING")
    print("="*70)
    
    output_dir = 'data/raw/videos'
    all_videos = []
    
    for keyword in COOKING_KEYWORDS:
        if len(all_videos) >= 50:
            break
        needed = min(4, 50 - len(all_videos))
        videos = search_and_download(keyword, max_results=needed, output_dir=output_dir)
        all_videos.extend(videos)
        print(f"   Progress: {len(all_videos)}/50")
    
    with open('data/raw/video_metadata.json', 'w') as f:
        json.dump(all_videos, f, indent=2)
    
    print(f"\n✅ Downloaded {len(all_videos)} videos")

if __name__ == "__main__":
    main()
