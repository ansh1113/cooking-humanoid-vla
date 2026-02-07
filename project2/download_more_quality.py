"""
Download MORE high-quality videos - adjusted strategy
Focus on what actually works on YouTube
"""
import yt_dlp
import json
from pathlib import Path

# ADJUSTED keywords - based on what YouTube actually has
WORKING_KEYWORDS = [
    "cooking no talking asmr",
    "silent cooking tutorial",
    "cooking sounds only",
    "satisfying cooking video",
    "cooking process video",
    "how to cook step by step",
    "cooking demonstration pov",
    "meal prep satisfying",
    "knife skills tutorial",
    "vegetable cutting skills",
    "food preparation video",
    "cooking techniques basic",
    "cooking skills lesson",
    "quick cooking recipe",
    "simple cooking method",
    "cooking basics step by step",
    "kitchen skills tutorial",
    "professional cooking techniques",
    "culinary skills basic",
    "home cooking tutorial"
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
        'match_filter': lambda info: (
            None if (
                120 <= info.get('duration', 0) <= 600 and
                info.get('view_count', 0) > 500  # Lower threshold
            ) else 'Does not meet quality criteria'
        ),
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
                'view_count': info.get('view_count', 0),
                'success': True
            }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def search_and_download(query, max_results=3, output_dir='videos'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    search_opts = {
        'default_search': 'ytsearch',
        'max_downloads': max_results,
        'quiet': True,
    }
    
    print(f"\n🔍 {query}")
    
    with yt_dlp.YoutubeDL(search_opts) as ydl:
        try:
            search_results = ydl.extract_info(f"ytsearch{max_results*3}:{query}", download=False)
            
            videos = []
            for entry in search_results['entries']:
                if len(videos) >= max_results:
                    break
                if entry and 120 <= entry.get('duration', 0) <= 600:
                    url = f"https://www.youtube.com/watch?v={entry['id']}"
                    result = download_video(url, output_dir)
                    if result['success']:
                        videos.append(result)
                        print(f"   ✅ {result['title'][:60]} ({result['duration']//60}:{result['duration']%60:02d})")
            return videos
        except Exception as e:
            print(f"   ❌ {e}")
            return []

def main():
    print("="*70)
    print("📺 DOWNLOADING MORE HIGH-QUALITY VIDEOS (TARGET: 32 MORE)")
    print("="*70)
    
    output_dir = 'data/raw/videos_quality'
    
    # Check existing
    existing_files = list(Path(output_dir).glob('*.mp4'))
    print(f"Already have: {len(existing_files)} videos")
    
    all_videos = []
    target = 32  # 18 + 32 = 50 total
    
    for keyword in WORKING_KEYWORDS:
        if len(all_videos) >= target:
            break
        needed = min(3, target - len(all_videos))
        videos = search_and_download(keyword, max_results=needed, output_dir=output_dir)
        all_videos.extend(videos)
        print(f"   Progress: {len(all_videos)}/{target} (Total: {len(existing_files) + len(all_videos)})")
    
    # Save metadata (append to existing)
    metadata_file = 'data/raw/video_metadata_quality.json'
    if Path(metadata_file).exists():
        with open(metadata_file, 'r') as f:
            existing = json.load(f)
    else:
        existing = []
    
    existing.extend(all_videos)
    
    with open(metadata_file, 'w') as f:
        json.dump(existing, f, indent=2)
    
    print(f"\n✅ Downloaded {len(all_videos)} new videos")
    print(f"📊 Total quality videos: {len(existing_files) + len(all_videos)}")

if __name__ == "__main__":
    main()
