"""
Download with cookies + delays to bypass 403
"""
import yt_dlp
import json
from pathlib import Path
import time
import random

COOKIES_FILE = 'cookies.txt'

KEYWORDS = [
    "cooking no talking asmr",
    "silent cooking tutorial", 
    "cooking sounds only",
    "satisfying cooking video",
    "meal prep satisfying",
    "knife skills tutorial",
    "vegetable cutting skills",
    "food preparation video",
    "quick cooking recipe",
    "simple cooking method"
]

def download_video(url, output_dir):
    ydl_opts = {
        'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'writesubtitles': True,
        'subtitleslangs': ['en'],
        'quiet': False,
        'cookiefile': COOKIES_FILE if Path(COOKIES_FILE).exists() else None,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
    }
    
    for attempt in range(3):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return {'id': info['id'], 'title': info['title'], 'duration': info['duration'], 'success': True}
        except Exception as e:
            print(f"   ⚠️  Retry {attempt+1}/3")
            time.sleep(5)
    
    return {'success': False}

def main():
    print("="*70)
    print("📺 STEALTH DOWNLOAD WITH COOKIES")
    print("="*70)
    
    if not Path(COOKIES_FILE).exists():
        print("❌ NO cookies.txt! Create it first!")
        return
    
    print("✅ Cookies found!")
    
    output_dir = 'data/raw/videos_quality'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    existing = list(Path(output_dir).glob('*.mp4'))
    print(f"Already have: {len(existing)} videos\n")
    
    all_videos = []
    target = 30
    
    for keyword in KEYWORDS:
        if len(all_videos) >= target:
            break
        
        print(f"\n🔍 {keyword}")
        
        search_opts = {'default_search': 'ytsearch', 'quiet': True, 'cookiefile': COOKIES_FILE}
        
        with yt_dlp.YoutubeDL(search_opts) as ydl:
            try:
                results = ydl.extract_info(f"ytsearch10:{keyword}", download=False)
                
                for entry in results['entries']:
                    if len(all_videos) >= target:
                        break
                    if entry and 120 <= entry.get('duration', 0) <= 600:
                        url = f"https://www.youtube.com/watch?v={entry['id']}"
                        
                        # Delay to look human
                        time.sleep(random.uniform(3, 7))
                        
                        print(f"   📥 {entry['title'][:50]}...")
                        result = download_video(url, output_dir)
                        
                        if result['success']:
                            all_videos.append(result)
                            print(f"   ✅ Success! Total: {len(existing) + len(all_videos)}")
                
            except Exception as e:
                print(f"   ❌ {e}")
    
    with open('data/raw/video_metadata_stealth.json', 'w') as f:
        json.dump(all_videos, f, indent=2)
    
    print(f"\n✅ Downloaded {len(all_videos)} new videos")
    print(f"📊 Grand total: {len(existing) + len(all_videos)}")

if __name__ == "__main__":
    main()
