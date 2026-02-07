"""
Download HIGH-QUALITY action-focused cooking videos
Target: Short demos, minimal talking, clear actions
"""
import yt_dlp
import json
from pathlib import Path

# BETTER search keywords - action-focused
QUALITY_KEYWORDS = [
    "cooking demonstration no talking",
    "silent cooking video",
    "cooking asmr no music",
    "how to chop vegetables demonstration",
    "knife skills silent",
    "food prep timelapse",
    "cooking technique demonstration",
    "basic cooking skills silent",
    "meal prep no talking",
    "cooking tutorial silent",
    "food preparation technique",
    "knife skills tutorial quiet",
    "cooking basics demonstration",
    "silent recipe tutorial",
    "cooking process video"
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
        # STRICT filters for quality
        'match_filter': lambda info: (
            None if (
                120 <= info.get('duration', 0) <= 600 and  # 2-10 min only
                info.get('view_count', 0) > 1000  # Some popularity filter
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

def search_and_download(query, max_results=4, output_dir='videos'):
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
    print("📺 DOWNLOADING 50 HIGH-QUALITY ACTION-FOCUSED VIDEOS")
    print("="*70)
    
    # Use SEPARATE directory for quality videos
    output_dir = 'data/raw/videos_quality'
    all_videos = []
    
    for keyword in QUALITY_KEYWORDS:
        if len(all_videos) >= 50:
            break
        needed = min(4, 50 - len(all_videos))
        videos = search_and_download(keyword, max_results=needed, output_dir=output_dir)
        all_videos.extend(videos)
        print(f"   Progress: {len(all_videos)}/50")
    
    with open('data/raw/video_metadata_quality.json', 'w') as f:
        json.dump(all_videos, f, indent=2)
    
    print(f"\n✅ Downloaded {len(all_videos)} quality videos")
    print(f"📁 Saved to: {output_dir}")

if __name__ == "__main__":
    main()
