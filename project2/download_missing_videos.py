"""
Download the 13 missing videos
"""
import json
import yt_dlp
import os

# Load missing videos
with open('missing_videos.json') as f:
    missing = json.load(f)

print("="*70)
print(f"📥 DOWNLOADING {len(missing)} MISSING VIDEOS")
print("="*70)

def download_video(url, video_id):
    filename = f"temp_golden_{video_id}.mp4"
    
    if os.path.exists(filename) and os.path.getsize(filename) > 10000:
        return True, f"Already exists"
    
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': filename,
        'quiet': True,
        'overwrites': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}}
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        size_mb = os.path.getsize(filename) / (1024*1024)
        return True, f"{size_mb:.1f} MB"
    except:
        try:
            ydl_opts['format'] = 'worst'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            size_mb = os.path.getsize(filename) / (1024*1024)
            return True, f"{size_mb:.1f} MB"
        except Exception as e:
            return False, str(e)[:50]

success = 0
failed = 0
failed_list = []

for i, video in enumerate(missing, 1):
    url = video['url']
    title = video['title']
    video_id = url.split('=')[-1]
    
    print(f"\n[{i}/{len(missing)}] {title[:45]:45s}", end=" ")
    
    result, info = download_video(url, video_id)
    
    if result:
        print(f"✅ {info}")
        success += 1
    else:
        print(f"❌ {info}")
        failed += 1
        failed_list.append({'title': title, 'url': url})

print("\n" + "="*70)
print(f"✅ Downloaded: {success}/{len(missing)}")
print(f"❌ Failed: {failed}/{len(missing)}")
print("="*70)

if failed_list:
    print("\n⚠️  Failed videos:")
    for v in failed_list:
        print(f"   - {v['title']}")
        print(f"     {v['url']}")

print(f"\n🎯 Next: Re-run feature extraction on all 27 videos!")
