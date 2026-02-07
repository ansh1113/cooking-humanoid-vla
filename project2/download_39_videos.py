"""
STEP 1: DOWNLOAD 37 VIDEOS (NO FILTERING)
You handpicked them - we trust you!
"""
import yt_dlp
import json
import os

URLS = [
    "https://www.youtube.com/watch?v=0WnMVO3TNpM",
    "https://www.youtube.com/watch?v=xnmwP3SDzi0",
    "https://www.youtube.com/watch?v=a2RSjbpKQD0",
    "https://www.youtube.com/watch?v=suXQ2mPfhSg",
    "https://www.youtube.com/watch?v=RSpzmG2Ja5M",
    "https://www.youtube.com/watch?v=YfPawA9CQnE",
    "https://www.youtube.com/watch?v=EK0L-U6OSXg",
    "https://www.youtube.com/watch?v=2iffB1N4vL0",
    "https://www.youtube.com/watch?v=zanj3Xb0Ej0",
    "https://www.youtube.com/watch?v=Tm9G9f-k8c4",
    "https://www.youtube.com/watch?v=Ahw7E6tUOlQ",
    "https://www.youtube.com/watch?v=-pXdls87VkU",
    "https://www.youtube.com/watch?v=rEyHUS1izkI",
    "https://www.youtube.com/watch?v=GMRPgVr3xFw",
    "https://www.youtube.com/watch?v=_48ZEdLxuBs",
    "https://www.youtube.com/watch?v=6T2kQw9glyU",
    "https://www.youtube.com/watch?v=zn9H0ilvnoo",
    "https://www.youtube.com/watch?v=WoNJbTxDKCc",
    "https://www.youtube.com/watch?v=Asi-D-RSDGs",
    "https://www.youtube.com/watch?v=CIFCtazOMTA",
    "https://www.youtube.com/watch?v=XxFCoCHSCWc",
    "https://www.youtube.com/watch?v=Vtie5A1dH3E",
    "https://www.youtube.com/watch?v=xg-P7sgHpAA",
    "https://www.youtube.com/watch?v=PccHoCjNukk",
    "https://www.youtube.com/watch?v=ugxAlP-JHrE",
    "https://www.youtube.com/watch?v=C-FtLm7pYFg",
    "https://www.youtube.com/watch?v=dhyHfLNYMJA",
    "https://www.youtube.com/watch?v=3uT_otaiOMw",
    "https://www.youtube.com/watch?v=iMCtBKkRa90",
    "https://www.youtube.com/watch?v=Bg6N47eU2UI",
    "https://www.youtube.com/watch?v=us4XCgBqSIw",
    "https://www.youtube.com/watch?v=Jfyv0e9UjLU",
    "https://www.youtube.com/watch?v=4x8LGkX3kUc",
    "https://www.youtube.com/watch?v=KdKNMI2x57g",
    "https://www.youtube.com/watch?v=GNvlcIuC7HE",
    "https://www.youtube.com/watch?v=1uBnFWMOyiY",
    "https://www.youtube.com/watch?v=7GwWi1cWRfE",
]

print("="*70)
print(f"📥 DOWNLOADING {len(URLS)} HANDPICKED VIDEOS")
print("="*70)

def download_video(url):
    video_id = url.split('=')[-1].split('&')[0]
    video_path = f"temp_golden_{video_id}.mp4"
    
    if os.path.exists(video_path) and os.path.getsize(video_path) > 10000:
        return video_path, "exists"
    
    # Try different clients
    clients = ['android', 'ios', 'web_creator']
    
    for client in clients:
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': video_path,
            'quiet': True,
            'extractor_args': {'youtube': {'player_client': [client]}},
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            if os.path.exists(video_path) and os.path.getsize(video_path) > 10000:
                size_mb = os.path.getsize(video_path) / (1024*1024)
                return video_path, f"{size_mb:.1f}MB"
            else:
                if os.path.exists(video_path): 
                    os.remove(video_path)
        except:
            pass
            
    return None, "Failed"

# Download all
downloaded_videos = []
stats = {'downloaded': 0, 'exists': 0, 'failed': 0}

for i, url in enumerate(URLS, 1):
    video_id = url.split('=')[-1].split('&')[0]
    
    print(f"[{i}/{len(URLS)}] {video_id}...", end=" ", flush=True)
    
    video_path, info = download_video(url)
    
    if video_path is None:
        print(f"❌ {info}")
        stats['failed'] += 1
    elif info == "exists":
        print(f"✅ Exists")
        downloaded_videos.append({'url': url, 'id': video_id})
        stats['exists'] += 1
    else:
        print(f"✅ {info}")
        downloaded_videos.append({'url': url, 'id': video_id})
        stats['downloaded'] += 1

# Save ALL downloaded videos
with open('batch_37_videos.json', 'w') as f:
    json.dump(downloaded_videos, f, indent=2)

print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)
print(f"✅ Downloaded: {stats['downloaded']}")
print(f"✅ Already existed: {stats['exists']}")
print(f"❌ Failed: {stats['failed']}")
print(f"\n📁 Total ready: {len(downloaded_videos)} videos")
print(f"💾 Saved: batch_37_videos.json")
print(f"🎯 Estimated labels: ~{len(downloaded_videos) * 10} = {len(downloaded_videos)*10}")
print(f"\n🚀 Next: Label with sbatch")
