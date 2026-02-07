"""
DOWNLOAD 35 MORE VIDEOS (NO FILTERING)
"""
import yt_dlp
import json
import os

URLS = [
    "https://www.youtube.com/watch?v=OPVUipeNZGY",
    "https://www.youtube.com/watch?v=W_ByyV9QmDM",
    "https://www.youtube.com/watch?v=qVu0krY7jeg",
    "https://www.youtube.com/watch?v=_zmiB6kaDGU",
    "https://www.youtube.com/watch?v=jc7-i8n03Cw",
    "https://www.youtube.com/watch?v=qooJlBCziNs",
    "https://www.youtube.com/watch?v=dRBJzyj9GFY",
    "https://www.youtube.com/watch?v=QZXi8oIHZ54",
    "https://www.youtube.com/watch?v=7EQgUdAoUkw",
    "https://www.youtube.com/watch?v=RcKEHz6m15U",
    "https://www.youtube.com/watch?v=WWHeI8bDqLc",
    "https://www.youtube.com/watch?v=2_q5pybV7mg",
    "https://www.youtube.com/watch?v=C5dylORppOo",
    "https://www.youtube.com/watch?v=yEMwra2ZjPE",
    "https://www.youtube.com/watch?v=25O-VoDp_W4",
    "https://www.youtube.com/watch?v=lRksMJe6qj0",
    "https://www.youtube.com/watch?v=EKPMkp9pAgk",
    "https://www.youtube.com/watch?v=4tUI1v6c5Ig",
    "https://www.youtube.com/watch?v=Y92EUVy5kdk",
    "https://www.youtube.com/watch?v=ojx6R6Qu1XI",
    "https://www.youtube.com/watch?v=tnEH4dqVeiw",
    "https://www.youtube.com/watch?v=AP-APzaEIbM",
    "https://www.youtube.com/watch?v=6szbQlhMrzE",
    "https://www.youtube.com/watch?v=fQiJThYdWeo",
    "https://www.youtube.com/watch?v=fCZxiLuSARM",
    "https://www.youtube.com/watch?v=BTiHI8APHGE",
    "https://www.youtube.com/watch?v=NhQnORKs4Ww",
    "https://www.youtube.com/watch?v=ev8_kta31Qc",
    "https://www.youtube.com/watch?v=pUn2CI8k_NI",
    "https://www.youtube.com/watch?v=6tCgVb4sMNw",
    "https://www.youtube.com/watch?v=2t66OO5P-Cw",
    "https://www.youtube.com/watch?v=3MaTbDZYcz8",
    "https://www.youtube.com/watch?v=esGf_mxlOpw",
    "https://www.youtube.com/watch?v=Q3rjDvN6CHA",
]

print("="*70)
print(f"📥 DOWNLOADING {len(URLS)} HANDPICKED VIDEOS")
print("="*70)

def download_video(url):
    video_id = url.split('=')[-1].split('&')[0]
    video_path = f"temp_golden_{video_id}.mp4"
    
    if os.path.exists(video_path) and os.path.getsize(video_path) > 10000:
        return video_path, "exists"
    
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

with open('batch_35_videos.json', 'w') as f:
    json.dump(downloaded_videos, f, indent=2)

print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)
print(f"✅ Downloaded: {stats['downloaded']}")
print(f"✅ Already existed: {stats['exists']}")
print(f"❌ Failed: {stats['failed']}")
print(f"\n📁 Total ready: {len(downloaded_videos)} videos")
print(f"💾 Saved: batch_35_videos.json")
print(f"🎯 Estimated labels: ~{len(downloaded_videos) * 10}")
