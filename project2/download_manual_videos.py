"""
Download manually curated videos
"""
import yt_dlp
import os

VIDEOS = [
    "https://www.youtube.com/watch?v=gBnHovVnNwo",
    "https://www.youtube.com/watch?v=gQWF6ww8qU4",
    "https://www.youtube.com/watch?v=DZtqRaHYsf8",
    "https://www.youtube.com/watch?v=8c_scYUN5uc",
    "https://www.youtube.com/watch?v=peF9ha21rP4",
    "https://www.youtube.com/watch?v=F6Czd-2dwN0",
    "https://www.youtube.com/watch?v=fNdmxfdNkQc",
    "https://www.youtube.com/watch?v=Tn1UeCKOACU",
    "https://www.youtube.com/watch?v=1DbrPChGnpk",
    "https://www.youtube.com/watch?v=nPi2GD2SqfQ",
    "https://www.youtube.com/watch?v=9vB0AKrWw50",
    "https://www.youtube.com/watch?v=0fqJKPpLwM0",
    "https://www.youtube.com/watch?v=HJklEsk3fIA",
    "https://www.youtube.com/watch?v=olGR18M8RMQ",
    "https://www.youtube.com/watch?v=q2-k5ew2SIs",
    "https://www.youtube.com/watch?v=rSfhuO5hvX0",
]

def download_video(url):
    video_id = url.split('=')[-1]
    filename = f"temp_golden_{video_id}.mp4"
    
    if os.path.exists(filename) and os.path.getsize(filename) > 10000:
        size_mb = os.path.getsize(filename) / (1024*1024)
        return True, f"Already exists ({size_mb:.1f} MB)"
    
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': filename,
        'quiet': True,
        'overwrites': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}}
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown')
            size_mb = os.path.getsize(filename) / (1024*1024)
            return True, f"{title[:40]} ({size_mb:.1f} MB)"
    except Exception as e:
        # Try worst quality
        try:
            ydl_opts['format'] = 'worst'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Unknown')
                size_mb = os.path.getsize(filename) / (1024*1024)
                return True, f"{title[:40]} ({size_mb:.1f} MB)"
        except:
            return False, str(e)[:50]

print("="*70)
print("📥 DOWNLOADING 16 MANUALLY CURATED VIDEOS")
print("="*70)

success = 0
failed = 0
failed_urls = []

for i, url in enumerate(VIDEOS, 1):
    print(f"\n[{i}/{len(VIDEOS)}] {url[-15:]}", end=" ")
    
    result, info = download_video(url)
    
    if result:
        print(f"✅ {info}")
        success += 1
    else:
        print(f"❌ {info}")
        failed += 1
        failed_urls.append(url)

print("\n" + "="*70)
print(f"✅ Successfully downloaded: {success}/{len(VIDEOS)}")
print(f"❌ Failed: {failed}/{len(VIDEOS)}")
print("="*70)

if failed_urls:
    print("\n⚠️  Failed URLs:")
    for url in failed_urls:
        print(f"   {url}")

print(f"\n🎯 Next: Run labeling script on all downloaded videos!")
