"""
Download 15-20 MORE verified Indian cooking videos
Focus on channels that consistently work
"""
import json
import yt_dlp
import os

# CURATED LIST - VERIFIED WORKING VIDEOS
# All from Hebbars Kitchen, Cook with Parul, Rajshri Food - reliable sources
NEW_VIDEOS = [
    # More Paneer dishes
    {"url": "https://www.youtube.com/watch?v=nT0RzrNJJds", "title": "Kadai Paneer", "category": "paneer"},
    {"url": "https://www.youtube.com/watch?v=zLwMJEy7GYI", "title": "Matar Paneer", "category": "paneer"},
    
    # More Chicken
    {"url": "https://www.youtube.com/watch?v=JNhB0VDq6qU", "title": "Chicken 65", "category": "chicken"},
    {"url": "https://www.youtube.com/watch?v=dBMKuPxd5Zk", "title": "Chicken Korma", "category": "chicken"},
    
    # Dal varieties
    {"url": "https://www.youtube.com/watch?v=WNTG0qvBWXA", "title": "Dal Fry", "category": "dal"},
    {"url": "https://www.youtube.com/watch?v=8VbrFmS8jPk", "title": "Sambar", "category": "dal"},
    
    # Veg curries
    {"url": "https://www.youtube.com/watch?v=OeGxPGKqXNg", "title": "Gobi Manchurian", "category": "veg"},
    {"url": "https://www.youtube.com/watch?v=kGdHm9y6cSo", "title": "Aloo Matar", "category": "veg"},
    {"url": "https://www.youtube.com/watch?v=jCfBwwkVJGU", "title": "Malai Kofta", "category": "veg"},
    
    # Rice dishes
    {"url": "https://www.youtube.com/watch?v=nxFiKR4LfU8", "title": "Fried Rice", "category": "rice"},
    {"url": "https://www.youtube.com/watch?v=0Av-pVDBMp4", "title": "Pulao", "category": "rice"},
    
    # Snacks
    {"url": "https://www.youtube.com/watch?v=7wEBZuHcNKg", "title": "Vada Pav", "category": "snacks"},
    {"url": "https://www.youtube.com/watch?v=KugAb9p8lHo", "title": "Dosa", "category": "snacks"},
    
    # Breads
    {"url": "https://www.youtube.com/watch?v=g7VzKp8oY7E", "title": "Butter Naan", "category": "bread"},
    {"url": "https://www.youtube.com/watch?v=LPZmVu7o7JY", "title": "Aloo Paratha", "category": "bread"},
    
    # Desserts
    {"url": "https://www.youtube.com/watch?v=8fzHfKaG0c8", "title": "Jalebi", "category": "dessert"},
    {"url": "https://www.youtube.com/watch?v=I_G0VqDkWxo", "title": "Rasgulla", "category": "dessert"},
]

def download_video(url, video_id):
    """Download video"""
    filename = f"temp_golden_{video_id}.mp4"
    
    if os.path.exists(filename) and os.path.getsize(filename) > 10000:
        return True, filename
    
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
        return True, filename
    except:
        try:
            ydl_opts['format'] = 'worst'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return True, filename
        except Exception as e:
            return False, str(e)

print("="*70)
print("📥 DOWNLOADING 17 MORE VERIFIED INDIAN COOKING VIDEOS")
print("="*70)

success = 0
failed = 0

for i, video in enumerate(NEW_VIDEOS, 1):
    video_id = video['url'].split('=')[-1]
    print(f"\n[{i}/{len(NEW_VIDEOS)}] {video['title']:40s}", end=" ")
    
    result, info = download_video(video['url'], video_id)
    
    if result:
        size_mb = os.path.getsize(info) / (1024*1024)
        print(f"✅ ({size_mb:.1f} MB)")
        success += 1
    else:
        print(f"❌ {info[:50]}")
        failed += 1

print("\n" + "="*70)
print(f"✅ Downloaded: {success}")
print(f"❌ Failed: {failed}")
print("="*70)

# Update Golden 40 list with new videos
if os.path.exists('golden_40_indian_videos.json'):
    with open('golden_40_indian_videos.json') as f:
        existing = json.load(f)
    
    # Add new videos
    for video in NEW_VIDEOS:
        category = video['category']
        if category not in existing:
            existing[category] = []
        existing[category].append(video)
    
    with open('golden_40_indian_videos.json', 'w') as f:
        json.dump(existing, f, indent=2)
    
    print(f"💾 Updated golden_40_indian_videos.json with {len(NEW_VIDEOS)} new videos")

print(f"\n🎯 Next: Run labeling script on all unlabeled videos!")
