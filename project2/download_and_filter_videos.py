"""
PART 1: DOWNLOAD & FILTER VIDEOS
Run interactively on login node (~30-60 min)
Scrapes channels, downloads, filters with Whisper
"""
import yt_dlp
import whisper
import json
import os
import subprocess

print("="*70)
print("📥 AUTOMATED VIDEO DOWNLOAD & FILTER")
print("="*70)

# ============================================================================
# SCRAPE CHANNEL
# ============================================================================
def scrape_channel(channel_url, max_videos=100):
    print(f"\n📡 Scraping: {channel_url}")
    
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'playlistend': max_videos,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(channel_url, download=False)
            
            if 'entries' not in info:
                return []
            
            videos = []
            for entry in info['entries']:
                if entry and 'id' in entry:
                    videos.append({
                        'url': f"https://www.youtube.com/watch?v={entry['id']}",
                        'title': entry.get('title', 'Unknown'),
                        'duration': entry.get('duration', 0),
                        'id': entry['id']
                    })
            
            print(f"   Found {len(videos)} videos")
            return videos
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return []

# ============================================================================
# FILTER VIDEO (WHISPER CHECK)
# ============================================================================
def is_good_video(video_path, min_keywords=15):
    """Check if video has cooking narration"""
    
    audio_path = video_path.replace('.mp4', '_temp.mp3')
    
    # Extract audio
    cmd = f'ffmpeg -y -i {video_path} -ar 16000 -ac 1 -t 120 {audio_path} 2>/dev/null'
    subprocess.run(cmd, shell=True)
    
    if not os.path.exists(audio_path):
        return False
    
    # Transcribe first 2 minutes
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, task="translate")
        text = result['text'].lower()
    except:
        os.remove(audio_path)
        return False
    
    # Count cooking keywords
    keywords = [
        'chop', 'dice', 'cut', 'slice', 'mince',
        'stir', 'mix', 'add', 'pour', 'heat',
        'cook', 'fry', 'boil', 'simmer', 'roast',
        'onion', 'garlic', 'tomato', 'masala', 'spice',
        'pan', 'pot', 'oil', 'water', 'salt', 'pepper',
        'curry', 'rice', 'dal', 'chicken', 'paneer'
    ]
    
    count = sum(1 for kw in keywords if kw in text)
    
    os.remove(audio_path)
    
    return count >= min_keywords, count

# ============================================================================
# DOWNLOAD & FILTER
# ============================================================================
def download_and_filter(videos, target=80):
    print(f"\n{'='*70}")
    print(f"📥 DOWNLOADING & FILTERING (target: {target} good videos)")
    print(f"{'='*70}")
    
    good_videos = []
    stats = {'attempted': 0, 'download_failed': 0, 'rejected': 0, 'accepted': 0}
    
    for video in videos:
        if len(good_videos) >= target:
            break
        
        stats['attempted'] += 1
        video_id = video['id']
        title = video['title'][:50]
        
        print(f"\n[{stats['attempted']}] {title}")
        
        # Check if already downloaded
        video_path = f"temp_golden_{video_id}.mp4"
        
        if os.path.exists(video_path):
            print(f"   ⏭️  Already exists")
        else:
            print(f"   📥 Downloading...", end=" ")
            
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': video_path,
                'quiet': True,
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video['url']])
                
                size_mb = os.path.getsize(video_path) / (1024*1024)
                print(f"{size_mb:.1f}MB")
            except Exception as e:
                print(f"❌ Failed")
                stats['download_failed'] += 1
                continue
        
        # Filter with Whisper
        print(f"   🎤 Checking audio...", end=" ")
        is_good, keyword_count = is_good_video(video_path)
        
        if is_good:
            print(f"✅ GOOD ({keyword_count} keywords)")
            good_videos.append(video)
            stats['accepted'] += 1
            print(f"   📊 Progress: {len(good_videos)}/{target} good videos")
        else:
            print(f"❌ BAD ({keyword_count} keywords - music/ASMR)")
            os.remove(video_path)
            stats['rejected'] += 1
    
    return good_videos, stats

# ============================================================================
# MAIN
# ============================================================================
def main():
    # Channels to scrape (80 videos target = ~20 from each + margin)
    CHANNELS = [
        ("https://www.youtube.com/@YourFoodLab/videos", 40),
        ("https://www.youtube.com/@hebbarskitchen/videos", 40),
        ("https://www.youtube.com/@Kabitaskitchen/videos", 30),
        ("https://www.youtube.com/@CookWithParul/videos", 30),
    ]
    
    TARGET = 80  # Good videos (should yield ~800 labels)
    
    all_videos = []
    
    # Scrape all channels
    print("\n" + "="*70)
    print("PHASE 1: SCRAPING CHANNELS")
    print("="*70)
    
    for channel_url, max_vids in CHANNELS:
        videos = scrape_channel(channel_url, max_videos=max_vids)
        
        # Filter by duration (5-20 minutes)
        videos = [v for v in videos if 300 <= v.get('duration', 0) <= 1200]
        
        print(f"   After duration filter: {len(videos)} videos")
        all_videos.extend(videos)
    
    print(f"\n✅ Total videos to test: {len(all_videos)}")
    
    # Download & filter
    print("\n" + "="*70)
    print("PHASE 2: DOWNLOAD & FILTER")
    print("="*70)
    
    good_videos, stats = download_and_filter(all_videos, target=TARGET)
    
    # Save results
    with open('curated_good_videos.json', 'w') as f:
        json.dump(good_videos, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("🎉 FILTERING COMPLETE!")
    print("="*70)
    print(f"📊 Stats:")
    print(f"   Attempted: {stats['attempted']}")
    print(f"   Download failed: {stats['download_failed']}")
    print(f"   Rejected (music/ASMR): {stats['rejected']}")
    print(f"   ✅ Accepted: {stats['accepted']}")
    print(f"   Success rate: {stats['accepted']/stats['attempted']*100:.1f}%")
    print(f"\n💾 Saved: curated_good_videos.json")
    print(f"📁 Videos: temp_golden_*.mp4 ({len(good_videos)} files)")
    print(f"\n🎯 Estimated labels: ~{len(good_videos)*10}")
    print(f"\n🚀 Next: Run labeling & feature extraction")
    print(f"   sbatch label_and_extract.sbatch")

if __name__ == "__main__":
    main()
