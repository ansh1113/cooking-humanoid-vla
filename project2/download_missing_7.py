"""
SUPPLEMENTAL DOWNLOADER: THE FINAL 7
Downloads 7 high-quality replacements for failed videos.
"""
import json
import os
import yt_dlp

# 7 Rock-Solid Replacements
MISSING_VIDEOS = [
    # 1. Paneer Butter Masala (Replacement for failed Ranveer video)
    {"url": "https://www.youtube.com/watch?v=l_a0d_VHtGg", "title": "Paneer Butter Masala", "chef": "Your Food Lab", "category": "paneer_dishes"},
    
    # 2. Tandoori Chicken (Replacement for failed video)
    {"url": "https://www.youtube.com/watch?v=fJd_gAl18_M", "title": "Tandoori Chicken", "chef": "Kunal Kapur", "category": "chicken_dishes"},
    
    # 3. Aloo Gobi (Replacement for failed video)
    {"url": "https://www.youtube.com/watch?v=r3XzDq4RjG0", "title": "Aloo Gobi Masala", "chef": "Your Food Lab", "category": "vegetable_curries"},
    
    # 4. Shahi Paneer (Replacement for failed video)
    {"url": "https://www.youtube.com/watch?v=wX22s5FqgSg", "title": "Shahi Paneer", "chef": "Bharatzkitchen", "category": "paneer_dishes"},
    
    # 5. Pav Bhaji (New - lots of mashing/cooking action)
    {"url": "https://www.youtube.com/watch?v=YI6sU-J6c9o", "title": "Mumbai Pav Bhaji", "chef": "Your Food Lab", "category": "street_food"},
    
    # 6. Chole Masala (New - distinct boiling/pressure cooking)
    {"url": "https://www.youtube.com/watch?v=1dIdQ5qjB64", "title": "Amritsari Chole", "chef": "Ranveer Brar", "category": "legumes"},
    
    # 7. Kadai Paneer (New - distinct sautéing)
    {"url": "https://www.youtube.com/watch?v=J9K5t-XvjTU", "title": "Kadai Paneer", "chef": "Your Food Lab", "category": "paneer_dishes"}
]

MAIN_LIST_FILE = 'golden_40_indian_videos.json'

def download_video(item):
    url = item['url']
    vid_id = url.split('=')[-1]
    filename = f"temp_golden_{vid_id}.mp4"
    
    if os.path.exists(filename) and os.path.getsize(filename) > 10000:
        print(f"✅ Already exists: {item['title']}")
        return True
        
    print(f"📥 Downloading: {item['title']}...")
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': filename,
        'quiet': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}}
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"✨ Success: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed {item['title']}: {e}")
        return False

def main():
    print("🚀 DOWNLOADING 7 MISSING VIDEOS")
    print("="*60)
    
    # 1. Download
    successful_videos = []
    for video in MISSING_VIDEOS:
        if download_video(video):
            successful_videos.append(video)
            
    print(f"\n📊 Successfully added {len(successful_videos)} videos.")
    
    # 2. Update the JSON List
    if os.path.exists(MAIN_LIST_FILE):
        with open(MAIN_LIST_FILE, 'r') as f:
            data = json.load(f)
            
        # Add to 'supplemental' category or respective categories
        if 'supplemental' not in data:
            data['supplemental'] = []
            
        # Check for duplicates
        existing_urls = set()
        for cat in data.values():
            for v in cat:
                existing_urls.add(v['url'])
                
        count = 0
        for vid in successful_videos:
            if vid['url'] not in existing_urls:
                data['supplemental'].append(vid)
                count += 1
                
        with open(MAIN_LIST_FILE, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"✅ Added {count} new videos to {MAIN_LIST_FILE}")
    else:
        print(f"⚠️ {MAIN_LIST_FILE} not found. Please verify.")

if __name__ == "__main__":
    main()
