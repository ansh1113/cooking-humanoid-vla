"""
AUTO-FILL 7: HYBRID SEARCH & RESCUE 🚑
Tries direct URLs. If blocked, searches YouTube for the title and downloads the first match.
Does NOT require cookies, but uses them if present.
"""
import json
import os
import yt_dlp
import time
import random

# The 7 Missing Targets
TARGETS = [
    {"url": "https://www.youtube.com/watch?v=l_a0d_VHtGg", "search_query": "Paneer Butter Masala Your Food Lab recipe", "title": "Paneer Butter Masala", "chef": "Your Food Lab", "category": "paneer_dishes"},
    {"url": "https://www.youtube.com/watch?v=fJd_gAl18_M", "search_query": "Tandoori Chicken Kunal Kapur recipe", "title": "Tandoori Chicken", "chef": "Kunal Kapur", "category": "chicken_dishes"},
    {"url": "https://www.youtube.com/watch?v=r3XzDq4RjG0", "search_query": "Aloo Gobi Masala Your Food Lab recipe", "title": "Aloo Gobi Masala", "chef": "Your Food Lab", "category": "vegetable_curries"},
    {"url": "https://www.youtube.com/watch?v=wX22s5FqgSg", "search_query": "Shahi Paneer Bharatzkitchen recipe", "title": "Shahi Paneer", "chef": "Bharatzkitchen", "category": "paneer_dishes"},
    {"url": "https://www.youtube.com/watch?v=YI6sU-J6c9o", "search_query": "Mumbai Pav Bhaji Your Food Lab recipe", "title": "Mumbai Pav Bhaji", "chef": "Your Food Lab", "category": "street_food"},
    {"url": "https://www.youtube.com/watch?v=1dIdQ5qjB64", "search_query": "Amritsari Chole Ranveer Brar recipe", "title": "Amritsari Chole", "chef": "Ranveer Brar", "category": "legumes"},
    {"url": "https://www.youtube.com/watch?v=J9K5t-XvjTU", "search_query": "Kadai Paneer Your Food Lab recipe", "title": "Kadai Paneer", "chef": "Your Food Lab", "category": "paneer_dishes"}
]

MAIN_LIST_FILE = 'golden_40_indian_videos.json'
COOKIE_FILE = 'cookies.txt'

def smart_download(item):
    print(f"\n🎯 Processing: {item['title']}")
    
    # Strategy 1: Direct URL (Fastest)
    vid_id = item['url'].split('=')[-1]
    filename = f"temp_golden_{vid_id}.mp4"
    
    if os.path.exists(filename) and os.path.getsize(filename) > 10000:
        print(f"   ✅ Already exists on disk.")
        return item

    # Configure Options
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': filename,
        'quiet': True,
        'noplaylist': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}}
    }
    
    # Add cookies if user uploaded them
    if os.path.exists(COOKIE_FILE):
        print("   🍪 Cookies found! Using them.")
        ydl_opts['cookiefile'] = COOKIE_FILE

    # Attempt 1: Direct URL
    try:
        print(f"   Trying Direct URL...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([item['url']])
        print(f"   ✨ Success (Direct)!")
        return item
    except Exception as e:
        print(f"   ⚠️ Direct failed. Switching to Search Strategy...")

    # Strategy 2: Search Fallback
    # Searching generates a FRESH video ID that is often not blocked!
    try:
        query = item['search_query']
        print(f"   🔍 Searching for: '{query}'...")
        ydl_opts['default_search'] = 'ytsearch1'
        ydl_opts['outtmpl'] = 'temp_golden_%(id)s.mp4' # Will save with NEW ID
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=True)
            if 'entries' in info:
                video_info = info['entries'][0]
            else:
                video_info = info
                
            new_id = video_info['id']
            new_filename = f"temp_golden_{new_id}.mp4"
            print(f"   ✨ Success (Search)! Downloaded: {new_filename}")
            
            # Update item with new details
            item['url'] = f"https://www.youtube.com/watch?v={new_id}"
            return item
            
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
        return None

def main():
    print("🚀 AUTO-FILL INITIATED")
    print("="*60)
    
    successful = []
    
    for item in TARGETS:
        result = smart_download(item)
        if result:
            successful.append(result)
        time.sleep(2) 
        
    print(f"\n📊 Captured {len(successful)}/7 videos.")
    
    # Update JSON List
    if successful and os.path.exists(MAIN_LIST_FILE):
        with open(MAIN_LIST_FILE, 'r') as f:
            data = json.load(f)
            
        if 'supplemental' not in data:
            data['supplemental'] = []
            
        existing_urls = set()
        for cat in data.values():
            for v in cat:
                existing_urls.add(v['url'])
                
        count = 0
        for vid in successful:
            if vid['url'] not in existing_urls:
                data['supplemental'].append(vid)
                count += 1
                
        with open(MAIN_LIST_FILE, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"✅ Registered {count} new videos in dataset list.")

    # Final Count Check
    total_files = len([f for f in os.listdir('.') if f.startswith('temp_golden_') and f.endswith('.mp4')])
    print(f"📈 TOTAL VIDEOS ON DISK: {total_files}")
    if total_files >= 40:
        print("🎉 DATASET COMPLETE. RUN LABELING NOW.")
    else:
        print(f"⚠️ Still missing {40 - total_files} videos.")

if __name__ == "__main__":
    main()
