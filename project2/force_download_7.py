"""
FORCE DOWNLOADER: CLIENT ROTATION 🔄
Tries multiple identities (iOS, Android, TV) to bypass IP blocks.
"""
import json
import os
import yt_dlp
import time
import random

# The 7 Missing Targets
TARGETS = [
    {"url": "https://www.youtube.com/watch?v=l_a0d_VHtGg", "title": "Paneer Butter Masala", "chef": "Your Food Lab", "category": "paneer_dishes"},
    {"url": "https://www.youtube.com/watch?v=fJd_gAl18_M", "title": "Tandoori Chicken", "chef": "Kunal Kapur", "category": "chicken_dishes"},
    {"url": "https://www.youtube.com/watch?v=r3XzDq4RjG0", "title": "Aloo Gobi Masala", "chef": "Your Food Lab", "category": "vegetable_curries"},
    {"url": "https://www.youtube.com/watch?v=wX22s5FqgSg", "title": "Shahi Paneer", "chef": "Bharatzkitchen", "category": "paneer_dishes"},
    {"url": "https://www.youtube.com/watch?v=YI6sU-J6c9o", "title": "Mumbai Pav Bhaji", "chef": "Your Food Lab", "category": "street_food"},
    {"url": "https://www.youtube.com/watch?v=1dIdQ5qjB64", "title": "Amritsari Chole", "chef": "Ranveer Brar", "category": "legumes"},
    {"url": "https://www.youtube.com/watch?v=J9K5t-XvjTU", "title": "Kadai Paneer", "chef": "Your Food Lab", "category": "paneer_dishes"}
]

# Client Disguises to rotate
CLIENTS = ['ios', 'android_creator', 'android', 'web']

MAIN_LIST_FILE = 'golden_40_indian_videos.json'

def download_with_disguise(item):
    url = item['url']
    vid_id = url.split('=')[-1]
    filename = f"temp_golden_{vid_id}.mp4"
    
    if os.path.exists(filename) and os.path.getsize(filename) > 10000:
        print(f"✅ Already have: {item['title']}")
        return True

    print(f"\n🎯 Target: {item['title']}")
    
    for client in CLIENTS:
        print(f"   🎭 Trying disguise: {client}...")
        
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': filename,
            'quiet': True,
            'extractor_args': {'youtube': {'player_client': [client]}}
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            if os.path.exists(filename) and os.path.getsize(filename) > 10000:
                print(f"   ✨ SUCCESS with {client}!")
                return True
        except Exception as e:
            # Check for specific ban message
            if "Sign in" in str(e):
                print(f"      ❌ {client} detected as bot.")
            else:
                print(f"      ❌ {client} failed: {str(e)[:50]}...")
            
            time.sleep(3) # Wait before changing disguise
            
    return False

def main():
    print("🚀 FORCE DOWNLOADER INITIATED")
    print("="*60)
    
    successful = []
    
    for item in TARGETS:
        if download_with_disguise(item):
            successful.append(item)
        print("❄️  Cooling down for 10 seconds...")
        time.sleep(10) # Hard wait to reset rate limits
        
    print(f"\n📊 Result: {len(successful)}/7 downloaded.")
    
    # Update JSON if we got any new ones
    if successful:
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
        print(f"✅ Updated database with {count} new videos.")

if __name__ == "__main__":
    main()
