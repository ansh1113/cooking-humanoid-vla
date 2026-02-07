"""
SMART FILLER: SEARCH & CAPTURE 🕵️‍♂️
Automatically finds and downloads working videos to fill the dataset.
"""
import json
import os
import yt_dlp

# Target Queries to fill the gaps
SEARCH_QUERIES = [
    {"query": "Paneer Butter Masala Ranveer Brar", "category": "paneer_dishes"},
    {"query": "Tandoori Chicken at home Kunal Kapur", "category": "chicken_dishes"},
    {"query": "Aloo Gobi Masala Your Food Lab", "category": "vegetable_curries"},
    {"query": "Shahi Paneer Bharatzkitchen", "category": "paneer_dishes"},
    {"query": "Pav Bhaji Recipe Your Food Lab", "category": "street_food"},
    {"query": "Amritsari Chole Ranveer Brar", "category": "legumes"},
    {"query": "Kadai Paneer Recipe Your Food Lab", "category": "paneer_dishes"}
]

MAIN_LIST_FILE = 'golden_40_indian_videos.json'

def search_and_download(query_obj):
    query = query_obj['query']
    print(f"\n🔍 Searching for: '{query}'...")
    
    # Search for top 3 results
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': 'temp_golden_%(id)s.mp4',
        'quiet': True,
        'noplaylist': True,
        # Search 3 candidates
        'default_search': 'ytsearch3',
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}}
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get metadata without downloading first
            search_results = ydl.extract_info(query, download=False)
            
            if 'entries' not in search_results:
                print("❌ No results found.")
                return None

            # Try downloading entries one by one
            for entry in search_results['entries']:
                if not entry: continue
                
                vid_id = entry['id']
                title = entry['title']
                filename = f"temp_golden_{vid_id}.mp4"
                url = f"https://www.youtube.com/watch?v={vid_id}"
                
                print(f"   👉 Trying: {title} ({url})")
                
                try:
                    # Download specific video
                    ydl.download([url])
                    
                    if os.path.exists(filename) and os.path.getsize(filename) > 10000:
                        print(f"   ✨ SUCCESS: {filename}")
                        return {
                            "url": url,
                            "title": title,
                            "chef": "Auto-Search",
                            "category": query_obj['category']
                        }
                except Exception as e:
                    print(f"   ⚠️ Failed: {e}")
                    continue
                    
    except Exception as e:
        print(f"❌ Search Error: {e}")
        
    return None

def main():
    print("🚀 SMART FILLER INITIATED")
    print("="*60)
    
    new_videos = []
    
    for q in SEARCH_QUERIES:
        result = search_and_download(q)
        if result:
            new_videos.append(result)
        else:
            print(f"❌ Could not fill gap for: {q['query']}")
            
    print(f"\n📊 Successfully captured {len(new_videos)} videos.")
    
    # Update JSON
    if os.path.exists(MAIN_LIST_FILE):
        with open(MAIN_LIST_FILE, 'r') as f:
            data = json.load(f)
            
        if 'supplemental' not in data:
            data['supplemental'] = []
            
        # Add non-duplicates
        existing_urls = set()
        for cat in data.values():
            for v in cat:
                existing_urls.add(v['url'])
                
        count = 0
        for vid in new_videos:
            if vid['url'] not in existing_urls:
                data['supplemental'].append(vid)
                count += 1
                
        with open(MAIN_LIST_FILE, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"✅ Added {count} new videos to {MAIN_LIST_FILE}")

if __name__ == "__main__":
    main()
