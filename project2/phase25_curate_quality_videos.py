"""
PHASE 2.5: Auto-curate 300 high-quality diverse cooking videos
From popular channels with quality filters
"""
from googleapiclient.discovery import build
import json
from datetime import datetime
import time

# Get free API key: https://console.cloud.google.com/apis/credentials
YOUTUBE_API_KEY = input("Enter your YouTube Data API key (free): ").strip()

# Curated high-quality cooking channels
QUALITY_CHANNELS = {
    'indian_vegetarian': [
        'Hebbar\'s Kitchen',
        'Rajshri Food',
        'Get Curried',
        'Kabita\'s Kitchen',
    ],
    'indian_nonveg': [
        'Kabita\'s Kitchen',
        'Ranveer Brar',
        'Kunal Kapur',
    ],
    'chinese': [
        'Chinese Cooking Demystified',
        'Woks of Life',
        'Made With Lau',
        'Souped Up Recipes',
    ],
    'mexican': [
        'Jauja Cocina Mexicana',
        'De Mi Rancho a Tu Cocina',
        'Rick Bayless',
    ],
    'italian': [
        'Pasta Grannies',
        'Italia Squisita',
        'Vincenzo\'s Plate',
    ],
    'thai': [
        'Pailin\'s Kitchen',
        'Hot Thai Kitchen',
    ],
    'japanese': [
        'Just One Cookbook',
        'Cooking with Dog',
        'Adam Liaw',
    ],
    'french': [
        'French Cooking Academy',
        'Chef Jean-Pierre',
    ],
    'middle_eastern': [
        'Middle Eats',
        'Feel Good Foodie',
    ],
    'korean': [
        'Maangchi',
        'Korean Bapsang',
    ],
    'baking': [
        'Preppy Kitchen',
        'Baking with Josh',
        'Sally\'s Baking Addiction',
    ],
    'american_general': [
        'Binging with Babish',
        'Bon Appétit',
        'Serious Eats',
        'Tasty',
        'America\'s Test Kitchen',
    ]
}

# Recipe keywords for each category
RECIPE_KEYWORDS = {
    'indian_vegetarian': [
        'paneer', 'dal', 'palak', 'aloo', 'chana', 'rajma', 
        'biryani vegetable', 'curry vegetarian', 'samosa', 'pakora'
    ],
    'indian_nonveg': [
        'butter chicken', 'chicken biryani', 'tandoori', 'rogan josh',
        'chicken curry', 'lamb curry', 'fish curry', 'chicken tikka'
    ],
    'chinese': [
        'fried rice', 'chow mein', 'kung pao', 'mapo tofu', 'dumplings',
        'spring rolls', 'wonton', 'hot pot', 'stir fry', 'sweet sour'
    ],
    'mexican': [
        'tacos', 'burritos', 'enchiladas', 'guacamole', 'salsa',
        'quesadilla', 'tamales', 'mole', 'pozole', 'carnitas'
    ],
    'italian': [
        'pasta carbonara', 'bolognese', 'pesto', 'lasagna', 'risotto',
        'pizza dough', 'tiramisu', 'gnocchi', 'ravioli', 'marinara'
    ],
    'thai': [
        'pad thai', 'green curry', 'tom yum', 'papaya salad', 'massaman',
        'thai basil', 'spring rolls', 'satay', 'khao soi'
    ],
    'japanese': [
        'sushi', 'ramen', 'tempura', 'teriyaki', 'miso soup',
        'katsu', 'onigiri', 'yakitori', 'takoyaki', 'okonomiyaki'
    ],
    'french': [
        'croissant', 'baguette', 'coq au vin', 'ratatouille', 'quiche',
        'soufflé', 'crème brûlée', 'bouillabaisse', 'beef bourguignon'
    ],
    'middle_eastern': [
        'hummus', 'falafel', 'shawarma', 'tabbouleh', 'baba ganoush',
        'kebab', 'fattoush', 'mansaf', 'kunafa'
    ],
    'korean': [
        'kimchi', 'bibimbap', 'bulgogi', 'tteokbokki', 'samgyeopsal',
        'japchae', 'korean fried chicken', 'kimchi jjigae', 'gimbap'
    ],
    'baking': [
        'sourdough', 'chocolate cake', 'cookies', 'brownies', 'macarons',
        'bagels', 'cinnamon rolls', 'croissants', 'cheesecake', 'muffins'
    ],
    'american_general': [
        'burger', 'steak', 'bbq', 'mac and cheese', 'fried chicken',
        'apple pie', 'pancakes', 'waffles', 'ribs', 'coleslaw'
    ]
}

def get_channel_id(youtube, channel_name):
    """Get channel ID from name"""
    try:
        request = youtube.search().list(
            q=channel_name,
            part='snippet',
            type='channel',
            maxResults=1
        )
        response = request.execute()
        
        if response['items']:
            return response['items'][0]['snippet']['channelId']
    except Exception as e:
        print(f"   ⚠️  Error finding channel {channel_name}: {e}")
    return None

def search_channel_videos(youtube, channel_id, keywords, max_results=10):
    """Search videos in a specific channel with keywords"""
    videos = []
    
    for keyword in keywords[:max_results]:  # Limit keywords
        try:
            request = youtube.search().list(
                channelId=channel_id,
                q=keyword,
                part='id,snippet',
                maxResults=2,
                type='video',
                order='viewCount',  # Most viewed first
                videoDuration='medium'  # 4-20 minutes
            )
            response = request.execute()
            
            for item in response['items']:
                video_id = item['id']['videoId']
                videos.append({
                    'video_id': video_id,
                    'url': f'https://www.youtube.com/watch?v={video_id}',
                    'title': item['snippet']['title'],
                    'channel': item['snippet']['channelTitle'],
                    'keyword': keyword
                })
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"   ⚠️  Error searching {keyword}: {e}")
            continue
    
    return videos

def get_video_stats(youtube, video_ids):
    """Get view count, likes for filtering"""
    try:
        request = youtube.videos().list(
            part='statistics,contentDetails',
            id=','.join(video_ids[:50])  # Max 50 per request
        )
        response = request.execute()
        
        stats = {}
        for item in response['items']:
            video_id = item['id']
            stats[video_id] = {
                'views': int(item['statistics'].get('viewCount', 0)),
                'likes': int(item['statistics'].get('likeCount', 0)),
                'duration': item['contentDetails']['duration']
            }
        
        return stats
    except Exception as e:
        print(f"   ⚠️  Error getting stats: {e}")
        return {}

def filter_quality_videos(videos, youtube, min_views=10000, min_duration=180):
    """Filter videos by quality metrics"""
    if not videos:
        return []
    
    # Get stats in batches
    video_ids = [v['video_id'] for v in videos]
    stats = get_video_stats(youtube, video_ids)
    
    filtered = []
    for video in videos:
        vid_stats = stats.get(video['video_id'])
        if vid_stats:
            if vid_stats['views'] >= min_views:
                video['views'] = vid_stats['views']
                video['likes'] = vid_stats['likes']
                filtered.append(video)
    
    # Sort by views
    filtered.sort(key=lambda x: x['views'], reverse=True)
    
    return filtered

def curate_category(youtube, category, channels, keywords, target_count):
    """Curate videos for one category"""
    print(f"\n📂 {category}")
    print(f"   Target: {target_count} videos")
    
    all_videos = []
    
    for channel_name in channels:
        print(f"   🔍 Searching {channel_name}...")
        
        channel_id = get_channel_id(youtube, channel_name)
        if not channel_id:
            continue
        
        videos = search_channel_videos(youtube, channel_id, keywords, max_results=5)
        all_videos.extend(videos)
    
    # Filter by quality
    print(f"   ✅ Found {len(all_videos)} videos, filtering...")
    filtered = filter_quality_videos(all_videos, youtube)
    
    # Take top N
    selected = filtered[:target_count]
    
    print(f"   ✅ Selected {len(selected)} high-quality videos")
    
    return selected

def main():
    print("="*70)
    print("🎬 AUTO-CURATING 300 DIVERSE COOKING VIDEOS")
    print("="*70)
    
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    
    # Target distribution
    targets = {
        'indian_vegetarian': 20,
        'indian_nonveg': 20,
        'chinese': 30,
        'mexican': 25,
        'italian': 25,
        'thai': 20,
        'japanese': 25,
        'french': 20,
        'middle_eastern': 20,
        'korean': 20,
        'baking': 30,
        'american_general': 45,
    }
    
    all_curated = {}
    total_found = 0
    
    for category in targets:
        channels = QUALITY_CHANNELS.get(category, [])
        keywords = RECIPE_KEYWORDS.get(category, [])
        
        if not channels or not keywords:
            print(f"⚠️  Skipping {category} (no channels/keywords)")
            continue
        
        videos = curate_category(
            youtube, 
            category, 
            channels, 
            keywords, 
            targets[category]
        )
        
        all_curated[category] = videos
        total_found += len(videos)
    
    # Save results
    with open('data/curated_diverse_videos.json', 'w') as f:
        json.dump(all_curated, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("✅ CURATION COMPLETE")
    print("="*70)
    print(f"🎯 Found: {total_found} high-quality videos")
    print(f"💾 Saved to: data/curated_diverse_videos.json")
    
    print(f"\n📊 Breakdown:")
    for category, videos in all_curated.items():
        print(f"   {category:25s}: {len(videos):3d} videos")
    
    print(f"\n💰 Labeling cost: ${total_found * 0.02:.2f}")
    print("="*70)

if __name__ == "__main__":
    main()
