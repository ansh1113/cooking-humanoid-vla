"""
Remove duplicate videos from curated list
"""
import json

# Load
with open('data/curated_diverse_videos.json') as f:
    curated = json.load(f)

# Remove duplicates
seen_urls = set()
cleaned = {}

for category, videos in curated.items():
    unique_videos = []
    for video in videos:
        url = video['url']
        if url not in seen_urls:
            seen_urls.add(url)
            unique_videos.append(video)
    
    cleaned[category] = unique_videos
    print(f"{category}: {len(videos)} → {len(unique_videos)} (removed {len(videos) - len(unique_videos)})")

# Save
with open('data/curated_diverse_videos_clean.json', 'w') as f:
    json.dump(cleaned, f, indent=2)

total_before = sum(len(v) for v in curated.values())
total_after = sum(len(v) for v in cleaned.values())

print(f"\nTotal: {total_before} → {total_after}")
print(f"Saved to: data/curated_diverse_videos_clean.json")
