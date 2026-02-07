"""
Add feature_path based on what's in the features_phase3 directory
"""
import json
import os

# Load data
with open('data/processed/training_labels_with_clips.json') as f:
    data = json.load(f)

print(f"📊 Processing {len(data)} samples...")

FEATURE_DIR = 'data/features_phase3'
added = 0
missing = 0

for item in data:
    video_id = item['video_url'].split('=')[-1]
    start_time = str(item['start']).replace('.', '_')
    save_name = f"{video_id}_{start_time}.pt"
    feature_path = os.path.join(FEATURE_DIR, save_name)
    
    if os.path.exists(feature_path):
        item['feature_path'] = feature_path
        added += 1
    else:
        missing += 1

print(f"✅ Added feature_path: {added}")
print(f"❌ Missing features: {missing}")

# Save
with open('data/processed/training_labels_with_features.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"💾 Saved to: data/processed/training_labels_with_features.json")
