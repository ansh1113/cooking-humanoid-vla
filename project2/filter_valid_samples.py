"""
Keep only samples that have cached features
"""
import json
import os

with open('data/processed/training_labels_grouped.json') as f:
    data = json.load(f)

print(f"📊 Total samples: {len(data)}")

# Filter to only samples with valid feature_path
valid = []
missing = 0

for item in data:
    if 'feature_path' in item and os.path.exists(item['feature_path']):
        valid.append(item)
    else:
        missing += 1

print(f"✅ Valid samples: {len(valid)}")
print(f"❌ Missing features: {missing}")

# Save valid only
with open('data/processed/training_labels_grouped_valid.json', 'w') as f:
    json.dump(valid, f, indent=2)

print(f"💾 Saved to: data/processed/training_labels_grouped_valid.json")

# Check distribution still good
from collections import Counter
grouped_counts = Counter(d['grouped_label'] for d in valid)

print(f"\n📈 Top 15 grouped actions (after filtering):")
for i, (action, count) in enumerate(grouped_counts.most_common(15), 1):
    print(f"   {i:2d}. {action:30s}: {count:4d}")

print(f"\n✅ Unique actions: {len(grouped_counts)}")
