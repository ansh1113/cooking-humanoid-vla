"""
Merge feature_path from cached file into grouped labels
"""
import json

# Load the file with feature paths
with open('data/processed/training_labels_with_clips.json') as f:
    original = json.load(f)

# Load the grouped labels
with open('data/processed/training_labels_grouped.json') as f:
    grouped = json.load(f)

print(f"Original: {len(original)} samples")
print(f"Grouped: {len(grouped)} samples")

# They should be the same length and order
if len(original) != len(grouped):
    print("❌ ERROR: Lengths don't match!")
    exit(1)

# Merge feature_path
for i in range(len(grouped)):
    if 'feature_path' in original[i]:
        grouped[i]['feature_path'] = original[i]['feature_path']
    else:
        print(f"⚠️  Missing feature_path at index {i}")

# Save
with open('data/processed/training_labels_grouped.json', 'w') as f:
    json.dump(grouped, f, indent=2)

print(f"✅ Merged feature paths!")
print(f"💾 Saved to: data/processed/training_labels_grouped.json")
