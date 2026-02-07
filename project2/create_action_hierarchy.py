"""
Group 791 fine-grained actions into ~50 action categories
"""
import json
from collections import Counter

# Load data
with open('data/processed/training_labels_with_clips.json') as f:
    data = json.load(f)

# Define action mapping
ACTION_GROUPS = {
    # Cutting actions
    'cutting': ['cutting', 'chopping', 'slicing', 'dicing', 'mincing', 'snipping'],
    
    # Mixing actions
    'stirring': ['stirring', 'mixing', 'whisking', 'beating', 'folding', 'combining'],
    
    # Adding actions
    'adding': ['adding', 'pouring', 'putting', 'sprinkling', 'transferring', 'placing'],
    
    # Cooking actions
    'cooking': ['cooking', 'frying', 'boiling', 'simmering', 'roasting', 'baking', 'grilling', 'searing', 'sautéing'],
    
    # Grinding actions
    'grinding': ['grinding', 'blending', 'crushing', 'mashing'],
    
    # Kneading
    'kneading': ['kneading'],
    
    # Heat prep
    'heating': ['heating', 'warming', 'preheating'],
    
    # Garnishing
    'garnishing': ['garnishing', 'topping', 'dipping'],
}

def map_action_to_group(label):
    """Map fine-grained label to action group"""
    label_lower = label.lower()
    
    # Extract verb
    for group, keywords in ACTION_GROUPS.items():
        for keyword in keywords:
            if keyword in label_lower:
                return group
    
    return 'other'

# Create grouped labels
print("📊 Analyzing action distribution...")
print(f"Original labels: {len(data)}")

for item in data:
    original_label = item['label']
    item['fine_label'] = original_label
    item['coarse_label'] = map_action_to_group(original_label)

# Show distribution
coarse_counts = Counter(d['coarse_label'] for d in data)

print(f"\n📈 Coarse action distribution:")
for action, count in coarse_counts.most_common():
    print(f"   {action:20s}: {count:4d} samples")

print(f"\n✅ Unique coarse actions: {len(coarse_counts)}")

# Save
with open('data/processed/training_labels_hierarchical.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n💾 Saved to: data/processed/training_labels_hierarchical.json")
