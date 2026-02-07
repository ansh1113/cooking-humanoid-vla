"""
Smart action grouping: verb + object category
"""
import json
from collections import Counter
import re

# Object categories
VEGETABLES = ['onion', 'tomato', 'garlic', 'ginger', 'carrot', 'potato', 'spinach', 
              'pepper', 'capsicum', 'cabbage', 'beans', 'peas', 'celery']
MEATS = ['chicken', 'pork', 'beef', 'meat', 'ribs', 'sausage']
SPICES = ['salt', 'pepper', 'chili', 'masala', 'cumin', 'turmeric', 'spice', 'powder']
LIQUIDS = ['water', 'oil', 'ghee', 'butter', 'milk', 'cream', 'sauce', 'broth']
GRAINS = ['rice', 'noodles', 'pasta', 'flour', 'dough', 'bread']
DAIRY = ['cheese', 'paneer', 'cream', 'yogurt', 'curd', 'milk']

# Verb groups
CUTTING_VERBS = ['cutting', 'chopping', 'slicing', 'dicing', 'mincing', 'snipping']
MIXING_VERBS = ['stirring', 'mixing', 'whisking', 'beating', 'folding', 'combining']
ADDING_VERBS = ['adding', 'pouring', 'putting', 'sprinkling', 'transferring', 'placing', 'garnishing']
COOKING_VERBS = ['cooking', 'frying', 'boiling', 'simmering', 'roasting', 'baking', 'searing', 'sautéing']
GRINDING_VERBS = ['grinding', 'blending', 'crushing', 'mashing']

def extract_verb(label):
    """Extract verb from label"""
    label_lower = label.lower()
    
    if any(v in label_lower for v in CUTTING_VERBS):
        return 'cutting'
    elif any(v in label_lower for v in MIXING_VERBS):
        return 'mixing'
    elif any(v in label_lower for v in ADDING_VERBS):
        return 'adding'
    elif any(v in label_lower for v in COOKING_VERBS):
        return 'cooking'
    elif any(v in label_lower for v in GRINDING_VERBS):
        return 'grinding'
    elif 'kneading' in label_lower or 'knead' in label_lower:
        return 'kneading'
    elif 'heat' in label_lower or 'warm' in label_lower:
        return 'heating'
    else:
        return 'other'

def extract_object_category(label):
    """Extract object category from label"""
    label_lower = label.lower()
    
    if any(v in label_lower for v in VEGETABLES):
        return 'vegetable'
    elif any(m in label_lower for m in MEATS):
        return 'meat'
    elif any(s in label_lower for s in SPICES):
        return 'spice'
    elif any(l in label_lower for l in LIQUIDS):
        return 'liquid'
    elif any(g in label_lower for g in GRAINS):
        return 'grain'
    elif any(d in label_lower for d in DAIRY):
        return 'dairy'
    else:
        return 'other'

# Load data
with open('data/processed/training_labels_with_clips.json') as f:
    data = json.load(f)

print(f"📊 Processing {len(data)} samples...")

# Create grouped labels
for item in data:
    label = item['label']
    verb = extract_verb(label)
    obj = extract_object_category(label)
    
    # Create grouped label
    grouped = f"{verb}_{obj}"
    
    item['fine_label'] = label
    item['grouped_label'] = grouped
    item['verb'] = verb
    item['object_category'] = obj

# Show distribution
grouped_counts = Counter(d['grouped_label'] for d in data)

print(f"\n📈 Top 30 grouped actions:")
for i, (action, count) in enumerate(grouped_counts.most_common(30), 1):
    print(f"   {i:2d}. {action:30s}: {count:4d}")

print(f"\n✅ Total unique grouped actions: {len(grouped_counts)}")
print(f"✅ Actions with 5+ samples: {sum(1 for c in grouped_counts.values() if c >= 5)}")
print(f"✅ Actions with 10+ samples: {sum(1 for c in grouped_counts.values() if c >= 10)}")

# Save
with open('data/processed/training_labels_grouped.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n💾 Saved to: data/processed/training_labels_grouped.json")
