"""
Rebuild grouped labels WITH all original fields including feature_path
"""
import json

# Load original with features
with open('data/processed/training_labels_with_clips.json') as f:
    original = json.load(f)

print(f"📊 Loaded {len(original)} original samples")

# Define grouping functions
VEGETABLES = ['onion', 'tomato', 'garlic', 'ginger', 'carrot', 'potato', 'spinach', 
              'pepper', 'capsicum', 'cabbage', 'beans', 'peas', 'celery']
MEATS = ['chicken', 'pork', 'beef', 'meat', 'ribs', 'sausage']
SPICES = ['salt', 'pepper', 'chili', 'masala', 'cumin', 'turmeric', 'spice', 'powder']
LIQUIDS = ['water', 'oil', 'ghee', 'butter', 'milk', 'cream', 'sauce', 'broth']
GRAINS = ['rice', 'noodles', 'pasta', 'flour', 'dough', 'bread']
DAIRY = ['cheese', 'paneer', 'cream', 'yogurt', 'curd', 'milk']

CUTTING_VERBS = ['cutting', 'chopping', 'slicing', 'dicing', 'mincing', 'snipping']
MIXING_VERBS = ['stirring', 'mixing', 'whisking', 'beating', 'folding', 'combining']
ADDING_VERBS = ['adding', 'pouring', 'putting', 'sprinkling', 'transferring', 'placing', 'garnishing']
COOKING_VERBS = ['cooking', 'frying', 'boiling', 'simmering', 'roasting', 'baking', 'searing', 'sautéing']
GRINDING_VERBS = ['grinding', 'blending', 'crushing', 'mashing']

def extract_verb(label):
    label_lower = label.lower()
    if any(v in label_lower for v in CUTTING_VERBS): return 'cutting'
    elif any(v in label_lower for v in MIXING_VERBS): return 'mixing'
    elif any(v in label_lower for v in ADDING_VERBS): return 'adding'
    elif any(v in label_lower for v in COOKING_VERBS): return 'cooking'
    elif any(v in label_lower for v in GRINDING_VERBS): return 'grinding'
    elif 'kneading' in label_lower: return 'kneading'
    elif 'heat' in label_lower: return 'heating'
    else: return 'other'

def extract_object_category(label):
    label_lower = label.lower()
    if any(v in label_lower for v in VEGETABLES): return 'vegetable'
    elif any(m in label_lower for m in MEATS): return 'meat'
    elif any(s in label_lower for s in SPICES): return 'spice'
    elif any(l in label_lower for l in LIQUIDS): return 'liquid'
    elif any(g in label_lower for g in GRAINS): return 'grain'
    elif any(d in label_lower for d in DAIRY): return 'dairy'
    else: return 'other'

# Add grouped labels to ALL original fields
for item in original:
    label = item['label']
    verb = extract_verb(label)
    obj = extract_object_category(label)
    
    # ADD grouped fields (keep everything else!)
    item['fine_label'] = label
    item['grouped_label'] = f"{verb}_{obj}"
    item['verb'] = verb
    item['object_category'] = obj

# Save
with open('data/processed/training_labels_final.json', 'w') as f:
    json.dump(original, f, indent=2)

print(f"✅ Saved {len(original)} samples with ALL fields including feature_path")
print(f"💾 File: data/processed/training_labels_final.json")

# Verify
from collections import Counter
grouped_counts = Counter(d['grouped_label'] for d in original)
print(f"\n📊 {len(grouped_counts)} unique grouped actions")
print(f"📊 Top 10:")
for action, count in grouped_counts.most_common(10):
    print(f"   {action:30s}: {count:4d}")
