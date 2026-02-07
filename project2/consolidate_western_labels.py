"""
Intelligently consolidate 96 labels into 30-40 Western cooking actions
"""
import json
from collections import Counter

# Load data
with open('data/processed/gpt4v_labels_enhanced.json') as f:
    data = json.load(f)

# Smart consolidation rules
CONSOLIDATION_MAP = {
    # Chopping variants
    'chopping tomato': 'chopping vegetables',
    'chopping shallot': 'chopping onion',
    'chopping celery': 'chopping vegetables',
    'chopping garlic': 'chopping vegetables',
    'chopping cabbage': 'chopping vegetables',
    'chopping zucchini': 'chopping vegetables',
    'chopping salmon': 'chopping meat',
    'cutting onion': 'chopping onion',
    
    # Slicing variants
    'slicing orange': 'slicing vegetables',
    'slicing zucchini': 'slicing vegetables',
    'slicing bell pepper': 'slicing vegetables',
    'slicing tomatoes': 'slicing tomato',
    
    # Stirring variants
    'stirring broccoli': 'stirring vegetables',
    'stirring soup': 'stirring mixture',
    'stirring cream': 'stirring mixture',
    'stirring beans': 'stirring mixture',
    'stirring ingredients': 'stirring mixture',
    'stirring vegetables and meat': 'stirring mixture',
    'stirring vegetables': 'stirring vegetables',
    
    # Mixing variants
    'mixing pasta salad': 'mixing salad',
    'mixing pasta': 'mixing ingredients',
    'mixing meat': 'mixing ingredients',
    'mixing vegetables': 'mixing ingredients',
    'mixing strawberries': 'mixing ingredients',
    
    # Cooking variants
    'cooking beef': 'cooking meat',
    'cooking meat': 'cooking meat',
    'cooking egg': 'frying egg',
    'cooking chicken': 'cooking meat',
    'cooking pasta': 'cooking pasta',
    'cooking food': 'cooking',
    
    # Frying variants
    'frying eggs': 'frying egg',
    'frying bacon': 'frying meat',
    'frying wrap': 'frying',
    'frying chicken': 'frying meat',
    'frying meat': 'frying meat',
    
    # Pouring variants
    'pouring cauliflower': 'pouring',
    'pouring tomato puree': 'pouring sauce',
    'pouring batter': 'pouring',
    'pouring oil': 'pouring',
    
    # Grating variants
    'grating lemon': 'grating',
    'grating garlic': 'grating',
    
    # NEW: Keep important actions that appeared
    'boiling potatoes': 'boiling',
    'boiling water': 'boiling',
    'mashing potatoes': 'mashing',
    'kneading dough': 'kneading',
    'beating eggs': 'whisking',
    'whisking batter': 'whisking',
    'blending smoothie': 'blending',
    'blending tomatoes': 'blending',
    'grilling meat': 'grilling',
    'roasting vegetables': 'roasting',
    
    # Remove noise
    'None': None,
    'holding pan': None,
    'arranging pans': None,
    'explaining cooking technique': None,
    'showing tools': None,
    'presenting tomatoes': None,
    'holding tool': None,
    'covering fire': None,
    'placing cake': None,
    'placing garlic': None,
}

# Apply consolidation
for item in data:
    label = item.get('action', item.get('label'))
    
    if label in CONSOLIDATION_MAP:
        new_label = CONSOLIDATION_MAP[label]
        if new_label is None:
            item['consolidated_label'] = 'REMOVE'
        else:
            item['consolidated_label'] = new_label
    else:
        item['consolidated_label'] = label

# Remove noise samples
data = [d for d in data if d.get('consolidated_label') != 'REMOVE']

# Count final distribution
final_counts = Counter(d['consolidated_label'] for d in data)

print(f"📊 CONSOLIDATED WESTERN COOKING DATASET")
print("="*70)
print(f"Original: 154 samples, 96 labels")
print(f"After consolidation: {len(data)} samples, {len(final_counts)} labels")
print()

print("📈 Final distribution (sorted by count):")
for label, count in sorted(final_counts.items(), key=lambda x: -x[1]):
    print(f"  {count:3d}x {label}")

# Show what we need MORE of
print("\n" + "="*70)
print("🎯 ACTIONS THAT NEED MORE SAMPLES (< 5):")
print("="*70)
for label, count in sorted(final_counts.items(), key=lambda x: x[1]):
    if count < 5:
        print(f"  {count:3d}x {label}")

# Save
with open('data/processed/western_cooking_consolidated.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n✅ Saved to: data/processed/western_cooking_consolidated.json")
