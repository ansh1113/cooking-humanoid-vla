import json

# Inputs
INDIAN_FILE = 'data/processed/all_diverse_cooking_labels.json' 
WESTERN_FILE = 'data/processed/gpt4v_labels_enhanced.json'     

OUTPUT_FILE = 'data/processed/verb_training_data.json'

def get_verb(label):
    # FIX: Safety check for empty/null labels
    if not label: 
        return None
        
    l = label.lower()
    # 1. CUTTING (Vertical Motion)
    if any(x in l for x in ['chop', 'cut', 'slice', 'dice', 'mince', 'snip']):
        return 'CUT'
    # 2. MIXING (Circular Motion)
    if any(x in l for x in ['stir', 'mix', 'whisk', 'beat', 'saute', 'fry']):
        return 'MIX' 
    # 3. ADDING (Transfer Motion)
    if any(x in l for x in ['add', 'pour', 'sprinkle', 'place', 'put', 'transfer', 'garnish']):
        return 'ADD'
    # 4. KNEADING (Pressing Motion)
    if any(x in l for x in ['knead', 'roll', 'dough']):
        return 'KNEAD'
    return None

data = []

# 1. Process Indian Data (The Target)
try:
    with open(INDIAN_FILE) as f:
        raw = json.load(f)
        for item in raw:
            # We want strictly Indian data here, or everything if you want max data
            if item.get('category') in ['indian_vegetarian', 'indian_nonveg']:
                verb = get_verb(item.get('label'))
                if verb:
                    item['verb_label'] = verb
                    data.append(item)
except FileNotFoundError:
    print(f"⚠️ Warning: {INDIAN_FILE} not found. Skipping.")

# 2. Process Western Data (The Supplement)
try:
    with open(WESTERN_FILE) as f:
        raw = json.load(f)
        for item in raw:
            verb = get_verb(item.get('label'))
            if verb:
                item['verb_label'] = verb
                item['category'] = 'western_supplement' 
                data.append(item)
except FileNotFoundError:
    print(f"⚠️ Warning: {WESTERN_FILE} not found. Skipping.")

# 3. Save
with open(OUTPUT_FILE, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✅ Created Verb Dataset with {len(data)} samples.")
from collections import Counter
print("Distribution:", Counter([d['verb_label'] for d in data]))
