"""
Add final actions to reach 85+ and maximize coverage
"""
import pandas as pd
import json
from pathlib import Path

# Load current mapping
df = pd.read_csv('data/epic_kitchens/epic_expanded_100.csv')

with open('data/epic_kitchens/action_vocab_100.json', 'r') as f:
    robot_actions = json.load(f)

# Add unmapped high-frequency verbs
additional_actions = {
    'PlaceOnObject': ['put-on', 'put-onto', 'place-on'],
    'TakeFromObject': ['take-from', 'take-off', 'remove-from'],
    'SpongeObject': ['sponge'],
    'ThrowIntoObject': ['throw-into'],
    'StirInObject': ['stir-in'],
    'PlaceInObject': ['place-in'],
    'PickObject': ['pick'],
    'PutAwayObject': ['put-away'],
    'EatObject': ['eat'],
    'TapObject': ['tap'],
    'StretchObject': ['stretch'],
    'HangObject': ['hang'],
    'WipeOffObject': ['wipe-off'],
    'ScrapeObject': ['scrape'],
    'SharpenObject': ['sharpen'],
    'CutIntoObject': ['cut-into'],
}

# Merge
robot_actions.update(additional_actions)

# Create mapping
epic_to_robot = {}
for robot_action, epic_verbs in robot_actions.items():
    for epic_verb in epic_verbs:
        epic_to_robot[epic_verb] = robot_action

# Remap
df['robot_action'] = df['verb'].map(epic_to_robot)

# Stats
mapped = df['robot_action'].notna().sum()
coverage = mapped / len(df) * 100

print("="*70)
print(f"✅ FINAL VOCABULARY: {len(robot_actions)} ACTIONS")
print(f"✅ COVERAGE: {coverage:.1f}% ({mapped:,}/{len(df):,} segments)")
print("="*70)

# Save
df.to_csv('data/epic_kitchens/epic_final_85.csv', index=False)

with open('data/epic_kitchens/action_vocab_85.json', 'w') as f:
    json.dump(robot_actions, f, indent=2)

print("\n💾 Saved:")
print("   - data/epic_kitchens/epic_final_85.csv")
print("   - data/epic_kitchens/action_vocab_85.json")
