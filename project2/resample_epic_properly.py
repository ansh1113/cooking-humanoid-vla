"""
Proper resampling: Get 150 clips per action (top 30 actions)
"""
import pandas as pd
from collections import defaultdict

# Load mapped data
df = pd.read_csv('data/epic_kitchens/epic_final_85.csv')
df_mapped = df[df['robot_action'].notna()].copy()

print("="*70)
print("🎯 PROPER RESAMPLING")
print("="*70)

# Focus on top 30 actions
action_counts = df_mapped['robot_action'].value_counts()
top_actions = action_counts.head(30).index.tolist()

print(f"\nFocusing on top 30 actions")
print(f"Total available segments: {len(df_mapped):,}")

# Sample clips properly
samples_per_action = 150
sampled_clips = []

for action in top_actions:
    action_df = df_mapped[df_mapped['robot_action'] == action].copy()
    
    # Sample up to 150 clips for this action
    if len(action_df) >= samples_per_action:
        sample = action_df.sample(samples_per_action, random_state=42)
    else:
        sample = action_df
    
    sampled_clips.append(sample)
    print(f"   {action:25s}: {len(sample):4d} clips")

# Combine
final_sample = pd.concat(sampled_clips, ignore_index=True)

# Verify no NaN
print(f"\n📊 VERIFICATION:")
print(f"   Total sampled: {len(final_sample):,} clips")
print(f"   Rows with NaN participant_id: {final_sample['participant_id'].isna().sum()}")

# Get unique participants
participants = sorted(final_sample['participant_id'].unique())
participant_list = ','.join(participants)

print(f"\n✅ PARTICIPANTS ({len(participants)}):")
print(f"   {participant_list}")
print(f"\n💾 Estimated download size: ~{len(participants) * 2} GB")

# Save
final_sample.to_csv('data/epic_kitchens/sampled_clips_fixed.csv', index=False)

with open('data/epic_kitchens/participants_to_download.txt', 'w') as f:
    f.write(participant_list)

print(f"\n💾 Saved:")
print(f"   - data/epic_kitchens/sampled_clips_fixed.csv")
print(f"   - data/epic_kitchens/participants_to_download.txt")
print("="*70)
