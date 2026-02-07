"""
Smart sampling: Get 100-200 clips per action (top 30 actions)
Total: ~3000-6000 clips instead of full dataset
"""
import pandas as pd
import json
from collections import defaultdict

# Load mapped data
df = pd.read_csv('data/epic_kitchens/epic_final_85.csv')
df_mapped = df[df['robot_action'].notna()]

# Focus on top 30 actions (cover 95%+ of use cases)
action_counts = df_mapped['robot_action'].value_counts()
top_actions = action_counts.head(30).index.tolist()

print("="*70)
print("🎯 SAMPLING STRATEGY")
print("="*70)
print(f"\nFocusing on top 30 actions (covers {action_counts.head(30).sum()/len(df_mapped)*100:.1f}% of data)")

# Sample clips
samples_per_action = 150  # 150 clips per action
sampled_clips = []

for action in top_actions:
    action_df = df_mapped[df_mapped['robot_action'] == action]
    
    # Sample evenly across different videos/participants
    sample = action_df.groupby('participant_id').apply(
        lambda x: x.sample(min(len(x), samples_per_action // 10))
    ).reset_index(drop=True)
    
    if len(sample) < samples_per_action:
        # If not enough, sample more
        additional = action_df.sample(min(len(action_df), samples_per_action - len(sample)))
        sample = pd.concat([sample, additional]).drop_duplicates()
    else:
        sample = sample.sample(samples_per_action)
    
    sampled_clips.append(sample)
    print(f"   {action:25s}: {len(sample):4d} clips")

# Combine
final_sample = pd.concat(sampled_clips)

print(f"\n📊 TOTAL SAMPLED: {len(final_sample):,} clips")
print(f"💾 Estimated size: ~{len(final_sample) * 10 / 1024:.1f} GB")

# Save sample list
final_sample.to_csv('data/epic_kitchens/sampled_clips.csv', index=False)

print(f"\n💾 Saved to: data/epic_kitchens/sampled_clips.csv")
print("="*70)
print("\n💡 NEXT: Download these clips using Epic-Kitchens downloader")
