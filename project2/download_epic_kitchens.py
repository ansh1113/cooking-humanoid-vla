"""
Download Epic-Kitchens dataset annotations
Epic-Kitchens has 97 verbs like: stir, pour, mix, peel, open, close, wash, dry, etc.
"""
import requests
import pandas as pd
from pathlib import Path

def download_epic_annotations():
    print("="*70)
    print("📥 DOWNLOADING EPIC-KITCHENS ANNOTATIONS")
    print("="*70)
    
    # Epic-Kitchens-100 annotations (free, no videos needed initially)
    base_url = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_train.csv"
    
    print("\n📥 Downloading training annotations...")
    df = pd.read_csv(base_url)
    
    print(f"✅ Downloaded {len(df)} action segments")
    print(f"✅ Unique verbs: {df['verb'].nunique()}")
    print(f"✅ Unique nouns: {df['noun'].nunique()}")
    
    # Save
    output_dir = Path('data/epic_kitchens')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / 'epic_train.csv', index=False)
    
    # Analyze verb distribution
    print("\n📊 TOP 30 COOKING VERBS:")
    verb_counts = df['verb'].value_counts().head(30)
    for i, (verb, count) in enumerate(verb_counts.items(), 1):
        print(f"   {i:2d}. {verb:20s}: {count:5d} segments")
    
    # Map to our action vocabulary
    print("\n🔄 MAPPING TO ROBOT ACTIONS...")
    
    # Define comprehensive action vocabulary
    robot_actions = {
        # Current 11 actions
        'PickupObject': ['take', 'pick-up', 'lift', 'get'],
        'PutObject': ['put', 'place', 'set', 'rest'],
        'SliceObject': ['cut', 'slice', 'chop', 'dice'],
        'CookObject': ['cook', 'fry', 'boil', 'heat'],
        'OpenObject': ['open'],
        'CloseObject': ['close', 'shut'],
        'ToggleObjectOn': ['turn-on', 'switch-on'],
        'ToggleObjectOff': ['turn-off', 'switch-off'],
        'DropHandObject': ['drop', 'release'],
        'ThrowObject': ['throw', 'toss'],
        'BreakObject': ['break', 'crack'],
        
        # NEW COOKING ACTIONS (expand to 50+)
        'StirObject': ['stir', 'mix', 'whisk'],
        'PourLiquid': ['pour'],
        'WashObject': ['wash', 'rinse', 'clean'],
        'DryObject': ['dry', 'wipe'],
        'PeelObject': ['peel'],
        'GrateObject': ['grate'],
        'ScoopObject': ['scoop'],
        'SpreadObject': ['spread', 'smear'],
        'SeasonObject': ['season', 'sprinkle', 'add'],
        'SqueezeObject': ['squeeze'],
        'ShakeObject': ['shake'],
        'FlipObject': ['flip', 'turn'],
        'MashObject': ['mash', 'crush'],
        'KneadObject': ['knead'],
        'RollObject': ['roll'],
        'FoldObject': ['fold'],
        'StrainObject': ['strain', 'sieve'],
        'DrainObject': ['drain'],
        'BlendObject': ['blend'],
        'GrindObject': ['grind'],
        'SauteObject': ['sauté'],
    }
    
    # Create mapping
    epic_to_robot = {}
    for robot_action, epic_verbs in robot_actions.items():
        for epic_verb in epic_verbs:
            epic_to_robot[epic_verb] = robot_action
    
    # Map dataset
    df['robot_action'] = df['verb'].map(epic_to_robot)
    
    # Stats
    mapped = df['robot_action'].notna().sum()
    print(f"\n✅ Mapped {mapped}/{len(df)} segments ({mapped/len(df)*100:.1f}%)")
    print(f"✅ Robot action vocabulary: {len(robot_actions)} actions")
    
    # Save mapped data
    df_mapped = df[df['robot_action'].notna()]
    df_mapped.to_csv(output_dir / 'epic_mapped.csv', index=False)
    
    print(f"\n💾 Saved to: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    download_epic_annotations()
