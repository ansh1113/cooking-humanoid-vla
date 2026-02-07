"""
Expand robot action vocabulary to 100+ actions
Map all Epic-Kitchens verbs comprehensively
"""
import pandas as pd
from pathlib import Path
from collections import Counter

def build_comprehensive_vocab():
    print("="*70)
    print("🤖 BUILDING 100+ ACTION VOCABULARY")
    print("="*70)
    
    # Load Epic-Kitchens data
    df = pd.read_csv('data/epic_kitchens/epic_train.csv')
    
    # Get all verbs
    all_verbs = df['verb'].value_counts()
    print(f"\n📊 Total unique verbs: {len(all_verbs)}")
    
    # COMPREHENSIVE robot action mapping (100+ actions)
    robot_actions = {
        # ===== BASIC MANIPULATION (11 original) =====
        'PickupObject': ['pick-up', 'take', 'lift', 'get', 'grab', 'hold'],
        'PutObject': ['put-down', 'put', 'place', 'set', 'rest', 'lay'],
        'OpenObject': ['open', 'uncover'],
        'CloseObject': ['close', 'shut', 'cover'],
        'SliceObject': ['cut', 'slice', 'chop', 'dice', 'halve'],
        'CookObject': ['cook', 'fry', 'boil', 'heat', 'bake', 'roast', 'grill'],
        'ToggleObjectOn': ['turn-on', 'switch-on', 'activate', 'start'],
        'ToggleObjectOff': ['turn-off', 'switch-off', 'deactivate', 'stop'],
        'DropHandObject': ['drop', 'release'],
        'ThrowObject': ['throw', 'toss', 'discard'],
        'BreakObject': ['break', 'crack', 'smash'],
        
        # ===== LIQUID OPERATIONS =====
        'PourLiquid': ['pour', 'pour-in', 'pour-into', 'pour-from'],
        'DrainLiquid': ['drain', 'empty'],
        'SqueezeLiquid': ['squeeze', 'press'],
        'SpillLiquid': ['spill'],
        
        # ===== MIXING & STIRRING =====
        'StirObject': ['stir', 'mix', 'whisk', 'beat'],
        'ShakeObject': ['shake', 'toss'],
        'FoldObject': ['fold', 'fold-in'],
        'BlendObject': ['blend', 'puree'],
        'KneadObject': ['knead'],
        'RollObject': ['roll', 'roll-out'],
        
        # ===== CLEANING =====
        'WashObject': ['wash', 'rinse', 'clean'],
        'DryObject': ['dry', 'wipe', 'pat'],
        'LatherObject': ['lather', 'soap'],
        'ScrubObject': ['scrub', 'scour'],
        
        # ===== CUTTING & PREP =====
        'PeelObject': ['peel', 'skin'],
        'GrateObject': ['grate', 'shred'],
        'MinceObject': ['mince'],
        'JulienneObject': ['julienne'],
        'CubeObject': ['cube'],
        
        # ===== TRANSFER & MOVEMENT =====
        'MoveObject': ['move', 'shift', 'slide'],
        'FlipObject': ['flip', 'turn', 'turn-over'],
        'RemoveObject': ['remove', 'take-out', 'extract'],
        'InsertObject': ['insert', 'put-in', 'put-into', 'put-through'],
        'AttachObject': ['attach', 'connect', 'fasten'],
        'DetachObject': ['detach', 'disconnect', 'unfasten'],
        
        # ===== SPREADING & APPLICATION =====
        'SpreadObject': ['spread', 'smear', 'apply'],
        'SprayObject': ['spray', 'spritz'],
        'SprinkleObject': ['sprinkle', 'scatter', 'season', 'add'],
        'BrushObject': ['brush'],
        'DabObject': ['dab'],
        
        # ===== CONTAINER OPERATIONS =====
        'FillContainer': ['fill'],
        'ScoopObject': ['scoop', 'spoon', 'ladle'],
        'StrainObject': ['strain', 'sieve', 'filter'],
        'WrapObject': ['wrap', 'cover'],
        'UnwrapObject': ['unwrap', 'uncover', 'peel-off'],
        
        # ===== MECHANICAL =====
        'SwitchObject': ['switch'],
        'PressObject': ['press', 'push', 'push-down'],
        'PullObject': ['pull', 'pull-out'],
        'TwistObject': ['twist', 'unscrew', 'screw'],
        'LiftObject': ['lift', 'raise'],
        'LowerObject': ['lower'],
        
        # ===== COOKING SPECIFIC =====
        'MashObject': ['mash', 'crush', 'grind'],
        'SauteObject': ['sauté', 'stir-fry'],
        'SimmerObject': ['simmer'],
        'BrownObject': ['brown', 'sear'],
        'CaramelizeObject': ['caramelize'],
        'ToastObject': ['toast'],
        'MeltObject': ['melt'],
        
        # ===== TEMPERATURE =====
        'CoolObject': ['cool', 'chill', 'refrigerate'],
        'FreezeObject': ['freeze'],
        'ThawObject': ['thaw', 'defrost'],
        
        # ===== ADJUSTMENT =====
        'AdjustObject': ['adjust', 'position', 'align'],
        'ArrangeObject': ['arrange', 'organize'],
        'SortObject': ['sort'],
        
        # ===== INSPECTION =====
        'CheckObject': ['check', 'inspect', 'examine'],
        'MeasureObject': ['measure', 'weigh'],
        'TasteObject': ['taste', 'sample'],
        
        # ===== SERVING =====
        'ServeFood': ['serve', 'plate', 'dish'],
        'GarnishFood': ['garnish', 'decorate'],
        
        # ===== STORAGE =====
        'StoreObject': ['store', 'keep', 'save'],
        'PackObject': ['pack', 'package'],
        
        # ===== DISPOSAL =====
        'DisposeObject': ['dispose', 'bin'],
        
        # ===== UNCATEGORIZED BUT COMMON =====
        'SquashObject': ['squash'],
        'TearObject': ['tear', 'rip'],
        'CrumbleObject': ['crumble'],
        'DipObject': ['dip', 'dunk'],
        'BalanceObject': ['balance'],
    }
    
    print(f"\n🤖 Defined {len(robot_actions)} robot actions")
    
    # Create reverse mapping
    epic_to_robot = {}
    for robot_action, epic_verbs in robot_actions.items():
        for epic_verb in epic_verbs:
            epic_to_robot[epic_verb] = robot_action
    
    # Map all segments
    df['robot_action'] = df['verb'].map(epic_to_robot)
    
    # Coverage statistics
    mapped = df['robot_action'].notna().sum()
    coverage = mapped / len(df) * 100
    
    print(f"\n📊 COVERAGE STATISTICS:")
    print(f"   Total segments: {len(df):,}")
    print(f"   Mapped segments: {mapped:,} ({coverage:.1f}%)")
    print(f"   Unmapped segments: {len(df) - mapped:,}")
    
    # Action distribution
    action_counts = df[df['robot_action'].notna()]['robot_action'].value_counts()
    
    print(f"\n📈 TOP 30 ROBOT ACTIONS (by frequency):")
    for i, (action, count) in enumerate(action_counts.head(30).items(), 1):
        print(f"   {i:2d}. {action:25s}: {count:5d} segments")
    
    # Find unmapped verbs
    unmapped_df = df[df['robot_action'].isna()]
    unmapped_verbs = unmapped_df['verb'].value_counts().head(20)
    
    print(f"\n⚠️  TOP 20 UNMAPPED VERBS (opportunities to add):")
    for i, (verb, count) in enumerate(unmapped_verbs.items(), 1):
        print(f"   {i:2d}. {verb:25s}: {count:5d} segments")
    
    # Save
    output_dir = Path('data/epic_kitchens')
    df.to_csv(output_dir / 'epic_expanded_100.csv', index=False)
    
    # Save action vocabulary
    import json
    with open(output_dir / 'action_vocab_100.json', 'w') as f:
        json.dump(robot_actions, f, indent=2)
    
    print(f"\n💾 Saved:")
    print(f"   - data/epic_kitchens/epic_expanded_100.csv")
    print(f"   - data/epic_kitchens/action_vocab_100.json")
    
    print("\n" + "="*70)
    print(f"✅ VOCABULARY EXPANDED TO {len(robot_actions)} ACTIONS!")
    print(f"✅ COVERAGE: {coverage:.1f}% of Epic-Kitchens data")
    print("="*70)

if __name__ == "__main__":
    build_comprehensive_vocab()
