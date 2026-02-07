"""
parse_captions.py - Extract action annotations from video captions
Maps natural language → Project 1 actions
"""
import json
from pathlib import Path
import re

# Map keywords to Project 1 actions
ACTION_KEYWORDS = {
    'PickupObject': ['pick up', 'pickup', 'grab', 'take', 'get', 'hold'],
    'PutObject': ['put', 'place', 'set down', 'add', 'transfer'],
    'OpenObject': ['open'],
    'CloseObject': ['close', 'shut'],
    'SliceObject': ['slice', 'cut', 'chop', 'dice', 'mince'],
    'CookObject': ['cook', 'heat', 'fry', 'bake', 'boil', 'sauté'],
    'ToggleObjectOn': ['turn on', 'start'],
    'ToggleObjectOff': ['turn off', 'stop'],
    'DropHandObject': ['drop', 'release'],
    'ThrowObject': ['throw', 'toss'],
    'BreakObject': ['break', 'crack', 'smash']
}

# Common cooking objects
COOKING_OBJECTS = [
    'knife', 'tomato', 'lettuce', 'onion', 'garlic', 'pepper', 'carrot',
    'potato', 'cucumber', 'apple', 'lemon', 'bowl', 'pan', 'pot',
    'plate', 'cutting board', 'spoon', 'fork', 'oil', 'salt', 'water',
    'cheese', 'bread', 'meat', 'chicken', 'egg', 'pasta', 'rice'
]

def parse_vtt_file(vtt_path):
    """Parse WebVTT subtitle file"""
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove WebVTT header
    content = re.sub(r'WEBVTT.*?\n\n', '', content, flags=re.DOTALL)
    
    # Split into subtitle blocks
    blocks = content.strip().split('\n\n')
    
    captions = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 2:
            # First line is timestamp, rest is text
            timestamp_line = lines[0]
            text = ' '.join(lines[1:])
            
            # Parse timestamp (format: 00:00:00.000 --> 00:00:03.000)
            if '-->' in timestamp_line:
                start, end = timestamp_line.split('-->')
                captions.append({
                    'start': start.strip(),
                    'end': end.strip(),
                    'text': text.strip()
                })
    
    return captions

def timestamp_to_seconds(timestamp):
    """Convert timestamp to seconds"""
    # Format: 00:00:12.500
    parts = timestamp.split(':')
    if len(parts) == 3:
        h, m, s = parts
        seconds = int(h) * 3600 + int(m) * 60 + float(s)
        return seconds
    return 0

def detect_action(text):
    """Detect action from text"""
    text_lower = text.lower()
    
    for action, keywords in ACTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return action
    
    return None

def detect_object(text):
    """Detect cooking object from text"""
    text_lower = text.lower()
    
    for obj in COOKING_OBJECTS:
        if obj in text_lower:
            return obj
    
    return None

def annotate_captions(captions):
    """Add action annotations to captions"""
    annotated = []
    
    for caption in captions:
        action = detect_action(caption['text'])
        obj = detect_object(caption['text'])
        
        if action:  # Only keep captions with detected actions
            annotated.append({
                'start_time': timestamp_to_seconds(caption['start']),
                'end_time': timestamp_to_seconds(caption['end']),
                'text': caption['text'],
                'action': action,
                'object': obj
            })
    
    return annotated

def process_all_videos():
    """Process all video captions"""
    
    print("="*70)
    print("📝 CAPTION PARSING & ACTION ANNOTATION")
    print("="*70)
    
    caption_dir = Path('data/raw/videos')
    vtt_files = list(caption_dir.glob('*.vtt'))
    
    print(f"Found {len(vtt_files)} caption files")
    
    all_annotations = {}
    total_actions = 0
    
    for vtt_file in vtt_files:
        video_id = vtt_file.stem.replace('.en', '')
        
        print(f"\n📹 Processing: {video_id}")
        
        # Parse captions
        captions = parse_vtt_file(vtt_file)
        print(f"   Found {len(captions)} caption segments")
        
        # Annotate with actions
        annotated = annotate_captions(captions)
        print(f"   Detected {len(annotated)} action segments")
        
        all_annotations[video_id] = annotated
        total_actions += len(annotated)
        
        # Show examples
        if annotated:
            print(f"   Examples:")
            for i, ann in enumerate(annotated[:3]):
                print(f"      • {ann['action']}: \"{ann['text'][:50]}...\"")
    
    # Save annotations
    output_file = Path('data/processed/action_annotations.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_annotations, f, indent=2)
    
    # Action statistics
    action_counts = {}
    for video_annotations in all_annotations.values():
        for ann in video_annotations:
            action = ann['action']
            action_counts[action] = action_counts.get(action, 0) + 1
    
    print("\n" + "="*70)
    print(f"✅ Processed {len(vtt_files)} videos")
    print(f"📊 Total action segments: {total_actions}")
    print(f"\n📊 Action Distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"   {action:20s} {count:4d}")
    print(f"\n💾 Saved to: {output_file}")
    print("="*70)

if __name__ == "__main__":
    process_all_videos()
