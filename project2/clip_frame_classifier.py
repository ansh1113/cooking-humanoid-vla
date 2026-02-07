"""
clip_frame_classifier.py - Use CLIP to classify video frames into actions
Maps frames → Project 1 actions using vision-language understanding
"""
import torch
import open_clip
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

# Project 1 actions with natural language descriptions
ACTION_DESCRIPTIONS = {
    'PickupObject': [
        'picking up an object',
        'grabbing something',
        'taking an item',
        'hand reaching for object'
    ],
    'PutObject': [
        'placing an object down',
        'putting something on surface',
        'setting down an item',
        'hand releasing object'
    ],
    'SliceObject': [
        'cutting with knife',
        'slicing food',
        'chopping vegetables',
        'knife cutting motion'
    ],
    'CookObject': [
        'cooking food in pan',
        'heating on stove',
        'food being cooked',
        'using heat to cook'
    ],
    'OpenObject': [
        'opening container',
        'hand opening door',
        'uncovering object'
    ],
    'CloseObject': [
        'closing container',
        'hand closing door',
        'covering object'
    ],
    'ToggleObjectOn': [
        'turning on appliance',
        'starting device',
        'hand pressing switch on'
    ],
    'ToggleObjectOff': [
        'turning off appliance',
        'stopping device',
        'hand pressing switch off'
    ],
    'DropHandObject': [
        'dropping object',
        'releasing item from hand',
        'letting go of object'
    ],
    'ThrowObject': [
        'throwing object',
        'tossing item',
        'hand throwing motion'
    ],
    'BreakObject': [
        'breaking object',
        'cracking egg',
        'smashing item'
    ]
}

def load_clip_model(device):
    """Load CLIP model"""
    print("📥 Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='laion2b_s34b_b79k',
        device=device
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    print("✅ CLIP loaded!")
    return model, preprocess, tokenizer

def create_action_prompts():
    """Create text prompts for each action"""
    prompts = {}
    for action, descriptions in ACTION_DESCRIPTIONS.items():
        # Use all descriptions for robust matching
        prompts[action] = descriptions
    return prompts

def classify_frame(model, preprocess, tokenizer, image_path, action_prompts, device):
    """Classify a single frame using CLIP"""
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Prepare all text prompts
    all_texts = []
    action_indices = []
    
    for action, descriptions in action_prompts.items():
        for desc in descriptions:
            all_texts.append(desc)
            action_indices.append(action)
    
    text_tokens = tokenizer(all_texts).to(device)
    
    # Get embeddings
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        similarities = (image_features @ text_features.T).squeeze(0)
    
    # Average similarities per action
    action_scores = {}
    for action in ACTION_DESCRIPTIONS.keys():
        action_mask = [i for i, a in enumerate(action_indices) if a == action]
        action_scores[action] = similarities[action_mask].mean().item()
    
    # Get top prediction
    predicted_action = max(action_scores.items(), key=lambda x: x[1])
    
    return predicted_action[0], action_scores

def process_video_frames(video_id, model, preprocess, tokenizer, action_prompts, device):
    """Process all frames for a video"""
    
    frame_dir = Path(f'data/processed/frames/{video_id}')
    
    if not frame_dir.exists():
        return []
    
    frame_files = sorted(frame_dir.glob('*.jpg'))
    
    print(f"   Processing {len(frame_files)} frames...")
    
    predictions = []
    
    for frame_file in tqdm(frame_files, desc=f"   {video_id}", leave=False):
        action, scores = classify_frame(
            model, preprocess, tokenizer, frame_file, action_prompts, device
        )
        
        # Extract timestamp from filename (frame_000123_t45.67s.jpg)
        timestamp_str = frame_file.stem.split('_t')[1].replace('s', '')
        timestamp = float(timestamp_str)
        
        predictions.append({
            'frame': frame_file.name,
            'timestamp': timestamp,
            'predicted_action': action,
            'scores': scores
        })
    
    return predictions

def main():
    print("="*70)
    print("🤖 CLIP-BASED FRAME CLASSIFICATION")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load CLIP
    model, preprocess, tokenizer = load_clip_model(device)
    
    # Create action prompts
    action_prompts = create_action_prompts()
    print(f"✅ Created prompts for {len(action_prompts)} actions")
    
    # Load frame metadata
    metadata_file = Path('data/processed/frames/frame_metadata.json')
    with open(metadata_file, 'r') as f:
        frame_metadata = json.load(f)
    
    video_ids = list(frame_metadata.keys())
    print(f"\n📹 Processing {len(video_ids)} videos...")
    
    all_predictions = {}
    
    for video_id in video_ids:
        print(f"\n📹 Video: {video_id}")
        predictions = process_video_frames(
            video_id, model, preprocess, tokenizer, action_prompts, device
        )
        
        if predictions:
            all_predictions[video_id] = predictions
            print(f"   ✅ Classified {len(predictions)} frames")
            
            # Show action distribution
            action_counts = {}
            for pred in predictions:
                action = pred['predicted_action']
                action_counts[action] = action_counts.get(action, 0) + 1
            
            print(f"   Top actions:")
            for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:3]:
                print(f"      {action}: {count}")
    
    # Save predictions
    output_file = Path('data/processed/clip_predictions.json')
    with open(output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    total_frames = sum(len(preds) for preds in all_predictions.values())
    
    print("\n" + "="*70)
    print(f"✅ Classified {total_frames} frames from {len(all_predictions)} videos")
    print(f"💾 Saved to: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()
