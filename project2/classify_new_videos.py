"""
classify_new_videos.py - Classify only new videos (not already in predictions)
"""
import torch
import open_clip
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm

# Action descriptions (same as before)
ACTION_DESCRIPTIONS = {
    'PickupObject': ['picking up an object', 'grabbing something', 'taking an item', 'hand reaching for object'],
    'PutObject': ['placing an object down', 'putting something on surface', 'setting down an item', 'hand releasing object'],
    'SliceObject': ['cutting with knife', 'slicing food', 'chopping vegetables', 'knife cutting motion'],
    'CookObject': ['cooking food in pan', 'heating on stove', 'food being cooked', 'using heat to cook'],
    'OpenObject': ['opening container', 'hand opening door', 'uncovering object'],
    'CloseObject': ['closing container', 'hand closing door', 'covering object'],
    'ToggleObjectOn': ['turning on appliance', 'starting device', 'hand pressing switch on'],
    'ToggleObjectOff': ['turning off appliance', 'stopping device', 'hand pressing switch off'],
    'DropHandObject': ['dropping object', 'releasing item from hand', 'letting go of object'],
    'ThrowObject': ['throwing object', 'tossing item', 'hand throwing motion'],
    'BreakObject': ['breaking object', 'cracking egg', 'smashing item']
}

def load_clip_model(device):
    print("📥 Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    print("✅ CLIP loaded!")
    return model, preprocess, tokenizer

def classify_frame(model, preprocess, tokenizer, image_path, action_prompts, device):
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    all_texts = []
    action_indices = []
    
    for action, descriptions in action_prompts.items():
        for desc in descriptions:
            all_texts.append(desc)
            action_indices.append(action)
    
    text_tokens = tokenizer(all_texts).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarities = (image_features @ text_features.T).squeeze(0)
    
    action_scores = {}
    for action in ACTION_DESCRIPTIONS.keys():
        action_mask = [i for i, a in enumerate(action_indices) if a == action]
        action_scores[action] = similarities[action_mask].mean().item()
    
    predicted_action = max(action_scores.items(), key=lambda x: x[1])
    return predicted_action[0], action_scores

def main():
    print("="*70)
    print("🤖 CLASSIFYING NEW VIDEOS ONLY")
    print("="*70)
    
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Load existing predictions
    existing_preds_file = Path('data/processed/clip_predictions.json')
    if existing_preds_file.exists():
        with open(existing_preds_file, 'r') as f:
            existing_predictions = json.load(f)
        print(f"✅ Found {len(existing_predictions)} already classified videos")
    else:
        existing_predictions = {}
    
    # Load frame metadata
    with open('data/processed/frames/frame_metadata.json', 'r') as f:
        frame_metadata = json.load(f)
    
    # Find new videos
    all_videos = set(frame_metadata.keys())
    already_done = set(existing_predictions.keys())
    new_videos = all_videos - already_done
    
    print(f"📊 Total videos: {len(all_videos)}")
    print(f"📊 Already classified: {len(already_done)}")
    print(f"📊 New to classify: {len(new_videos)}")
    
    if not new_videos:
        print("\n✅ All videos already classified!")
        return
    
    # Load CLIP
    model, preprocess, tokenizer = load_clip_model(device)
    action_prompts = ACTION_DESCRIPTIONS
    
    # Classify new videos
    new_predictions = {}
    
    for video_id in sorted(new_videos):
        print(f"\n📹 Video: {video_id}")
        frame_dir = Path(f'data/processed/frames/{video_id}')
        
        if not frame_dir.exists():
            continue
        
        frame_files = sorted(frame_dir.glob('*.jpg'))
        print(f"   Processing {len(frame_files)} frames...")
        
        predictions = []
        for frame_file in tqdm(frame_files, desc=f"   {video_id}", leave=False):
            action, scores = classify_frame(
                model, preprocess, tokenizer, frame_file, action_prompts, device
            )
            
            timestamp_str = frame_file.stem.split('_t')[1].replace('s', '')
            timestamp = float(timestamp_str)
            
            predictions.append({
                'frame': frame_file.name,
                'timestamp': timestamp,
                'predicted_action': action,
                'scores': scores
            })
        
        new_predictions[video_id] = predictions
        print(f"   ✅ Classified {len(predictions)} frames")
    
    # Merge with existing
    all_predictions = {**existing_predictions, **new_predictions}
    
    # Save
    with open('data/processed/clip_predictions.json', 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    total_frames = sum(len(preds) for preds in all_predictions.values())
    
    print("\n" + "="*70)
    print(f"✅ Total videos classified: {len(all_predictions)}")
    print(f"✅ Total frames: {total_frames}")
    print(f"✅ New videos added: {len(new_predictions)}")
    print("="*70)

if __name__ == "__main__":
    main()
