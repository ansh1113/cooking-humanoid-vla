"""
PHASE 1: Label videos with LLaVA-Video
Uses LanguageBind/Video-LLaVA-7B for action recognition
"""
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from pathlib import Path
import cv2
import json
from tqdm import tqdm
import numpy as np

def load_video_frames(video_dir, num_frames=8):
    """Load uniformly sampled frames from video directory"""
    frame_files = sorted(list(Path(video_dir).glob('frame_*.jpg')))
    
    if len(frame_files) == 0:
        return None
    
    # Sample uniformly
    indices = np.linspace(0, len(frame_files)-1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        img = cv2.imread(str(frame_files[idx]))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    
    return frames if len(frames) == num_frames else None

def main():
    print("="*70)
    print("🎬 PHASE 1: AUTO-LABELING WITH LLaVA-Video")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    if not torch.cuda.is_available():
        print("❌ No GPU available! Exiting...")
        return
    
    # Load LLaVA model
    print("\n📥 Loading LLaVA-Video model (this takes ~5 minutes)...")
    model_name = "LanguageBind/Video-LLaVA-7B"
    
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Using simpler Video-ChatGPT approach...")
        # Fallback to alternative
        return
    
    # Find all video directories
    frames_dir = Path('data/processed/frames_all')
    if not frames_dir.exists():
        print(f"❌ Directory not found: {frames_dir}")
        return
    
    video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
    print(f"\n📊 Found {len(video_dirs)} videos to label")
    
    # Prompts for cooking action recognition
    prompts = [
        "What cooking action is being performed in this video? Respond with just the action name (e.g., 'chopping', 'stirring', 'pouring').",
        "Describe the specific cooking technique shown in this video clip in one or two words.",
        "What is the person doing with the food in this video?"
    ]
    
    # Label each video
    results = []
    
    for video_dir in tqdm(video_dirs, desc="🎬 Labeling videos"):
        try:
            frames = load_video_frames(video_dir, num_frames=8)
            if frames is None:
                print(f"   ⚠️  Skipping {video_dir.name} (insufficient frames)")
                continue
            
            # Process with LLaVA
            inputs = processor(
                text=prompts[0],
                images=frames,
                return_tensors="pt"
            ).to(device)
            
            # Generate response
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
            
            response = processor.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract action (clean up response)
            action = response.split('\n')[0].strip().lower()
            action = action.replace('answer:', '').replace('action:', '').strip()
            
            results.append({
                'video_id': video_dir.name,
                'action': action,
                'raw_response': response
            })
            
            if len(results) % 10 == 0:
                print(f"   Processed {len(results)}/{len(video_dirs)} videos...")
            
        except Exception as e:
            print(f"   ❌ Error processing {video_dir.name}: {e}")
            continue
    
    # Save results
    output_file = 'data/processed/llava_labels.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Labeled {len(results)}/{len(video_dirs)} videos")
    print(f"💾 Saved to: {output_file}")
    
    # Show action distribution
    from collections import Counter
    actions = [r['action'] for r in results]
    action_counts = Counter(actions)
    
    print(f"\n📊 Top 20 Actions:")
    for action, count in action_counts.most_common(20):
        print(f"   {action}: {count}")
    
    print("\n" + "="*70)
    print("✅ PHASE 1 COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
