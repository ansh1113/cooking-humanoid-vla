"""
PHASE 1: GPT-4V Labeling with Structured JSON Output
BEST POSSIBLE APPROACH - Industry Standard
"""
import base64
import json
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import time

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def label_video(client, video_dir):
    """Label one video with 8 frames + structured JSON output"""
    all_frames = sorted(list(video_dir.glob('frame_*.jpg')))
    
    if len(all_frames) < 8:
        return None
    
    # Sample 8 frames evenly across the video
    n = len(all_frames)
    indices = [0, n//7, 2*n//7, 3*n//7, 4*n//7, 5*n//7, 6*n//7, n-1]
    frames = [all_frames[min(i, n-1)] for i in indices]
    
    # Encode frames
    images_b64 = [encode_image(f) for f in frames]
    
    # Structured prompt for JSON output
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "Analyze these 8 sequential frames from a cooking video. Identify the MAIN cooking action being performed.\n\n"
                        "Return ONLY a JSON object (no markdown, no extra text) with three keys:\n"
                        "1. 'verb': The manipulation action (e.g., 'slicing', 'pouring', 'stirring', 'mixing', 'chopping', 'whisking')\n"
                        "2. 'noun': The object being manipulated (e.g., 'onion', 'water', 'batter', 'vegetables')\n"
                        "3. 'label': The combined action (e.g., 'slicing onion', 'pouring water', 'stirring batter')\n\n"
                        "Example output:\n"
                        '{"verb": "chopping", "noun": "garlic", "label": "chopping garlic"}'
                    )
                },
                *[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "low"}
                    } 
                    for img in images_b64
                ]
            ]
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=100,
            temperature=0,
            response_format={"type": "json_object"}  # Force JSON output
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON
        try:
            action_data = json.loads(content)
            return action_data
        except json.JSONDecodeError:
            # Fallback: extract from text
            return {
                'verb': 'unknown',
                'noun': 'unknown', 
                'label': content.lower()
            }
    
    except Exception as e:
        print(f"      Error: {e}")
        time.sleep(2)
        return None

def main():
    print("="*70)
    print("🏆 PHASE 1: GPT-4V LABELING (INDUSTRY STANDARD)")
    print("="*70)
    print("\n✨ Features:")
    print("   • 8-frame sparse sampling")
    print("   • Structured JSON output (verb + noun + label)")
    print("   • GPT-4o with vision")
    print("="*70)
    
    # Get API key
    api_key = input("\n🔑 Enter your OpenAI API key: ").strip()
    
    if not api_key or not api_key.startswith('sk-'):
        print("❌ Invalid API key!")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Find videos
    frames_dir = Path('data/processed/frames_all')
    video_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    
    print(f"\n📊 Found {len(video_dirs)} videos to label")
    print(f"🎬 Using 8 frames per video")
    print(f"💰 Estimated cost: ${len(video_dirs) * 8 * 0.0015:.2f} - ${len(video_dirs) * 8 * 0.002:.2f}")
    
    proceed = input("\n⚡ Proceed with labeling? (yes/no): ").strip().lower()
    if proceed != 'yes':
        print("Cancelled.")
        return
    
    # Label each video
    results = []
    failed = []
    
    print("\n🎬 Labeling videos...")
    for video_dir in tqdm(video_dirs, desc="Processing"):
        action_data = label_video(client, video_dir)
        
        if action_data:
            results.append({
                'video_id': video_dir.name,
                **action_data
            })
        else:
            failed.append(video_dir.name)
        
        # Rate limiting (50 RPM limit)
        time.sleep(1.2)
    
    # Save results
    output_file = 'data/processed/gpt4v_labels.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ LABELING COMPLETE!")
    print("="*70)
    print(f"✅ Successfully labeled: {len(results)}/{len(video_dirs)} videos")
    
    if failed:
        print(f"❌ Failed: {len(failed)} videos")
    
    print(f"💾 Saved to: {output_file}")
    
    # Analyze results
    from collections import Counter
    
    verbs = [r['verb'] for r in results if 'verb' in r]
    nouns = [r['noun'] for r in results if 'noun' in r]
    labels = [r['label'] for r in results if 'label' in r]
    
    print(f"\n📊 STATISTICS:")
    print(f"   Unique verbs: {len(Counter(verbs))}")
    print(f"   Unique nouns: {len(Counter(nouns))}")
    print(f"   Unique actions: {len(Counter(labels))}")
    
    print(f"\n🔪 Top 15 Verbs:")
    for i, (verb, count) in enumerate(Counter(verbs).most_common(15), 1):
        print(f"   {i:2d}. {verb:20s}: {count:3d}")
    
    print(f"\n🥕 Top 15 Objects:")
    for i, (noun, count) in enumerate(Counter(nouns).most_common(15), 1):
        print(f"   {i:2d}. {noun:20s}: {count:3d}")
    
    print(f"\n🎯 Top 30 Actions:")
    for i, (label, count) in enumerate(Counter(labels).most_common(30), 1):
        print(f"   {i:2d}. {label:30s}: {count:3d}")
    
    print("\n" + "="*70)
    print("🎉 PHASE 1 COMPLETE!")
    print("📋 Next: Phase 2 - Retrain Temporal VLA with these labels")
    print("="*70)

if __name__ == "__main__":
    main()
