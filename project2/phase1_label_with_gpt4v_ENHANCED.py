"""
PHASE 1: GPT-4V Labeling with ENHANCED Metadata
Industry Standard + Rich Training Data
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
    """Label one video with 8 frames + enhanced structured metadata"""
    all_frames = sorted(list(video_dir.glob('frame_*.jpg')))
    
    if len(all_frames) < 8:
        return None
    
    # Sample 8 frames evenly across the video
    n = len(all_frames)
    indices = [0, n//7, 2*n//7, 3*n//7, 4*n//7, 5*n//7, 6*n//7, n-1]
    frames = [all_frames[min(i, n-1)] for i in indices]
    
    # Encode frames
    images_b64 = [encode_image(f) for f in frames]
    
    # ENHANCED prompt with rich metadata
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "Analyze these 8 sequential frames from a cooking video. Identify the MAIN cooking action and relevant context.\n\n"
                        "Return ONLY a JSON object (no markdown, no extra text) with these keys:\n\n"
                        "1. 'verb': The manipulation action (e.g., 'slicing', 'pouring', 'stirring', 'mixing', 'chopping', 'whisking', 'peeling', 'grating')\n"
                        "2. 'noun': The primary object being manipulated (e.g., 'onion', 'water', 'batter', 'vegetables', 'meat')\n"
                        "3. 'label': The combined action (e.g., 'slicing onion', 'pouring water', 'stirring batter')\n"
                        "4. 'tools': List of tools visible/being used (e.g., ['knife', 'cutting board'], ['whisk', 'bowl'], ['spatula', 'pan']). Empty list [] if none clearly visible.\n"
                        "5. 'container': Primary container if any (e.g., 'bowl', 'pan', 'pot', 'plate', 'cutting board'). Use null if none.\n"
                        "6. 'confidence': Your confidence in the label - 'high', 'medium', or 'low'\n\n"
                        "Example outputs:\n"
                        '{"verb": "chopping", "noun": "garlic", "label": "chopping garlic", "tools": ["knife", "cutting board"], "container": "cutting board", "confidence": "high"}\n'
                        '{"verb": "stirring", "noun": "soup", "label": "stirring soup", "tools": ["ladle"], "container": "pot", "confidence": "high"}\n'
                        '{"verb": "pouring", "noun": "oil", "label": "pouring oil", "tools": ["bottle"], "container": "pan", "confidence": "medium"}'
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
            max_tokens=150,
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON
        try:
            action_data = json.loads(content)
            # Ensure all fields exist
            action_data.setdefault('tools', [])
            action_data.setdefault('container', None)
            action_data.setdefault('confidence', 'medium')
            return action_data
        except json.JSONDecodeError:
            print(f"      JSON parse error: {content[:100]}")
            return None
    
    except Exception as e:
        print(f"      Error: {e}")
        time.sleep(2)
        return None

def main():
    print("="*70)
    print("🏆 PHASE 1: GPT-4V ENHANCED LABELING")
    print("="*70)
    print("\n✨ Features:")
    print("   • 8-frame sparse sampling")
    print("   • Structured JSON output")
    print("   • Verb + Noun + Label")
    print("   • Tools + Container metadata")
    print("   • Confidence scores")
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
    print(f"💎 Enhanced with: tools, containers, confidence scores")
    
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
        
        # Rate limiting (50 RPM)
        time.sleep(1.2)
    
    # Save results
    output_file = 'data/processed/gpt4v_labels_enhanced.json'
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
    
    # NEW: Analyze tools and containers
    all_tools = []
    for r in results:
        if 'tools' in r and r['tools']:
            all_tools.extend(r['tools'])
    
    containers = [r['container'] for r in results if r.get('container')]
    confidence_dist = Counter([r.get('confidence', 'unknown') for r in results])
    
    print(f"\n📊 STATISTICS:")
    print(f"   Unique verbs: {len(Counter(verbs))}")
    print(f"   Unique nouns: {len(Counter(nouns))}")
    print(f"   Unique actions: {len(Counter(labels))}")
    print(f"   Unique tools: {len(Counter(all_tools))}")
    print(f"   Unique containers: {len(Counter(containers))}")
    
    print(f"\n🔪 Top 15 Verbs:")
    for i, (verb, count) in enumerate(Counter(verbs).most_common(15), 1):
        print(f"   {i:2d}. {verb:20s}: {count:3d}")
    
    print(f"\n🥕 Top 15 Objects:")
    for i, (noun, count) in enumerate(Counter(nouns).most_common(15), 1):
        print(f"   {i:2d}. {noun:20s}: {count:3d}")
    
    print(f"\n🔧 Top 10 Tools:")
    for i, (tool, count) in enumerate(Counter(all_tools).most_common(10), 1):
        print(f"   {i:2d}. {tool:20s}: {count:3d}")
    
    print(f"\n🥘 Top 10 Containers:")
    for i, (container, count) in enumerate(Counter(containers).most_common(10), 1):
        print(f"   {i:2d}. {container:20s}: {count:3d}")
    
    print(f"\n📈 Confidence Distribution:")
    for conf, count in confidence_dist.most_common():
        print(f"   {conf:10s}: {count:3d} ({count/len(results)*100:.1f}%)")
    
    print(f"\n🎯 Top 30 Actions:")
    for i, (label, count) in enumerate(Counter(labels).most_common(30), 1):
        print(f"   {i:2d}. {label:35s}: {count:3d}")
    
    print("\n" + "="*70)
    print("🎉 PHASE 1 COMPLETE!")
    print("💎 Dataset includes:")
    print("   ✅ Actions (verb + noun)")
    print("   ✅ Tools used")
    print("   ✅ Containers")
    print("   ✅ Confidence scores")
    print("\n📋 Next: Phase 2 - Train VLA with this rich dataset!")
    print("="*70)

if __name__ == "__main__":
    main()
