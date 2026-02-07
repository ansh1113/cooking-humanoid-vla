"""
RE-LABEL with the GOOD prompt from Phase 1
"""
import json
from openai import OpenAI
import base64
import cv2
from pathlib import Path
from tqdm import tqdm
import time
from stage1_generate_plan import download_youtube_video, extract_frames

def encode_image_cv2(frame):
    """Encode OpenCV frame to base64"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def download_and_extract_frames(video_url):
    """Download video and extract 8 frames"""
    result = download_youtube_video(video_url)
    if isinstance(result, tuple):
        video_path, title = result
    else:
        video_path = result
        title = "Unknown"
    
    if not video_path:
        return None, None
    
    frames = extract_frames(video_path, max_frames=60)
    
    if len(frames) >= 8:
        indices = [0, len(frames)//7, 2*len(frames)//7, 3*len(frames)//7,
                   4*len(frames)//7, 5*len(frames)//7, 6*len(frames)//7, len(frames)-1]
        sampled = [frames[i] for i in indices]
    else:
        sampled = frames
    
    return sampled, title

def label_video_gpt4v(client, frames):
    """Label with THE GOOD PROMPT from Phase 1"""
    images_b64 = [encode_image_cv2(f) for f in frames]
    
    messages = [{
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
            *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}", "detail": "low"}}
              for img in images_b64]
        ]
    }]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=150,
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"      Error: {e}")
        time.sleep(2)
        return None

def main():
    print("="*70)
    print("🔧 RE-LABELING WITH BETTER PROMPT")
    print("="*70)
    
    api_key = input("\n🔑 OpenAI API key: ").strip()
    client = OpenAI(api_key=api_key)
    
    # Load curated videos
    with open('data/curated_diverse_videos_clean.json') as f:
        curated = json.load(f)
    
    # Flatten
    all_videos = []
    for category, videos in curated.items():
        for video in videos:
            video['category'] = category
            all_videos.append(video)
    
    print(f"\n📊 Re-labeling {len(all_videos)} videos with BETTER prompt")
    print(f"💰 Cost: ${len(all_videos) * 8 * 0.0015:.2f}")
    
    proceed = input("\nProceed? (yes/no): ").strip().lower()
    if proceed != 'yes':
        return
    
    results = []
    failed = []
    
    print("\n🎬 Re-labeling...")
    for video in tqdm(all_videos):
        try:
            frames, title = download_and_extract_frames(video['url'])
            if frames is None:
                failed.append(video['url'])
                continue
            
            label_data = label_video_gpt4v(client, frames)
            if label_data:
                results.append({
                    'video_id': video['url'].split('v=')[-1],
                    'url': video['url'],
                    'title': title,
                    'category': video['category'],
                    **label_data
                })
            else:
                failed.append(video['url'])
            
            time.sleep(1.2)
            
        except Exception as e:
            print(f"      Error: {e}")
            failed.append(video['url'])
    
    # Save
    with open('data/processed/new_diverse_labels_FIXED.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ RE-LABELING COMPLETE!")
    print("="*70)
    print(f"✅ Labeled: {len(results)}/{len(all_videos)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"💾 Saved to: data/processed/new_diverse_labels_FIXED.json")
    
    # Show stats
    from collections import Counter
    actions = [r['label'] for r in results]
    verbs = [r['verb'] for r in results]
    nouns = [r['noun'] for r in results]
    
    print(f"\n📊 Top 20 Actions:")
    for i, (action, count) in enumerate(Counter(actions).most_common(20), 1):
        print(f"   {i:2d}. {action:35s}: {count:3d}")
    
    print(f"\n🔪 Top 15 Verbs:")
    for i, (verb, count) in enumerate(Counter(verbs).most_common(15), 1):
        print(f"   {i:2d}. {verb:20s}: {count:3d}")
    
    print(f"\n🥕 Top 15 Nouns:")
    for i, (noun, count) in enumerate(Counter(nouns).most_common(15), 1):
        print(f"   {i:2d}. {noun:20s}: {count:3d}")

if __name__ == "__main__":
    main()
