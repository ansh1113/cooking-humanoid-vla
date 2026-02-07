"""
Label the 120 newly curated videos with GPT-4V
"""
import json
from openai import OpenAI
import base64
import subprocess
import cv2
from pathlib import Path
from tqdm import tqdm
import time

def download_and_extract_frames(video_url):
    """Download video and extract 8 frames"""
    # Download
    from stage1_generate_plan import download_youtube_video, extract_frames
    
    result = download_youtube_video(video_url)
    if isinstance(result, tuple):
        video_path, title = result
    else:
        video_path = result
        title = "Unknown"
    
    if not video_path:
        return None, None
    
    # Extract frames
    frames = extract_frames(video_path, max_frames=60)
    
    # Sample 8 frames
    if len(frames) >= 8:
        indices = [0, len(frames)//7, 2*len(frames)//7, 3*len(frames)//7,
                   4*len(frames)//7, 5*len(frames)//7, 6*len(frames)//7, len(frames)-1]
        sampled = [frames[i] for i in indices]
    else:
        sampled = frames
    
    return sampled, title

def encode_image_cv2(frame):
    """Encode OpenCV frame to base64"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def label_video_gpt4v(client, frames):
    """Label video with GPT-4V (same as Phase 1)"""
    images_b64 = [encode_image_cv2(f) for f in frames]
    
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Analyze these 8 sequential frames from a cooking video. "
                    "Return ONLY a JSON object with:\n"
                    "1. 'verb': Main action (e.g., 'slicing', 'stirring')\n"
                    "2. 'noun': Object (e.g., 'onion', 'curry')\n"
                    "3. 'label': Combined (e.g., 'slicing onion')\n"
                    "4. 'tools': List of tools (e.g., ['knife', 'pan'])\n"
                    "5. 'container': Container (e.g., 'bowl', 'pan')\n"
                    "6. 'confidence': 'high', 'medium', or 'low'\n"
                    "No markdown, just JSON."
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
    print("🏷️  LABELING 120 NEW DIVERSE VIDEOS")
    print("="*70)
    
    # Get API key
    api_key = input("\n🔑 OpenAI API key: ").strip()
    client = OpenAI(api_key=api_key)
    
    # Load curated videos
    with open('data/curated_diverse_videos_clean.json') as f:
        curated = json.load(f)
    
    # Flatten to list
    all_videos = []
    for category, videos in curated.items():
        for video in videos:
            video['category'] = category
            all_videos.append(video)
    
    print(f"\n📊 Total videos to label: {len(all_videos)}")
    print(f"💰 Estimated cost: ${len(all_videos) * 8 * 0.0015:.2f}")
    
    proceed = input("\nProceed? (yes/no): ").strip().lower()
    if proceed != 'yes':
        return
    
    # Label each video
    results = []
    failed = []
    
    print("\n🎬 Labeling videos...")
    for video in tqdm(all_videos):
        try:
            # Download and extract
            frames, title = download_and_extract_frames(video['url'])
            if frames is None:
                failed.append(video['url'])
                continue
            
            # Label with GPT-4V
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
            
            time.sleep(1.2)  # Rate limiting
            
        except Exception as e:
            print(f"      Error: {e}")
            failed.append(video['url'])
    
    # Save
    with open('data/processed/new_diverse_labels.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ LABELING COMPLETE!")
    print("="*70)
    print(f"✅ Labeled: {len(results)}/{len(all_videos)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"💾 Saved to: data/processed/new_diverse_labels.json")
    
    # Show stats
    from collections import Counter
    actions = [r['label'] for r in results]
    print(f"\n📊 Top 20 Actions:")
    for i, (action, count) in enumerate(Counter(actions).most_common(20), 1):
        print(f"   {i:2d}. {action:35s}: {count:3d}")

if __name__ == "__main__":
    main()
