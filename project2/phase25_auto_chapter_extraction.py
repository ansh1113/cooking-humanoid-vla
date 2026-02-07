"""
PHASE 2.5: Auto-Chapter Detection and Labeling
Uses Whisper to detect cooking steps, extracts chapters, labels each
"""
import whisper
import json
import re
from pathlib import Path
import cv2
import numpy as np
from openai import OpenAI
import base64
import time
from stage1_generate_plan import download_youtube_video

# Cooking action keywords for chapter detection
COOKING_KEYWORDS = {
    'chopping': ['chop', 'dice', 'cut', 'slice', 'mince'],
    'stirring': ['stir', 'mix', 'combine', 'fold'],
    'adding': ['add', 'pour', 'put', 'sprinkle'],
    'cooking': ['cook', 'fry', 'sauté', 'roast', 'boil', 'simmer'],
    'grinding': ['grind', 'blend', 'paste', 'puree'],
    'marinating': ['marinate', 'coat', 'season'],
    'heating': ['heat', 'warm', 'preheat'],
    'kneading': ['knead', 'dough'],
    'garnishing': ['garnish', 'top', 'serve'],
    'peeling': ['peel', 'skin'],
    'grating': ['grate', 'shred'],
    'whisking': ['whisk', 'beat', 'whip'],
}

def transcribe_video_with_whisper(video_path):
    """Transcribe video with word-level timestamps"""
    print("   🎤 Transcribing with Whisper...")
    
    # Extract audio
    import subprocess
    audio_path = 'temp_audio.mp3'
    cmd = ['ffmpeg', '-y', '-i', video_path, '-ab', '160k', '-ac', '2', 
           '-ar', '44100', '-vn', audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Transcribe
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)
    
    return result

def detect_chapters_from_transcript(transcript):
    """
    Detect cooking steps from transcript
    Returns list of chapters with start/end times
    """
    print("   📖 Detecting chapters from transcript...")
    
    chapters = []
    current_chapter = None
    
    for segment in transcript['segments']:
        text = segment['text'].lower()
        start = segment['start']
        end = segment['end']
        
        # Check for cooking keywords
        detected_action = None
        detected_object = None
        
        for action_type, keywords in COOKING_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    detected_action = action_type
                    break
            if detected_action:
                break
        
        # Extract object (noun after the verb)
        # Simple heuristic: look for common ingredients
        ingredients = ['onion', 'tomato', 'garlic', 'ginger', 'chili', 'pepper',
                      'paneer', 'chicken', 'potato', 'carrot', 'spinach', 'masala',
                      'spice', 'oil', 'butter', 'ghee', 'curry', 'rice', 'dal',
                      'flour', 'dough', 'water', 'milk', 'cream', 'cheese']
        
        for ingredient in ingredients:
            if ingredient in text:
                detected_object = ingredient
                break
        
        # If we detected an action, start a new chapter
        if detected_action:
            # Close previous chapter
            if current_chapter:
                current_chapter['end'] = start
                chapters.append(current_chapter)
            
            # Start new chapter
            current_chapter = {
                'action': detected_action,
                'object': detected_object,
                'start': start,
                'end': None,
                'text': text.strip()
            }
    
    # Close last chapter
    if current_chapter:
        current_chapter['end'] = transcript['segments'][-1]['end']
        chapters.append(current_chapter)
    
    # Merge very short chapters (< 10 seconds) with next
    merged = []
    i = 0
    while i < len(chapters):
        chapter = chapters[i]
        duration = chapter['end'] - chapter['start']
        
        if duration < 10 and i < len(chapters) - 1:
            # Merge with next
            chapters[i+1]['start'] = chapter['start']
            chapters[i+1]['text'] = chapter['text'] + ' ' + chapters[i+1]['text']
        else:
            merged.append(chapter)
        
        i += 1
    
    return merged

def extract_chapter_frames(video_path, chapter, num_frames=8):
    """Extract frames from a specific chapter"""
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(chapter['start'] * fps)
    end_frame = int(chapter['end'] * fps)
    
    total_chapter_frames = end_frame - start_frame
    
    if total_chapter_frames < num_frames:
        indices = range(start_frame, end_frame)
    else:
        step = total_chapter_frames / num_frames
        indices = [int(start_frame + i * step) for i in range(num_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames

def encode_image_cv2(frame):
    """Encode frame to base64"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def label_chapter_with_gpt4v(client, frames, chapter_hint):
    """Label chapter with GPT-4V, using chapter hint"""
    images_b64 = [encode_image_cv2(f) for f in frames]
    
    # Build hint from detected chapter
    hint_text = ""
    if chapter_hint.get('action') and chapter_hint.get('object'):
        hint_text = f" The transcript mentions '{chapter_hint['action']}' and '{chapter_hint['object']}'."
    
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"Analyze these {len(frames)} sequential frames from a cooking video chapter.{hint_text}\n\n"
                    "Identify the MAIN cooking action in this specific step.\n\n"
                    "Return ONLY a JSON object with:\n"
                    "1. 'verb': The manipulation action (e.g., 'chopping', 'stirring', 'adding', 'grinding')\n"
                    "2. 'noun': The SPECIFIC object (e.g., 'onion', 'garlic', 'paneer', 'masala' - be specific!)\n"
                    "3. 'label': Combined action (e.g., 'chopping onion', 'grinding masala')\n"
                    "4. 'tools': List of tools (e.g., ['knife', 'cutting board'])\n"
                    "5. 'container': Container (e.g., 'bowl', 'pan')\n"
                    "6. 'confidence': 'high', 'medium', or 'low'\n\n"
                    "Be SPECIFIC with the noun - don't say 'vegetable' or 'ingredient', identify the actual item!\n\n"
                    "Examples:\n"
                    '{"verb": "chopping", "noun": "onion", "label": "chopping onion", ...}\n'
                    '{"verb": "grinding", "noun": "ginger garlic", "label": "grinding ginger garlic", ...}\n'
                    '{"verb": "stirring", "noun": "curry", "label": "stirring curry", ...}'
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
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"         Error: {e}")
        return None

def process_video_with_chapters(video_url, client):
    """
    Complete pipeline:
    1. Download video
    2. Transcribe with Whisper
    3. Detect chapters
    4. Extract frames per chapter
    5. Label each chapter with GPT-4V
    """
    print(f"\n{'='*70}")
    print(f"📹 Processing: {video_url}")
    print('='*70)
    
    # Download
    print("📥 Downloading...")
    result = download_youtube_video(video_url)
    if isinstance(result, tuple):
        video_path, title = result
    else:
        video_path = result
        title = "Unknown"
    
    print(f"   ✅ {title}")
    
    # Transcribe
    transcript = transcribe_video_with_whisper(video_path)
    
    # Detect chapters
    chapters = detect_chapters_from_transcript(transcript)
    
    print(f"\n   📚 Detected {len(chapters)} chapters:")
    for i, chapter in enumerate(chapters, 1):
        duration = chapter['end'] - chapter['start']
        print(f"      {i}. [{chapter['start']:.0f}s-{chapter['end']:.0f}s] "
              f"({duration:.0f}s) {chapter['action']} {chapter.get('object', '?')}")
        print(f"         \"{chapter['text'][:60]}...\"")
    
    # Label each chapter
    print(f"\n   🏷️  Labeling {len(chapters)} chapters with GPT-4V...")
    labeled_chapters = []
    
    for i, chapter in enumerate(chapters, 1):
        print(f"      Chapter {i}/{len(chapters)}...")
        
        # Extract frames
        frames = extract_chapter_frames(video_path, chapter, num_frames=8)
        
        if len(frames) < 4:
            print(f"         ⚠️  Too few frames, skipping")
            continue
        
        # Label with GPT-4V
        label_data = label_chapter_with_gpt4v(client, frames, chapter)
        
        if label_data:
            labeled_chapters.append({
                'video_url': video_url,
                'video_title': title,
                'chapter_index': i,
                'start_time': chapter['start'],
                'end_time': chapter['end'],
                'duration': chapter['end'] - chapter['start'],
                'transcript_hint': chapter['text'],
                **label_data
            })
            print(f"         ✅ {label_data['label']}")
        
        time.sleep(1.2)  # Rate limiting
    
    return labeled_chapters

def main():
    print("="*70)
    print("🎬 AUTO-CHAPTER EXTRACTION & LABELING")
    print("="*70)
    
    # Get API key
    api_key = input("\n🔑 OpenAI API key: ").strip()
    client = OpenAI(api_key=api_key)
    
    # Load Indian cooking videos
    with open('data/curated_diverse_videos_clean.json') as f:
        curated = json.load(f)
    
    indian_videos = (curated.get('indian_vegetarian', []) + 
                    curated.get('indian_nonveg', []))
    
    print(f"\n📊 Found {len(indian_videos)} Indian cooking videos")
    print(f"💰 Estimated cost: ${len(indian_videos) * 5 * 8 * 0.0015:.2f} (assuming ~5 chapters/video)")
    
    proceed = input("\nProceed? (yes/no): ").strip().lower()
    if proceed != 'yes':
        return
    
    # Process each video
    all_chapters = []
    
    for i, video in enumerate(indian_videos, 1):
        print(f"\n\n{'#'*70}")
        print(f"VIDEO {i}/{len(indian_videos)}")
        print('#'*70)
        
        try:
            chapters = process_video_with_chapters(video['url'], client)
            all_chapters.extend(chapters)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Save all labeled chapters
    with open('data/processed/indian_cooking_chapters.json', 'w') as f:
        json.dump(all_chapters, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ CHAPTER EXTRACTION COMPLETE!")
    print("="*70)
    print(f"📊 Total chapters labeled: {len(all_chapters)}")
    print(f"💾 Saved to: data/processed/indian_cooking_chapters.json")
    
    # Show stats
    from collections import Counter
    actions = [c['label'] for c in all_chapters]
    
    print(f"\n📊 Top 20 Actions:")
    for i, (action, count) in enumerate(Counter(actions).most_common(20), 1):
        print(f"   {i:2d}. {action:35s}: {count:3d}")

if __name__ == "__main__":
    main()
