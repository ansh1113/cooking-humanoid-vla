"""
Test chapter extraction on ONE video
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
    
    # Merge very short chapters (< 10 seconds)
    merged = []
    i = 0
    while i < len(chapters):
        chapter = chapters[i]
        duration = chapter['end'] - chapter['start']
        
        if duration < 10 and i < len(chapters) - 1:
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
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def label_chapter_with_gpt4v(client, frames, chapter_hint):
    """Label chapter with GPT-4V"""
    images_b64 = [encode_image_cv2(f) for f in frames]
    
    hint_text = ""
    if chapter_hint.get('action') and chapter_hint.get('object'):
        hint_text = f" The transcript mentions '{chapter_hint['action']}' and '{chapter_hint['object']}'."
    
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"Analyze these {len(frames)} frames from a cooking video chapter.{hint_text}\n\n"
                    "Identify the MAIN cooking action.\n\n"
                    "Return ONLY a JSON object with:\n"
                    "1. 'verb': Action (e.g., 'chopping', 'stirring', 'grinding')\n"
                    "2. 'noun': SPECIFIC object (e.g., 'onion', 'garlic', 'paneer' - be specific!)\n"
                    "3. 'label': Combined (e.g., 'chopping onion')\n"
                    "4. 'tools': List of tools\n"
                    "5. 'container': Container\n"
                    "6. 'confidence': 'high', 'medium', or 'low'\n\n"
                    "Be SPECIFIC - don't say 'vegetable', identify the actual item!\n\n"
                    "Examples:\n"
                    '{"verb": "chopping", "noun": "onion", "label": "chopping onion", ...}\n'
                    '{"verb": "grinding", "noun": "masala", "label": "grinding masala", ...}'
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

# TEST ON BUTTER CHICKEN VIDEO
video_url = "https://www.youtube.com/watch?v=oYZ--rdHL6I"

print("="*70)
print("🧪 TESTING CHAPTER EXTRACTION ON ONE VIDEO")
print("="*70)
print(f"Video: {video_url}")

# Download
print("\n📥 Downloading...")
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

print(f"\n📚 Detected {len(chapters)} chapters:")
print("="*70)
for i, chapter in enumerate(chapters, 1):
    duration = chapter['end'] - chapter['start']
    print(f"{i}. [{chapter['start']:.0f}s - {chapter['end']:.0f}s] ({duration:.0f}s)")
    print(f"   Action: {chapter['action']}")
    print(f"   Object: {chapter.get('object', 'unknown')}")
    print(f"   Text: \"{chapter['text'][:80]}...\"")
    print()

# Ask user if they want to label
print("\n" + "="*70)
proceed = input(f"\n👀 Do chapters look good? Label with GPT-4V? (yes/no): ").strip().lower()

if proceed == 'yes':
    api_key = input("🔑 OpenAI API key: ").strip()
    client = OpenAI(api_key=api_key)
    
    print(f"\n🏷️  Labeling {len(chapters)} chapters...")
    print("="*70)
    
    labeled = []
    for i, chapter in enumerate(chapters, 1):
        print(f"\nChapter {i}/{len(chapters)}: {chapter['action']} {chapter.get('object', '?')}")
        
        frames = extract_chapter_frames(video_path, chapter, num_frames=8)
        
        if len(frames) < 4:
            print(f"   ⚠️  Too few frames, skipping")
            continue
        
        label_data = label_chapter_with_gpt4v(client, frames, chapter)
        
        if label_data:
            print(f"   ✅ GPT-4V: {label_data['label']} (confidence: {label_data['confidence']})")
            labeled.append({
                'chapter_index': i,
                'start_time': chapter['start'],
                'end_time': chapter['end'],
                'duration': chapter['end'] - chapter['start'],
                'transcript_hint': chapter['text'],
                **label_data
            })
        
        time.sleep(1.2)
    
    # Save
    with open('test_butter_chicken_chapters.json', 'w') as f:
        json.dump(labeled, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✅ Labeled {len(labeled)} chapters")
    print(f"💾 Saved to: test_butter_chicken_chapters.json")
    
    print(f"\n📊 Results:")
    for ch in labeled:
        print(f"   • {ch['label']} ({ch['start_time']:.0f}s-{ch['end_time']:.0f}s)")
