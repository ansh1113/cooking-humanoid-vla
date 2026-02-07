"""
Test chapter extraction on Paneer video - CLEAN VERSION
"""
import whisper
import json
import cv2
from openai import OpenAI
import base64
import time
import subprocess
import os
import uuid

# ============================================================================
# PART 1: DOWNLOAD VIDEO
# ============================================================================

def download_youtube_video(url):
    """Download YouTube video"""
    import yt_dlp
    
    output_path = f'video_{uuid.uuid4().hex[:8]}.mp4'
    
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': output_path,
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get('title', 'Unknown')
    
    return output_path, title

# ============================================================================
# PART 2: TRANSCRIBE WITH WHISPER
# ============================================================================

def transcribe_video(video_path):
    """Extract audio and transcribe"""
    print("   🎤 Transcribing with Whisper...")
    
    # Extract audio to unique filename
    audio_path = f'audio_{uuid.uuid4().hex[:8]}.mp3'
    
    cmd = ['ffmpeg', '-y', '-i', video_path, '-ab', '160k', 
           '-ac', '2', '-ar', '44100', '-vn', audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Transcribe
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    
    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    return result

# ============================================================================
# PART 3: DETECT CHAPTERS
# ============================================================================

COOKING_KEYWORDS = {
    'chopping': ['chop', 'dice', 'cut', 'slice', 'mince'],
    'stirring': ['stir', 'mix', 'combine', 'fold'],
    'adding': ['add', 'pour', 'put', 'sprinkle', 'throw'],
    'cooking': ['cook', 'fry', 'sauté', 'saute', 'roast', 'boil', 'simmer'],
    'grinding': ['grind', 'blend', 'paste', 'puree'],
    'marinating': ['marinate', 'coat', 'season'],
    'heating': ['heat', 'warm', 'preheat'],
    'garnishing': ['garnish', 'top', 'serve'],
    'peeling': ['peel', 'skin'],
    'grating': ['grate', 'shred'],
}

INGREDIENTS = ['onion', 'tomato', 'garlic', 'ginger', 'chili', 'pepper',
               'paneer', 'chicken', 'potato', 'carrot', 'spinach', 'masala',
               'spice', 'oil', 'butter', 'ghee', 'curry', 'rice', 'dal',
               'cashew', 'cream', 'milk', 'water', 'salt']

def detect_chapters(transcript):
    """Detect cooking steps from transcript"""
    print("   📖 Detecting chapters...")
    
    chapters = []
    current_chapter = None
    
    for segment in transcript['segments']:
        text = segment['text'].lower()
        start = segment['start']
        
        # Detect action
        detected_action = None
        for action_type, keywords in COOKING_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                detected_action = action_type
                break
        
        # Detect ingredient
        detected_object = None
        for ingredient in INGREDIENTS:
            if ingredient in text:
                detected_object = ingredient
                break
        
        # Start new chapter if action detected
        if detected_action:
            if current_chapter:
                current_chapter['end'] = start
                chapters.append(current_chapter)
            
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
    
    # Filter: keep chapters >= 15 seconds
    filtered = [c for c in chapters if (c['end'] - c['start']) >= 15]
    
    return filtered

# ============================================================================
# PART 4: EXTRACT FRAMES FROM CHAPTERS
# ============================================================================

def extract_chapter_frames(video_path, start_time, end_time, num_frames=8):
    """Extract frames from time window"""
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    total = end_frame - start_frame
    
    if total < num_frames:
        indices = range(start_frame, end_frame)
    else:
        step = total / num_frames
        indices = [int(start_frame + i * step) for i in range(num_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames

# ============================================================================
# PART 5: LABEL WITH GPT-4V
# ============================================================================

def encode_frame(frame):
    """Encode frame to base64"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def label_chapter(client, frames, hint):
    """Label with GPT-4V"""
    images_b64 = [encode_frame(f) for f in frames]
    
    hint_text = ""
    if hint.get('action') and hint.get('object'):
        hint_text = f" Transcript mentions '{hint['action']}' and '{hint['object']}'."
    
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"Analyze these cooking frames.{hint_text}\n\n"
                    "Return JSON with:\n"
                    "1. 'verb': Action (chopping, stirring, adding, grinding)\n"
                    "2. 'noun': SPECIFIC object (onion, paneer, masala - be specific!)\n"
                    "3. 'label': Combined\n"
                    "4. 'tools': List\n"
                    "5. 'container': Container\n"
                    "6. 'confidence': high/medium/low\n\n"
                    "Examples:\n"
                    '{"verb":"chopping","noun":"onion","label":"chopping onion",...}\n'
                    '{"verb":"grinding","noun":"masala","label":"grinding masala",...}'
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

# ============================================================================
# MAIN
# ============================================================================

print("="*70)
print("🧪 TESTING: PANEER BUTTER MASALA")
print("="*70)

url = "https://www.youtube.com/watch?v=oYZ--rdHL6I"

# Step 1: Download
print("\n📥 Downloading...")
video_path, title = download_youtube_video(url)
print(f"   ✅ {title}")

# Step 2: Transcribe
transcript = transcribe_video(video_path)

# Step 3: Detect chapters
chapters = detect_chapters(transcript)

print(f"\n📚 Detected {len(chapters)} chapters (>=15s):")
print("="*70)
for i, ch in enumerate(chapters, 1):
    dur = ch['end'] - ch['start']
    print(f"{i}. [{ch['start']:.0f}s-{ch['end']:.0f}s] ({dur:.0f}s)")
    print(f"   {ch['action']} {ch.get('object', '?')}")
    print(f"   \"{ch['text'][:70]}...\"")
    print()

# Step 4: Ask to proceed
proceed = input("👀 Look good? Label with GPT-4V? (yes/no): ").strip().lower()

if proceed == 'yes':
    api_key = input("🔑 OpenAI API key: ").strip()
    client = OpenAI(api_key=api_key)
    
    print(f"\n🏷️  Labeling {len(chapters)} chapters...")
    print("="*70)
    
    labeled = []
    for i, ch in enumerate(chapters, 1):
        print(f"\nChapter {i}/{len(chapters)}: {ch['action']} {ch.get('object', '?')}")
        
        frames = extract_chapter_frames(video_path, ch['start'], ch['end'])
        
        if len(frames) < 4:
            print("   ⚠️  Too few frames")
            continue
        
        result = label_chapter(client, frames, ch)
        
        if result:
            print(f"   ✅ {result['label']} (conf: {result['confidence']})")
            labeled.append({
                'chapter_index': i,
                'start_time': ch['start'],
                'end_time': ch['end'],
                'transcript_hint': ch['text'],
                **result
            })
        
        time.sleep(1.2)
    
    # Save
    with open('paneer_chapters.json', 'w') as f:
        json.dump(labeled, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✅ Labeled {len(labeled)} chapters")
    print(f"💾 Saved to: paneer_chapters.json")
    
    for ch in labeled:
        print(f"   • {ch['label']} ({ch['start_time']:.0f}s-{ch['end_time']:.0f}s)")

# Cleanup
if os.path.exists(video_path):
    os.remove(video_path)
