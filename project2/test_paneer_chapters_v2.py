"""
FIXED PANEER CHAPTER TESTER (V3)
- Expanded Ingredient List (Chillies, Ghee, Sugar, etc.)
- "Open-Minded" GPT-4V Prompt (No fixed list of nouns)
- Tool Suppression (Ignore 'spoon', focus on 'sugar')
"""
import whisper
import json
import cv2
import base64
import time
import subprocess
import os
import yt_dlp
import uuid

# ============================================================================
# 1. ROBUST DOWNLOADER
# ============================================================================

def download_fresh_video(url, filename="paneer_video.mp4"):
    if os.path.exists(filename):
        # Only delete if it looks corrupted (small)
        if os.path.getsize(filename) < 10000:
            os.remove(filename)
        else:
            print(f"✅ Using existing {filename}")
            return filename
        
    print(f"📥 Downloading fresh video from {url}...")
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': filename,
        'quiet': True,
        'overwrites': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}}
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        ydl_opts['format'] = 'worst'
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
    return filename

def extract_audio(video_path, audio_path="paneer_audio.mp3"):
    if os.path.exists(audio_path):
        os.remove(audio_path)
    cmd = ['ffmpeg', '-y', '-i', video_path, '-ab', '160k', '-ac', '2', '-ar', '44100', '-vn', audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

# ============================================================================
# 2. TRANSCRIBE & DETECT
# ============================================================================

def get_transcript(audio_path):
    print("🎤 Transcribing with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result

COOKING_KEYWORDS = {
    'chopping': ['chop', 'dice', 'cut', 'slice', 'mince'],
    'stirring': ['stir', 'mix', 'combine', 'fold', 'sauté', 'saute'],
    'adding': ['add', 'pour', 'put', 'sprinkle', 'transfer'],
    'cooking': ['cook', 'fry', 'roast', 'boil', 'simmer', 'heat'],
    'grinding': ['grind', 'blend', 'paste'],
}

# EXPANDED LIST based on Indian Cooking
INGREDIENTS = [
    'onion', 'tomato', 'garlic', 'ginger', 'chili', 'chillies', 'paneer', 'masala', 
    'oil', 'butter', 'ghee', 'salt', 'turmeric', 'powder', 'coriander', 
    'cumin', 'curry', 'cashew', 'cream', 'water', 'sugar', 'milk', 'leaves'
]

def detect_chapters(transcript):
    print("📖 Detecting cooking steps...")
    chapters = []
    current_chapter = None
    
    for segment in transcript['segments']:
        text = segment['text'].lower()
        start = segment['start']
        end = segment['end']
        
        found_action = None
        for action, kws in COOKING_KEYWORDS.items():
            if any(kw in text for kw in kws):
                found_action = action
                break
        
        found_obj = None
        for ing in INGREDIENTS:
            if ing in text:
                found_obj = ing
                break
        
        if found_action:
            if current_chapter:
                if found_action != current_chapter['action'] or (start - current_chapter['end'] > 8):
                    chapters.append(current_chapter)
                    current_chapter = None
            
            if not current_chapter:
                current_chapter = {
                    'start': start,
                    'end': end,
                    'action': found_action,
                    'object': found_obj, 
                    'text': text
                }
            else:
                current_chapter['end'] = end
                current_chapter['text'] += " " + text
                if not current_chapter['object'] and found_obj:
                    current_chapter['object'] = found_obj

    if current_chapter:
        chapters.append(current_chapter)
        
    return [c for c in chapters if (c['end'] - c['start']) > 3]

# ============================================================================
# 3. GPT-4V LABELING (THE FIX)
# ============================================================================

def extract_frames(video_path, start, end, num=4):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    duration = end - start
    if duration <= 0: duration = 1
    step = duration / num
    
    for i in range(num):
        t = start + (i * step)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(buffer).decode('utf-8')
            frames.append(b64)
    cap.release()
    return frames

def label_with_gpt4v(client, frames_b64, context_text):
    # THIS PROMPT IS THE FIX
    prompt = f"""
    You are a professional chef dataset labeler.
    
    AUDIO TRANSCRIPT: "{context_text}"
    
    Task: Identify the action and the SPECIFIC INGREDIENT.
    
    Rules:
    1. NO TOOLS: Do not say 'adding spoon' or 'adding bowl'. Tell me WHAT is in the spoon (e.g., 'adding sugar', 'adding spice').
    2. SPECIFICITY: If the audio mentions 'ghee', 'chillies', or 'sugar', use those exact words. Do not just say 'adding food'.
    3. VISUAL CHECK: Use the images to confirm. If audio says 'add chillies' and you see red things, output 'adding chillies'.
    
    Output JSON only:
    {{
        "verb": "chopping" | "stirring" | "pouring" | "frying" | "adding",
        "noun": string (e.g. "onion", "ghee", "sugar", "paneer"),
        "confidence": "high" | "low"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in frames_b64]
                ]}
            ],
            response_format={"type": "json_object"},
            max_tokens=100
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"GPT Error: {e}")
        return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🧪 FIXED PANEER CHAPTER TESTER (V3 - INTELLIGENT LABELS)")
    print("="*70)
    
    url = "https://www.youtube.com/watch?v=oYZ--rdHL6I"
    video_file = download_fresh_video(url, "paneer_video.mp4")
    audio_file = extract_audio(video_file, "paneer_audio.mp3")
    transcript = get_transcript(audio_file)
    chapters = detect_chapters(transcript)
    
    print(f"\n📚 Detected {len(chapters)} Potential Steps:")
    valid_chapters = []
    
    for i, ch in enumerate(chapters, 1):
        duration = ch['end'] - ch['start']
        if duration > 4: 
            print(f"{i}. [{ch['start']:.0f}s-{ch['end']:.0f}s] {ch['action']} {ch['object'] or '?'}")
            print(f"   \"{ch['text'][:60]}...\"")
            valid_chapters.append(ch)
            
    print("\n" + "-"*70)
    choice = input("👀 Verify with GPT-4V? (yes/no): ").strip().lower()
    
    if choice == 'yes':
        from openai import OpenAI
        api_key = input("🔑 Enter OpenAI API Key: ").strip()
        client = OpenAI(api_key=api_key)
        
        print("\n🏷️  Labeling with GPT-4V...")
        final_dataset = []
        
        for i, ch in enumerate(valid_chapters):
            print(f"   Processing Step {i+1}/{len(valid_chapters)}...", end="\r")
            frames = extract_frames(video_file, ch['start'], ch['end'])
            if not frames: continue
            
            label = label_with_gpt4v(client, frames, ch['text'])
            
            if label:
                # Print nicely formatted
                print(f"   ✅ [{ch['start']:.0f}s] {label['verb']} {label['noun']} ({label['confidence']})")
                
                final_dataset.append({
                    'start': ch['start'],
                    'end': ch['end'],
                    'label': f"{label['verb']} {label['noun']}",
                    'video_id': 'oYZ--rdHL6I'
                })
                
        print(f"\n💾 Saving {len(final_dataset)} labeled clips to paneer_labels.json")
        with open('paneer_labels.json', 'w') as f:
            json.dump(final_dataset, f, indent=2)

if __name__ == "__main__":
    main()
