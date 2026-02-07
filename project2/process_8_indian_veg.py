"""
UNIVERSAL COOKING DATASET GENERATOR (PLATINUM EDITION)
- Auto-Translates Hindi/Regional Audio -> English
- Open-Ended Ingredient Discovery
- CRASH-PROOF: Incremental Saving
- MEMORY-SAFE: Aggressive GC for low-RAM clusters
- LOGGING: Dual outputs (Console + File)
"""
import whisper
import json
import cv2
import base64
import time
import subprocess
import os
import yt_dlp
import logging
import gc
import random
from openai import OpenAI

# ============================================================================
# 0. SETUP LOGGING
# ============================================================================
OUTPUT_FILE = 'data/processed/indian_veg_universal_labels.json'
LOG_FILE = 'batch_processing.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ============================================================================
# 1. ROBUST DOWNLOADER
# ============================================================================

def download_fresh_video(url, output_name):
    if os.path.exists(output_name):
        os.remove(output_name)
        
    logger.info(f"📥 Downloading {output_name}...")
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_name,
        'quiet': True,
        'overwrites': True,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}}
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except:
        logger.warning("Primary download failed. Trying fallback...")
        ydl_opts['format'] = 'worst'
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
    if not os.path.exists(output_name) or os.path.getsize(output_name) < 1000:
        raise ValueError("Download failed: File empty")
    return output_name

def extract_audio(video_path, audio_path):
    if os.path.exists(audio_path):
        os.remove(audio_path)
    # Extract audio at 16k rate (optimized for Whisper)
    cmd = ['ffmpeg', '-y', '-i', video_path, '-ar', '16000', '-ac', '1', audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

# ============================================================================
# 2. UNIVERSAL TRANSCRIPTION & DETECTION
# ============================================================================

def get_transcript(audio_path):
    logger.info("🎤 Transcribing (Auto-Translate to English)...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, task="translate") 
    return result

COOKING_KEYWORDS = {
    'chopping': ['chop', 'dice', 'cut', 'slice', 'mince'],
    'stirring': ['stir', 'mix', 'combine', 'fold', 'sauté', 'saute', 'fry'],
    'adding': ['add', 'pour', 'put', 'sprinkle', 'transfer', 'garnish'],
    'cooking': ['cook', 'roast', 'boil', 'simmer', 'heat', 'steam', 'pressure'],
    'grinding': ['grind', 'blend', 'paste', 'crush'],
}

def detect_chapters(transcript):
    logger.info("📖 Detecting cooking actions...")
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
                    'text': text
                }
            else:
                current_chapter['end'] = end
                current_chapter['text'] += " " + text

    if current_chapter:
        chapters.append(current_chapter)
        
    return [c for c in chapters if (c['end'] - c['start']) > 4]

# ============================================================================
# 3. ROBUST GPT-4V LABELING (EXPONENTIAL BACKOFF)
# ============================================================================

def extract_frames(video_path, start, end, num=4):
    cap = cv2.VideoCapture(video_path)
    frames = []
    duration = end - start
    if duration <= 0:
        duration = 1
    step = duration / num
    
    try:
        for i in range(num):
            t = start + (i * step)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                b64 = base64.b64encode(buffer).decode('utf-8')
                frames.append(b64)
    finally:
        cap.release()
        
    return frames

def label_with_gpt4v(client, frames_b64, context_text):
    prompt = f"""
    You are labeling a cooking dataset.
    Audio Transcript: "{context_text}"
    Task: Identify the specific ingredient being manipulated.
    Rules:
    1. IGNORE TOOLS: Output the FOOD (e.g., 'adding sugar', not 'adding spoon').
    2. BE SPECIFIC: Use specific words like 'ghee', 'paneer', 'chillies'.
    3. VISUAL PRIORITY: Trust the image over the text.
    Output JSON: {{ "verb": "...", "noun": "...", "confidence": "..." }}
    """
    
    # EXPONENTIAL BACKOFF RETRY LOGIC
    max_retries = 5
    base_delay = 2
    
    for attempt in range(max_retries):
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
            delay = base_delay * (2 ** attempt) + (random.random() * 1)
            logger.warning(f"GPT Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
            
    logger.error("❌ GPT-4V Failed after all retries.")
    return None

# ============================================================================
# MAIN BATCH PROCESSOR
# ============================================================================

def main():
    print("="*70)
    print("🍛 PROCESSING 8 INDIAN VEGETARIAN VIDEOS (PLATINUM EDITION)")
    print("="*70)
    
    # 1. Load Data Source
    try:
        with open('data/curated_diverse_videos_clean.json') as f:
            curated = json.load(f)
        video_list = curated.get('indian_vegetarian', [])
    except:
        logger.error("❌ Could not load curated_diverse_videos_clean.json")
        return
    
    # 2. Check for Existing Progress (Resume Logic)
    all_labeled_data = []
    processed_urls = set()
    
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                all_labeled_data = json.load(f)
                processed_urls = {item['video_url'] for item in all_labeled_data}
            logger.info(f"🔄 Resuming... Found {len(all_labeled_data)} existing labels from {len(processed_urls)} videos.")
        except:
            logger.warning("Output file corrupt or empty. Starting fresh.")

    # 3. Start Processing
    logger.info(f"Total videos queue: {len(video_list)}")
    logger.info(f"Skipping {len(processed_urls)} already processed.")
    
    videos_to_do = [v for v in video_list if v['url'] not in processed_urls]
    
    if not videos_to_do:
        logger.info("✅ All videos already processed!")
        return

    api_key = input("\n🔑 OpenAI API key: ").strip()
    client = OpenAI(api_key=api_key)
    
    for i, video in enumerate(videos_to_do, 1):
        logger.info(f"\n🎥 Video {i}/{len(videos_to_do)}: {video['title'][:40]}...")
        
        vid_file = f"temp_vid_{i}.mp4"
        aud_file = f"temp_aud_{i}.mp3"
        
        try:
            download_fresh_video(video['url'], vid_file)
            extract_audio(vid_file, aud_file)
            
            transcript = get_transcript(aud_file)
            chapters = detect_chapters(transcript)
            logger.info(f"📚 Detected {len(chapters)} potential actions")
            
            # Label loop
            new_labels_count = 0
            for ch in chapters:
                frames = extract_frames(vid_file, ch['start'], ch['end'])
                if len(frames) < 3:
                    continue
                
                label = label_with_gpt4v(client, frames, ch['text'])
                
                if label:
                    print(f"      ✅ [{ch['start']:.0f}s] {label['verb']} {label['noun']}")
                    all_labeled_data.append({
                        'video_url': video['url'],
                        'video_title': video['title'],
                        'start': ch['start'],
                        'end': ch['end'],
                        'label': f"{label['verb']} {label['noun']}",
                        'transcript': ch['text']
                    })
                    new_labels_count += 1
                
                # MEMORY MANAGEMENT
                del frames
                del label
                
            # Manual GC to keep RAM usage low
            gc.collect() 
            
            # INCREMENTAL SAVE
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(all_labeled_data, f, indent=2)
            logger.info(f"💾 Saved {new_labels_count} labels to disk.")
                
        except Exception as e:
            logger.error(f"❌ Failed processing video: {e}")
        
        finally:
            if os.path.exists(vid_file):
                os.remove(vid_file)
            if os.path.exists(aud_file):
                os.remove(aud_file)

    logger.info("="*70)
    logger.info(f"🎉 BATCH COMPLETE! Total labels: {len(all_labeled_data)}")
    logger.info(f"📂 File: {OUTPUT_FILE}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
