"""
GOLDEN 40 LABELING - WITH ERROR HANDLING FOR UNAVAILABLE VIDEOS
"""
import whisper
import json
import cv2
import base64
import subprocess
import os
import yt_dlp
from openai import OpenAI
import time
import random

# THE 40 GOLDEN ACTIONS
GOLDEN_40_ACTIONS = [
    "chopping onion", "chopping vegetables", "slicing tomato", "grinding paste",
    "kneading dough", "washing ingredients", "peeling", "mincing ginger garlic",
    "adding oil", "adding ghee", "adding butter", "tempering spices",
    "frying onions", "sautéing", "adding water", "boiling", "simmering",
    "pressure cooking", "roasting", "grilling", "steaming", "deep frying", "shallow frying",
    "adding tomatoes", "adding masala", "adding ginger garlic paste", "adding chili powder",
    "adding turmeric", "adding coriander", "adding paneer", "adding chicken",
    "adding vegetables", "adding rice",
    "stirring curry", "stirring gently", "mixing thoroughly",
    "garnishing with cream", "garnishing with coriander", "plating", "serving"
]

def download_video(url, filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 10000:
        return filename
        
    print(f"📥 Downloading {url}...")
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
        # Try worst quality as fallback
        try:
            ydl_opts['format'] = 'worst'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except:
            raise Exception(f"Failed to download: {str(e)}")
            
    return filename

def extract_audio(video_path, audio_path):
    if os.path.exists(audio_path):
        os.remove(audio_path)
    cmd = ['ffmpeg', '-y', '-i', video_path, '-ar', '16000', '-ac', '1', audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def get_transcript(audio_path):
    print("🎤 Transcribing...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, task="translate")
    return result

COOKING_KEYWORDS = {
    'chopping': ['chop', 'dice', 'cut', 'slice', 'mince'],
    'stirring': ['stir', 'mix', 'combine', 'fold', 'sauté', 'saute', 'fry'],
    'adding': ['add', 'pour', 'put', 'sprinkle', 'transfer', 'garnish'],
    'cooking': ['cook', 'roast', 'boil', 'simmer', 'heat', 'steam', 'pressure'],
    'grinding': ['grind', 'blend', 'paste', 'crush'],
    'kneading': ['knead', 'dough'],
    'tempering': ['tadka', 'temper', 'crackle'],
}

def detect_chapters(transcript):
    print("📖 Detecting chapters...")
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
                current_chapter = {'start': start, 'end': end, 'action': found_action, 'text': text}
            else:
                current_chapter['end'] = end
                current_chapter['text'] += " " + text

    if current_chapter:
        chapters.append(current_chapter)
        
    return [c for c in chapters if (c['end'] - c['start']) > 4]

def extract_frames(video_path, start, end, num=4):
    cap = cv2.VideoCapture(video_path)
    frames = []
    duration = end - start
    if duration <= 0:
        duration = 1
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

def label_with_constrained_gpt4v(client, frames_b64, context_text):
    actions_list = "\n".join([f"- {action}" for action in GOLDEN_40_ACTIONS])
    
    prompt = f"""You are labeling an Indian cooking video dataset.

AUDIO TRANSCRIPT: "{context_text}"

TASK: Identify the cooking action happening in these frames.

CRITICAL RULE: You MUST select ONE action from this list. Do NOT create new actions:

{actions_list}

Rules:
1. Pick the BEST MATCH from the list above
2. If multiple actions fit, choose the most specific one
3. Use the transcript as a hint, but trust the visual
4. Output ONLY the exact action text from the list

Output JSON:
{{
  "action": "<exact_action_from_list>",
  "confidence": "high" | "medium" | "low"
}}
"""
    
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
                max_tokens=100,
                temperature=0
            )
            result = json.loads(response.choices[0].message.content)
            
            if result['action'] not in GOLDEN_40_ACTIONS:
                print(f"⚠️  Invalid action: {result['action']}, retrying...")
                continue
                
            return result
            
        except Exception as e:
            delay = base_delay * (2 ** attempt) + (random.random() * 1)
            print(f"   Retry {attempt+1}/{max_retries} after {delay:.1f}s...")
            time.sleep(delay)
            
    return None

def is_video_already_processed(url, output_file):
    if not os.path.exists(output_file):
        return False
    
    with open(output_file, 'r') as f:
        existing_data = json.load(f)
    
    for item in existing_data:
        if item.get('video_url') == url:
            return True
    
    return False

def process_video(video_info, client, output_file):
    url = video_info['url']
    title = video_info['title']
    
    if is_video_already_processed(url, output_file):
        print(f"\n⏭️  SKIPPING: {title} (already processed)")
        return
    
    print(f"\n{'='*70}")
    print(f"🎬 {title}")
    print(f"{'='*70}")
    
    vid_file = f"temp_golden_{url.split('=')[-1]}.mp4"
    aud_file = f"temp_audio_{url.split('=')[-1]}.mp3"
    
    try:
        download_video(url, vid_file)
        extract_audio(vid_file, aud_file)
        transcript = get_transcript(aud_file)
        chapters = detect_chapters(transcript)
        
        print(f"📚 Found {len(chapters)} chapters")
        
        labeled_data = []
        
        for i, ch in enumerate(chapters, 1):
            print(f"   Chapter {i}/{len(chapters)}: [{ch['start']:.0f}s-{ch['end']:.0f}s]", end="")
            
            frames = extract_frames(vid_file, ch['start'], ch['end'])
            if len(frames) < 3:
                print(" ❌ Too few frames")
                continue
            
            label = label_with_constrained_gpt4v(client, frames, ch['text'])
            
            if label:
                print(f" ✅ {label['action']}")
                
                labeled_data.append({
                    'video_url': url,
                    'video_title': title,
                    'start': ch['start'],
                    'end': ch['end'],
                    'action': label['action'],
                    'confidence': label['confidence'],
                    'transcript': ch['text']
                })
            else:
                print(" ❌ Labeling failed")
        
        existing_data = []
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
        
        existing_data.extend(labeled_data)
        
        with open(output_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        print(f"💾 Saved {len(labeled_data)} labels")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print(f"⏭️  Skipping this video...")
        
    finally:
        if os.path.exists(vid_file):
            os.remove(vid_file)
        if os.path.exists(aud_file):
            os.remove(aud_file)

def main():
    print("="*70)
    print("🌟 GOLDEN 40 DATASET CREATION - CONSTRAINED LABELING")
    print("="*70)
    
    with open('golden_40_indian_videos.json') as f:
        videos_by_category = json.load(f)
    
    all_videos = []
    for category, videos in videos_by_category.items():
        for v in videos:
            v['category'] = category
            all_videos.append(v)
    
    print(f"📊 Total videos: {len(all_videos)}")
    print(f"📚 Target actions: {len(GOLDEN_40_ACTIONS)}")
    
    api_key = input("\n🔑 OpenAI API Key: ").strip()
    client = OpenAI(api_key=api_key)
    
    output_file = 'data/processed/golden_40_dataset.json'
    
    already_done = sum(1 for v in all_videos if is_video_already_processed(v['url'], output_file))
    
    print(f"\n✅ Already processed: {already_done}/{len(all_videos)}")
    print(f"📝 Remaining: {len(all_videos) - already_done}")
    
    print(f"\n{'='*70}")
    print("🚀 STARTING BATCH PROCESSING")
    print("="*70)
    
    for i, video in enumerate(all_videos, 1):
        print(f"\nVideo {i}/{len(all_videos)}")
        process_video(video, client, output_file)
    
    print(f"\n{'='*70}")
    print("✅ GOLDEN 40 DATASET COMPLETE!")
    print(f"{'='*70}")
    
    with open(output_file) as f:
        final_data = json.load(f)
    
    from collections import Counter
    action_counts = Counter(d['action'] for d in final_data)
    
    print(f"\nTotal labels: {len(final_data)}")
    print(f"Unique actions: {len(action_counts)}")
    print(f"\nDistribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:3d}x {action}")

if __name__ == "__main__":
    main()
