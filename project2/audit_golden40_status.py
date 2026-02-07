"""
Complete audit of Golden 40 dataset status
"""
import json
import os
import whisper
import subprocess
from collections import Counter, defaultdict

print("="*70)
print("📊 GOLDEN 40 DATASET AUDIT")
print("="*70)

# ===========================================================================
# STEP 1: CHECK DOWNLOADED VIDEOS
# ===========================================================================
print("\n🎬 STEP 1: CHECKING DOWNLOADED VIDEOS")
print("-"*70)

downloaded_videos = []
for fname in os.listdir('.'):
    if fname.endswith('.mp4') and 'temp_golden' in fname:
        size_mb = os.path.getsize(fname) / (1024*1024)
        video_id = fname.replace('temp_golden_', '').replace('.mp4', '')
        downloaded_videos.append({
            'filename': fname,
            'video_id': video_id,
            'size_mb': size_mb
        })

if os.path.exists('videos'):
    for fname in os.listdir('videos'):
        if fname.endswith('.mp4'):
            size_mb = os.path.getsize(os.path.join('videos', fname)) / (1024*1024)
            video_id = fname.replace('.mp4', '')
            downloaded_videos.append({
                'filename': os.path.join('videos', fname),
                'video_id': video_id,
                'size_mb': size_mb
            })

print(f"✅ Found {len(downloaded_videos)} downloaded videos")
print(f"   Total size: {sum(v['size_mb'] for v in downloaded_videos):.1f} MB")

# ===========================================================================
# STEP 2: CHECK ALREADY LABELED DATA
# ===========================================================================
print(f"\n📋 STEP 2: CHECKING ALREADY LABELED DATA")
print("-"*70)

labeled_data = []
if os.path.exists('data/processed/golden_40_dataset.json'):
    with open('data/processed/golden_40_dataset.json') as f:
        labeled_data = json.load(f)

print(f"✅ Found {len(labeled_data)} labeled segments")

if len(labeled_data) > 0:
    # Count by video (handle missing keys)
    videos_labeled = defaultdict(int)
    for item in labeled_data:
        title = item.get('video_title', item.get('title', 'Unknown'))
        videos_labeled[title] += 1
    
    print(f"   From {len(videos_labeled)} unique videos")
    print(f"\n   Videos with labels:")
    for video, count in sorted(videos_labeled.items(), key=lambda x: -x[1])[:15]:
        print(f"   - {video:50s}: {count:2d} labels")
    
    # Count by action
    action_counts = Counter(d.get('action', 'unknown') for d in labeled_data)
    print(f"\n   Action distribution (top 20):")
    for action, count in action_counts.most_common(20):
        print(f"   - {action:35s}: {count:3d}")
    
    # Show URLs that are already done
    labeled_urls = set(d.get('video_url', '') for d in labeled_data if d.get('video_url'))
    print(f"\n   ✅ {len(labeled_urls)} unique video URLs processed")

# ===========================================================================
# STEP 3: MATCH DOWNLOADED VIDEOS WITH LABELED DATA
# ===========================================================================
print(f"\n🔗 STEP 3: MATCHING DOWNLOADED VIDEOS WITH LABELS")
print("-"*70)

# Map video IDs to labels
labeled_video_ids = set()
for item in labeled_data:
    url = item.get('video_url', '')
    if url:
        video_id = url.split('=')[-1]
        labeled_video_ids.add(video_id)

downloaded_ids = {v['video_id'] for v in downloaded_videos}

already_labeled = downloaded_ids & labeled_video_ids
not_yet_labeled = downloaded_ids - labeled_video_ids

print(f"✅ Downloaded videos already labeled: {len(already_labeled)}")
print(f"❓ Downloaded videos NOT yet labeled: {len(not_yet_labeled)}")

if len(not_yet_labeled) > 0:
    print(f"\n   Videos ready to label:")
    for vid in list(not_yet_labeled)[:10]:
        matching = [v for v in downloaded_videos if v['video_id'] == vid]
        if matching:
            print(f"   - {matching[0]['filename']:50s} ({matching[0]['size_mb']:5.1f} MB)")
    if len(not_yet_labeled) > 10:
        print(f"   ... and {len(not_yet_labeled)-10} more")

# ===========================================================================
# STEP 4: QUICK ELIGIBILITY CHECK
# ===========================================================================
print(f"\n🔍 STEP 4: CHECKING ELIGIBILITY (SAMPLE)")
print("-"*70)

def extract_audio_temp(video_path):
    audio_path = video_path.replace('.mp4', '_temp.mp3')
    if os.path.exists(audio_path):
        return audio_path
    cmd = ['ffmpeg', '-y', '-i', video_path, '-ar', '16000', '-ac', '1', audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def quick_chapter_check(video_path):
    try:
        audio_path = extract_audio_temp(video_path)
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, task="translate")
        
        COOKING_KEYWORDS = ['chop', 'dice', 'cut', 'slice', 'stir', 'mix', 
                           'add', 'pour', 'cook', 'fry', 'boil', 'grind', 'knead']
        
        keyword_count = 0
        for segment in result['segments']:
            text = segment['text'].lower()
            if any(kw in text for kw in COOKING_KEYWORDS):
                keyword_count += 1
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return {
            'eligible': keyword_count >= 3,
            'keyword_count': keyword_count,
            'duration': result.get('duration', 0)
        }
    except Exception as e:
        return {'eligible': False, 'error': str(e)}

# Test up to 5 unlabeled videos
test_videos = [v for v in downloaded_videos if v['video_id'] in not_yet_labeled][:5]

print(f"Testing {len(test_videos)} unlabeled videos for eligibility...")
print("(This takes ~1 min per video)\n")

eligible_count = 0
ineligible_count = 0

for i, video in enumerate(test_videos, 1):
    print(f"   [{i}/{len(test_videos)}] {video['filename'][:45]}...", end=" ")
    result = quick_chapter_check(video['filename'])
    
    if result.get('eligible'):
        print(f"✅ ELIGIBLE ({result['keyword_count']} actions)")
        eligible_count += 1
    else:
        error = result.get('error', f"Only {result.get('keyword_count', 0)} actions")
        print(f"❌ SKIP ({error[:40]})")
        ineligible_count += 1

# ===========================================================================
# STEP 5: GOLDEN 40 LIST STATUS
# ===========================================================================
print(f"\n📚 STEP 5: GOLDEN 40 LIST STATUS")
print("-"*70)

golden_40_videos = []
if os.path.exists('golden_40_indian_videos.json'):
    with open('golden_40_indian_videos.json') as f:
        data = json.load(f)
        for category, videos in data.items():
            for v in videos:
                v['category'] = category
                golden_40_videos.append(v)

print(f"📋 Golden 40 list: {len(golden_40_videos)} videos")

processed = []
not_processed = []

for video in golden_40_videos:
    video_id = video['url'].split('=')[-1]
    if video_id in labeled_video_ids:
        processed.append(video)
    else:
        not_processed.append(video)

print(f"   ✅ Processed: {len(processed)}")
print(f"   ❌ Remaining: {len(not_processed)}")

# ===========================================================================
# SUMMARY
# ===========================================================================
print(f"\n{'='*70}")
print("📊 SUMMARY")
print("="*70)

TARGET = 300

print(f"\n✅ Current dataset:")
print(f"   Labels: {len(labeled_data)}")
print(f"   Videos: {len(labeled_video_ids)}")
print(f"   Actions: {len(set(d.get('action', 'unknown') for d in labeled_data))}")

print(f"\n📥 Downloads:")
print(f"   Total downloaded: {len(downloaded_videos)}")
print(f"   Already labeled: {len(already_labeled)}")
print(f"   Ready to label: {len(not_yet_labeled)}")
print(f"   Eligible (sample): {eligible_count}/{len(test_videos)} tested")

print(f"\n🎯 Progress to target ({TARGET} labels):")
current = len(labeled_data)
remaining = max(0, TARGET - current)
progress = min(100, (current / TARGET) * 100)
print(f"   Current: {current}/{TARGET} ({progress:.1f}%)")
print(f"   Needed: {remaining} more labels")

if remaining > 0:
    est_videos = remaining // 8
    print(f"   Est. videos needed: ~{est_videos} (at 8 labels/video)")
    print(f"   Available unlabeled: {len(not_yet_labeled)} downloaded")
    print(f"   Available in Golden 40: {len(not_processed)}")

print("\n" + "="*70)
print("🎯 NEXT STEPS:")
if remaining <= 0:
    print("   🎉 TARGET REACHED! You have enough data!")
elif len(not_yet_labeled) >= est_videos:
    print(f"   1. Label the {len(not_yet_labeled)} downloaded unlabeled videos")
    print(f"   2. This should get you to {current + len(not_yet_labeled)*8}+ labels")
else:
    print(f"   1. Label {len(not_yet_labeled)} downloaded videos")
    print(f"   2. Download & label {est_videos - len(not_yet_labeled)} more from Golden 40")
print("="*70)

# Save summary
summary = {
    'current_labels': len(labeled_data),
    'target_labels': TARGET,
    'progress_percent': progress,
    'downloaded_videos': len(downloaded_videos),
    'videos_ready_to_label': len(not_yet_labeled),
    'eligible_sample': f"{eligible_count}/{len(test_videos)}",
    'golden40_remaining': len(not_processed)
}

with open('data/processed/audit_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n💾 Summary saved to: data/processed/audit_summary.json")
