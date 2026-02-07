"""
PART 2: LABEL & EXTRACT FEATURES
Run on GPU via sbatch (~2-3 hours)
Labels with GPT-4V, extracts CLIP features
"""
import torch
import json
import cv2
import os
import open_clip
from PIL import Image
import whisper
import base64
from openai import OpenAI
import time
import random

print("="*70)
print("🏷️  AUTOMATED LABELING & FEATURE EXTRACTION")
print("="*70)

# Load curated videos
with open('curated_good_videos.json') as f:
    videos = json.load(f)

print(f"✅ Found {len(videos)} curated videos")

# Load existing dataset
if os.path.exists('data/processed/golden_40_dataset.json'):
    with open('data/processed/golden_40_dataset.json') as f:
        existing_data = json.load(f)
    print(f"📊 Existing labels: {len(existing_data)}")
else:
    existing_data = []

# Add new videos to the labeling list
new_video_list = []
for v in videos:
    new_video_list.append({
        'url': v['url'],
        'title': v['title'],
        'category': 'auto_curated'
    })

# Save combined list
with open('golden_40_indian_videos.json', 'w') as f:
    json.dump({'auto_curated': new_video_list}, f, indent=2)

print(f"✅ Created labeling queue: {len(new_video_list)} videos")

# Now run the EXISTING labeling script
print("\n🚀 Running labeling script...")
print("="*70)

# Import and run existing labeling function
exec(open('label_golden_40_constrained.py').read())
