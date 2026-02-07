"""
Manually test GPT-4V on ONE video to debug
"""
import json
from openai import OpenAI
import base64
import cv2
from stage1_generate_plan import download_youtube_video, extract_frames

def encode_image_cv2(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# Test with Butter Chicken video
url = "https://www.youtube.com/watch?v=a03U45jFxOI"

print("Downloading Butter Chicken video...")
result = download_youtube_video(url)
if isinstance(result, tuple):
    video_path, title = result
else:
    video_path = result

print(f"Extracting frames from: {title}")
frames = extract_frames(video_path, max_frames=60)

# Sample 8 frames
n = len(frames)
indices = [0, n//7, 2*n//7, 3*n//7, 4*n//7, 5*n//7, 6*n//7, n-1]
sampled = [frames[i] for i in indices]

print(f"Extracted {len(sampled)} frames")

# Save frames to disk to verify
import os
os.makedirs('debug_frames', exist_ok=True)
for i, frame in enumerate(sampled):
    cv2.imwrite(f'debug_frames/frame_{i}.jpg', frame)
print("Saved frames to debug_frames/ - CHECK THESE MANUALLY!")

# Now test GPT-4V
api_key = input("\nEnter API key to test GPT-4V: ").strip()
client = OpenAI(api_key=api_key)

images_b64 = [encode_image_cv2(f) for f in sampled]

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

print("\nCalling GPT-4V...")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    max_tokens=150,
    temperature=0,
    response_format={"type": "json_object"}
)

result = json.loads(response.choices[0].message.content)

print("\n" + "="*70)
print("GPT-4V RESPONSE:")
print("="*70)
print(json.dumps(result, indent=2))
