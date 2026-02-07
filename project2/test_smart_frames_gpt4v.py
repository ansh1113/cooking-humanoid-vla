"""
Test GPT-4V with SMART sampled frames
"""
import json
from openai import OpenAI
import base64
import cv2

def encode_image_cv2(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# Load the smart frames
frames = []
for i in range(8):
    frame = cv2.imread(f'debug_frames_smart/frame_{i}.jpg')
    frames.append(frame)

api_key = input("Enter API key: ").strip()
client = OpenAI(api_key=api_key)

images_b64 = [encode_image_cv2(f) for f in frames]

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

print("\nCalling GPT-4V with SMART frames...")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    max_tokens=150,
    temperature=0,
    response_format={"type": "json_object"}
)

result = json.loads(response.choices[0].message.content)

print("\n" + "="*70)
print("GPT-4V RESPONSE (SMART FRAMES):")
print("="*70)
print(json.dumps(result, indent=2))
print("\nExpected: cooking/stirring chicken or similar")
print("This is Butter Chicken recipe - should see chicken, curry, spices")
