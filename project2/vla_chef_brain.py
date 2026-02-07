"""
🇮🇳 ROBOTIC SOUS-CHEF: THE VLA BRAIN (FIXED)
===========================================
A Multimodal Action Planner for Indian Cooking.
Integrates:
  1. Whisper (Audio Stream) -> Intent Recognition
  2. CLIP + Temporal Transformer (Video Stream) -> Motion Verification
  3. Logic Core -> Robot Action Planning
"""
import torch
import json
import cv2
import open_clip
from PIL import Image
import numpy as np
import os
import sys
from phase2_temporal_vla_model import TemporalVLA
from process_8_indian_veg import download_fresh_video, extract_audio, get_transcript, detect_chapters

# --- CONFIGURATION ---
MODEL_PATH = 'models/temporal_vla_phase2_best.pt' # The Robust Western Model
VOCAB_PATH = 'data/processed/action_vocab_phase2_fixed.json'

class VLABrain:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🧠 Initializing VLA Brain on {self.device}...")
        
        # Load Vision Model
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        with open(VOCAB_PATH) as f:
            self.vocab = json.load(f)
            
        # --- THE FIX: Create Inverse Vocabulary (Index -> Name) ---
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        # --------------------------------------------------------
        
        self.vision_model = TemporalVLA(512, 512, len(self.vocab), 8, 6).to(self.device)
        self.vision_model.load_state_dict(checkpoint['model_state_dict'])
        self.vision_model.eval()
        
        # Load CLIP
        print("👁️  Loading Vision Encoder (CLIP)...")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device)
        
    def analyze_video_segment(self, video_path, start, end):
        """Returns the raw visual primitive detected in the segment."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        indices = np.linspace(start, end, 30)
        for t in indices:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.preprocess(Image.fromarray(frame)))
        cap.release()
        
        if len(frames) < 30: return "UNKNOWN"
        
        img_tensor = torch.stack(frames).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.encode_image(img_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = self.vision_model(feats.unsqueeze(0))
            pred_idx = torch.argmax(logits, dim=1).item()
            
        # Map Western labels to Physics Primitives
        # --- THE FIX: Use inv_vocab ---
        label = self.inv_vocab[pred_idx] 
        # ------------------------------
        
        if 'chop' in label or 'slice' in label: return 'CUT'
        if 'stir' in label or 'mix' in label: return 'MIX'
        if 'pour' in label or 'add' in label: return 'ADD'
        if 'cook' in label or 'fry' in label: return 'COOK'
        return 'MIX' # Default for ambiguous circular motion

    def get_audio_intent(self, text):
        t = text.lower()
        if any(x in t for x in ['cut', 'chop', 'slice', 'dice']): return 'CUT'
        if any(x in t for x in ['add', 'pour', 'put', 'transfer']): return 'ADD'
        if any(x in t for x in ['stir', 'mix', 'saute', 'fry']): return 'MIX'
        if any(x in t for x in ['cook', 'heat', 'boil']): return 'COOK'
        return None

    def generate_plan(self, youtube_url):
        print(f"\n🎥 PROCESSING: {youtube_url}")
        vid_file = "temp_vla_input.mp4"
        aud_file = "temp_vla_input.mp3"
        
        try:
            download_fresh_video(youtube_url, vid_file)
            extract_audio(vid_file, aud_file)
            transcript = get_transcript(aud_file)
            chapters = detect_chapters(transcript)
            
            print(f"\n🤖 GENERATING ROBOT ACTION PLAN")
            print("=" * 60)
            
            plan = []
            for i, ch in enumerate(chapters, 1):
                if (ch['end'] - ch['start']) < 4: continue
                
                # 1. SENSE
                audio_intent = self.get_audio_intent(ch['text'])
                visual_motion = self.analyze_video_segment(vid_file, ch['start'], ch['end'])
                
                # 2. REASON (Fusion Logic)
                # If audio is explicit, trust it. If vague, trust eyes.
                action = audio_intent if audio_intent else visual_motion
                
                # 3. CONTEXT (Object Detection from Text)
                text = ch['text'].lower()
                obj = "Ingredients"
                if "onion" in text: obj = "Onion"
                elif "paneer" in text: obj = "Paneer"
                elif "tomato" in text: obj = "Tomato"
                elif "salt" in text: obj = "Salt"
                
                # 4. PLAN
                cmd = "Wait()"
                if action == 'CUT': cmd = f"SliceObject({obj})"
                elif action == 'MIX': cmd = f"Stir(Pan)"
                elif action == 'ADD': cmd = f"Transfer({obj}, Pan)"
                elif action == 'COOK': cmd = f"Cook(Pan)"
                
                print(f"STEP {i} [{ch['start']:.0f}s]: {cmd}")
                print(f"   Context: \"{ch['text'][:40]}...\"")
                print(f"   Debug:   Audio={audio_intent} | Video={visual_motion}")
                print("-" * 60)
                
                plan.append({"step": i, "command": cmd})
                
            return plan
            
        finally:
            if os.path.exists(vid_file): os.remove(vid_file)
            if os.path.exists(aud_file): os.remove(aud_file)

if __name__ == "__main__":
    brain = VLABrain()
    # The Paneer Test
    brain.generate_plan("https://www.youtube.com/watch?v=oYZ--rdHL6I")
