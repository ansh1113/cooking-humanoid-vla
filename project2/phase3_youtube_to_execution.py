"""
PHASE 3: Complete Pipeline - YouTube to AI2-THOR Execution
Watch ANY YouTube video → Predict actions → Execute in simulator
"""
import torch
import sys
from pathlib import Path
import json

# Add Project 1 path
sys.path.insert(0, '/root/vla_project/code')

# Simple tokenizer for Project 1
class SimpleLanguageTokenizer:
    def __init__(self, vocab):
        self.word_to_idx = vocab
    def encode(self, instruction, max_len=10):
        return [self.word_to_idx.get(w, 1) for w in instruction.lower().split()][:max_len]

# Import stage 1 (YouTube download + prediction)
from stage1_generate_plan import download_video, extract_frames, encode_frames_with_clip

# Import stage 2 (AI2-THOR execution)
try:
    import PROJECT1_FINAL as P1
    PROJECT1_AVAILABLE = True
except ImportError:
    PROJECT1_AVAILABLE = False
    print("⚠️  Project 1 not available - will run in prediction-only mode")

from phase2_temporal_vla_model import TemporalVLA

class YouTubeToRobotPipeline:
    """Complete pipeline from YouTube to robot execution"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained VLA model
        print("📥 Loading Temporal VLA model...")
        checkpoint = torch.load('models/temporal_vla_phase2_best.pt', map_location=self.device)
        
        self.action_vocab = checkpoint['action_vocab']
        self.id_to_action = {v: k for k, v in self.action_vocab.items()}
        
        self.model = TemporalVLA(
            embedding_dim=512,
            hidden_dim=512,
            num_actions=len(self.action_vocab),
            num_heads=8,
            num_layers=6
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded! Actions: {len(self.action_vocab)}")
    
    def predict_action(self, video_url, window_size=30):
        """
        Download YouTube video and predict cooking action
        """
        print(f"\n🎬 Processing: {video_url}")
        
        # Step 1: Download and extract frames
        print("   📥 Downloading video...")
        video_path = download_video(video_url)
        
        print("   🎞️  Extracting frames...")
        frames_dir = extract_frames(video_path)
        
        # Step 2: Encode with CLIP
        print("   🔍 Encoding with CLIP...")
        embeddings = encode_frames_with_clip(frames_dir)
        
        if embeddings is None or len(embeddings) == 0:
            print("   ❌ No embeddings generated")
            return None
        
        # Step 3: Predict with Temporal VLA
        print(f"   🧠 Predicting action ({len(embeddings)} frames)...")
        
        # Take sliding windows and average predictions
        predictions = []
        confidences = []
        
        if len(embeddings) < window_size:
            # Pad if too short
            padding = torch.zeros(window_size - len(embeddings), embeddings.shape[1])
            sequence = torch.cat([embeddings, padding], dim=0)
            sequence = sequence.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred, conf = self.model.predict_with_confidence(sequence)
            
            predictions.append(pred.item())
            confidences.append(conf.item())
        else:
            # Sliding windows
            stride = window_size // 2
            for start in range(0, len(embeddings) - window_size + 1, stride):
                end = start + window_size
                sequence = embeddings[start:end].unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    pred, conf = self.model.predict_with_confidence(sequence)
                
                predictions.append(pred.item())
                confidences.append(conf.item())
        
        # Get most confident prediction
        best_idx = confidences.index(max(confidences))
        action_id = predictions[best_idx]
        confidence = confidences[best_idx]
        
        action = self.id_to_action[action_id]
        
        return {
            'action': action,
            'action_id': action_id,
            'confidence': confidence,
            'num_windows': len(predictions),
            'avg_confidence': sum(confidences) / len(confidences)
        }
    
    def execute_in_simulator(self, action_data):
        """Execute predicted action in AI2-THOR"""
        if not PROJECT1_AVAILABLE:
            print("\n⚠️  AI2-THOR execution not available (Project 1 not imported)")
            return False
        
        action = action_data['action']
        confidence = action_data['confidence']
        
        print(f"\n🤖 Executing: {action} (confidence: {confidence:.2%})")
        
        # Parse action into verb and object
        parts = action.split()
        if len(parts) >= 2:
            verb = parts[0]  # chopping, slicing, stirring, etc.
            obj = ' '.join(parts[1:])  # onion, carrot, etc.
        else:
            verb = action
            obj = "Object"
        
        # Map to Project 1 actions
        action_mapping = {
            'chopping': 'SliceObject',
            'slicing': 'SliceObject',
            'stirring': 'PickupObject',  # Simplified
            'mixing': 'PickupObject',
            'frying': 'CookObject',
            'cooking': 'CookObject',
            'grating': 'SliceObject',
            'pouring': 'PourLiquid',
        }
        
        robot_action = action_mapping.get(verb, 'PickupObject')
        
        print(f"   🎯 Robot action: {robot_action}")
        print(f"   🎯 Target: {obj.capitalize()}")
        
        # TODO: Execute in AI2-THOR using Project 1
        # This would launch controller and execute
        
        return True

def main():
    print("="*70)
    print("🎬 PHASE 3: YOUTUBE TO ROBOT EXECUTION")
    print("="*70)
    
    # Initialize pipeline
    pipeline = YouTubeToRobotPipeline()
    
    # Test video
    print("\n📋 Test Video:")
    test_url = input("Enter YouTube URL (or press Enter for default): ").strip()
    
    if not test_url:
        test_url = "https://www.youtube.com/watch?v=G-Fg7l7G1zw"  # Knife skills
    
    # Predict action
    result = pipeline.predict_action(test_url)
    
    if result:
        print("\n" + "="*70)
        print("✅ PREDICTION RESULTS")
        print("="*70)
        print(f"   Action: {result['action']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Avg Confidence: {result['avg_confidence']:.2%}")
        print(f"   Windows analyzed: {result['num_windows']}")
        print("="*70)
        
        # Execute (if available)
        if PROJECT1_AVAILABLE:
            execute = input("\n🤖 Execute in simulator? (yes/no): ").strip().lower()
            if execute == 'yes':
                pipeline.execute_in_simulator(result)
    else:
        print("\n❌ Prediction failed")
    
    print("\n" + "="*70)
    print("🎉 PHASE 3 DEMO COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
