"""
Day 1: Visual Encoding - Convert frames to embeddings
No labels needed, just pure visual understanding
"""
import torch
import open_clip
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm

def main():
    print("="*70)
    print("👁️  DAY 1: VISUAL ENCODING PIPELINE")
    print("="*70)
    
    device = torch.device('cpu')  # Will use CPU for now
    print(f"Device: {device}")
    
    # Load CLIP as feature extractor
    print("\n📥 Loading CLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
    )
    model.eval()
    print("✅ CLIP loaded!")
    
    # Setup paths
    frame_dir = Path('data/processed/frames_all')
    output_dir = Path('data/processed/embeddings')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all video directories
    videos = sorted([d for d in frame_dir.iterdir() if d.is_dir()])
    print(f"\n📊 Found {len(videos)} videos")
    
    total_frames = 0
    
    # Process each video
    for video_dir in tqdm(videos, desc="Encoding videos"):
        video_id = video_dir.name
        frame_files = sorted(video_dir.glob('*.jpg'))
        
        if not frame_files:
            continue
        
        # Batch processing
        batch_size = 32
        video_embeddings = []
        
        for i in range(0, len(frame_files), batch_size):
            batch_files = frame_files[i:i+batch_size]
            
            # Load and preprocess images
            images = []
            for f in batch_files:
                try:
                    img = Image.open(f).convert('RGB')
                    images.append(preprocess(img))
                except:
                    continue
            
            if not images:
                continue
            
            image_tensor = torch.stack(images).to(device)
            
            # Extract embeddings
            with torch.no_grad():
                features = model.encode_image(image_tensor)
                # Normalize (crucial for matching later)
                features = features / features.norm(dim=-1, keepdim=True)
            
            video_embeddings.append(features.cpu())
        
        # Save embeddings
        if video_embeddings:
            full_tensor = torch.cat(video_embeddings)
            save_path = output_dir / f"{video_id}.pt"
            torch.save(full_tensor, save_path)
            total_frames += len(frame_files)
    
    print("\n" + "="*70)
    print(f"✅ Encoded {total_frames} frames from {len(videos)} videos")
    print(f"💾 Saved to: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
