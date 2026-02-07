"""
PHASE 2: Build Training Dataset from GPT-4V Labels
"""
import json
import torch
from pathlib import Path
import numpy as np
from collections import Counter

def load_labels():
    """Load GPT-4V labels"""
    with open('data/processed/gpt4v_labels_enhanced.json', 'r') as f:
        return json.load(f)

def load_embeddings(video_id):
    """Load CLIP embeddings for a video"""
    emb_path = Path(f'data/processed/embeddings/{video_id}.pt')
    if emb_path.exists():
        return torch.load(emb_path)
    return None

def build_action_vocabulary(labels):
    """Build action vocabulary from labels"""
    # Get all unique actions
    actions = [l['label'] for l in labels]
    action_counts = Counter(actions)
    
    # Keep actions with at least 2 examples
    vocab = {action: idx for idx, (action, count) in enumerate(action_counts.items()) if count >= 2}
    
    print(f"\n📊 Action Vocabulary:")
    print(f"   Total unique actions: {len(action_counts)}")
    print(f"   Actions with 2+ examples: {len(vocab)}")
    
    return vocab, action_counts

def create_training_sequences(labels, action_vocab, window_size=30, stride=15):
    """Create training sequences"""
    sequences = []
    
    print(f"\n🎬 Creating training sequences...")
    print(f"   Window size: {window_size} frames")
    print(f"   Stride: {stride} frames")
    
    for label_data in labels:
        video_id = label_data['video_id']
        action = label_data['label']
        confidence = label_data.get('confidence', 'medium')
        
        # Skip low confidence
        if confidence == 'low':
            continue
        
        # Skip if action not in vocab
        if action not in action_vocab:
            continue
        
        # Load embeddings
        embeddings = load_embeddings(video_id)
        if embeddings is None:
            continue
        
        # Create sequences with sliding window
        num_frames = len(embeddings)
        
        if num_frames < window_size:
            # If video is shorter, just use the whole thing
            sequences.append({
                'embeddings': embeddings,
                'action': action,
                'action_id': action_vocab[action],
                'video_id': video_id,
                'confidence': confidence
            })
        else:
            # Sliding window
            for start in range(0, num_frames - window_size + 1, stride):
                end = start + window_size
                sequences.append({
                    'embeddings': embeddings[start:end],
                    'action': action,
                    'action_id': action_vocab[action],
                    'video_id': video_id,
                    'confidence': confidence
                })
    
    return sequences

def main():
    print("="*70)
    print("📊 PHASE 2: BUILD TRAINING DATASET")
    print("="*70)
    
    # Load labels
    labels = load_labels()
    print(f"\n✅ Loaded {len(labels)} labeled videos")
    
    # Build vocabulary
    action_vocab, action_counts = build_action_vocabulary(labels)
    
    # Show top actions
    print(f"\n🔥 Top 20 Actions:")
    for i, (action, count) in enumerate(action_counts.most_common(20), 1):
        in_vocab = "✓" if action in action_vocab else "✗"
        print(f"   {i:2d}. [{in_vocab}] {action:35s}: {count:3d} videos")
    
    # Create sequences
    sequences = create_training_sequences(labels, action_vocab)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total sequences: {len(sequences)}")
    print(f"   Unique actions: {len(action_vocab)}")
    
    # Split into train/val/test
    np.random.seed(42)
    np.random.shuffle(sequences)
    
    n_train = int(0.8 * len(sequences))
    n_val = int(0.1 * len(sequences))
    
    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:n_train+n_val]
    test_seqs = sequences[n_train+n_val:]
    
    print(f"\n📊 Split:")
    print(f"   Train: {len(train_seqs)} sequences")
    print(f"   Val:   {len(val_seqs)} sequences")
    print(f"   Test:  {len(test_seqs)} sequences")
    
    # Save dataset
    dataset = {
        'action_vocab': action_vocab,
        'train': train_seqs,
        'val': val_seqs,
        'test': test_seqs
    }
    
    torch.save(dataset, 'data/processed/training_dataset_phase2.pt')
    
    # Save vocab separately
    with open('data/processed/action_vocab_phase2.json', 'w') as f:
        json.dump(action_vocab, f, indent=2)
    
    print(f"\n💾 Saved:")
    print(f"   data/processed/training_dataset_phase2.pt")
    print(f"   data/processed/action_vocab_phase2.json")
    
    print("\n" + "="*70)
    print("✅ DATASET READY FOR TRAINING!")
    print("="*70)

if __name__ == "__main__":
    main()
