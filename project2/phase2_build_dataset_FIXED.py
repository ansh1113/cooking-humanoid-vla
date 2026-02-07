"""
PHASE 2: Build Training Dataset (FIXED - Proper action ID mapping)
"""
import json
import torch
from pathlib import Path
import numpy as np
from collections import Counter

def load_labels():
    """Load GPT-4V labels"""
    with open('data/processed/gpt4v_labels_enhanced.json', 'r') as f:
        labels = json.load(f)
    return [l for l in labels if l.get('label')]

def load_embeddings(video_id):
    """Load CLIP embeddings for a video"""
    emb_path = Path(f'data/processed/embeddings/{video_id}.pt')
    if emb_path.exists():
        return torch.load(emb_path, weights_only=False)
    return None

def build_action_vocabulary(labels):
    """Build action vocabulary with CONTINUOUS IDs starting from 0"""
    actions = [l['label'] for l in labels if l.get('label')]
    action_counts = Counter(actions)
    
    # Keep actions with at least 2 examples
    valid_actions = [action for action, count in action_counts.items() if count >= 2]
    
    # Create vocabulary with CONTINUOUS IDs from 0 to N-1
    vocab = {action: idx for idx, action in enumerate(sorted(valid_actions))}
    
    print(f"\n📊 Action Vocabulary:")
    print(f"   Total unique actions: {len(action_counts)}")
    print(f"   Actions with 2+ examples: {len(vocab)}")
    print(f"   Action IDs: 0 to {len(vocab)-1}")
    
    return vocab, action_counts

def create_training_sequences(labels, action_vocab, window_size=30, stride=15):
    """Create training sequences"""
    sequences = []
    
    print(f"\n🎬 Creating training sequences...")
    print(f"   Window size: {window_size} frames")
    print(f"   Stride: {stride} frames")
    
    skipped_no_embeddings = 0
    skipped_not_in_vocab = 0
    skipped_low_confidence = 0
    
    for label_data in labels:
        video_id = label_data['video_id']
        action = label_data.get('label')
        confidence = label_data.get('confidence', 'medium')
        
        if not action:
            continue
        
        # Skip low confidence
        if confidence == 'low':
            skipped_low_confidence += 1
            continue
        
        # Skip if action not in vocab
        if action not in action_vocab:
            skipped_not_in_vocab += 1
            continue
        
        # Load embeddings
        embeddings = load_embeddings(video_id)
        if embeddings is None:
            skipped_no_embeddings += 1
            continue
        
        action_id = action_vocab[action]
        
        # Verify action_id is valid
        assert 0 <= action_id < len(action_vocab), f"Invalid action_id {action_id} for action '{action}'"
        
        # Create sequences with sliding window
        num_frames = len(embeddings)
        
        if num_frames < window_size:
            sequences.append({
                'embeddings': embeddings,
                'action': action,
                'action_id': action_id,
                'video_id': video_id,
                'confidence': confidence
            })
        else:
            for start in range(0, num_frames - window_size + 1, stride):
                end = start + window_size
                sequences.append({
                    'embeddings': embeddings[start:end],
                    'action': action,
                    'action_id': action_id,
                    'video_id': video_id,
                    'confidence': confidence
                })
    
    print(f"\n⚠️  Skipped:")
    print(f"   No embeddings: {skipped_no_embeddings}")
    print(f"   Not in vocab: {skipped_not_in_vocab}")
    print(f"   Low confidence: {skipped_low_confidence}")
    
    return sequences

def main():
    print("="*70)
    print("📊 PHASE 2: BUILD TRAINING DATASET (FIXED)")
    print("="*70)
    
    labels = load_labels()
    print(f"\n✅ Loaded {len(labels)} labeled videos")
    
    vocab, action_counts = build_action_vocabulary(labels)
    
    print(f"\n🔥 Actions in Vocabulary:")
    for action, action_id in sorted(vocab.items(), key=lambda x: x[1]):
        count = action_counts[action]
        print(f"   ID {action_id:2d}: {action:35s} ({count} videos)")
    
    sequences = create_training_sequences(labels, vocab)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total sequences: {len(sequences)}")
    print(f"   Unique actions: {len(vocab)}")
    
    if len(sequences) == 0:
        print("\n❌ ERROR: No sequences created!")
        return
    
    # Verify all action_ids are valid
    all_action_ids = [s['action_id'] for s in sequences]
    min_id = min(all_action_ids)
    max_id = max(all_action_ids)
    print(f"\n✅ Action ID range: {min_id} to {max_id}")
    print(f"   Expected: 0 to {len(vocab)-1}")
    assert min_id >= 0 and max_id < len(vocab), "Action IDs out of range!"
    
    # Split
    np.random.seed(42)
    np.random.shuffle(sequences)
    
    n_train = int(0.8 * len(sequences))
    n_val = int(0.1 * len(sequences))
    
    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:n_train+n_val]
    test_seqs = sequences[n_train+n_val:]
    
    print(f"\n📊 Split:")
    print(f"   Train: {len(train_seqs)}")
    print(f"   Val:   {len(val_seqs)}")
    print(f"   Test:  {len(test_seqs)}")
    
    dataset = {
        'action_vocab': vocab,
        'train': train_seqs,
        'val': val_seqs,
        'test': test_seqs
    }
    
    torch.save(dataset, 'data/processed/training_dataset_phase2_fixed.pt')
    
    with open('data/processed/action_vocab_phase2_fixed.json', 'w') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"\n💾 Saved:")
    print(f"   data/processed/training_dataset_phase2_fixed.pt")
    print(f"   data/processed/action_vocab_phase2_fixed.json")
    
    print("\n" + "="*70)
    print("✅ DATASET READY!")
    print("="*70)

if __name__ == "__main__":
    main()
