"""
extract_sequences.py - Extract action sequences from CLIP predictions
Discovers common manipulation patterns from cooking videos
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_clip_predictions():
    """Load CLIP frame classifications"""
    with open('data/processed/clip_predictions.json', 'r') as f:
        return json.load(f)

def extract_action_sequences(video_predictions, min_duration=2.0, max_gap=3.0):
    """
    Extract action sequences from video predictions
    
    Args:
        video_predictions: List of frame predictions with timestamps
        min_duration: Minimum duration (seconds) for an action to be valid
        max_gap: Maximum gap (seconds) between actions in a sequence
    
    Returns:
        List of action sequences
    """
    
    # Group consecutive frames by action
    action_segments = []
    current_action = None
    segment_start = None
    segment_frames = []
    
    for pred in sorted(video_predictions, key=lambda x: x['timestamp']):
        action = pred['predicted_action']
        timestamp = pred['timestamp']
        
        if action != current_action:
            # Save previous segment if valid
            if current_action and segment_frames:
                duration = segment_frames[-1]['timestamp'] - segment_frames[0]['timestamp']
                if duration >= min_duration:
                    action_segments.append({
                        'action': current_action,
                        'start_time': segment_frames[0]['timestamp'],
                        'end_time': segment_frames[-1]['timestamp'],
                        'duration': duration,
                        'num_frames': len(segment_frames)
                    })
            
            # Start new segment
            current_action = action
            segment_start = timestamp
            segment_frames = [pred]
        else:
            segment_frames.append(pred)
    
    # Save last segment
    if current_action and segment_frames:
        duration = segment_frames[-1]['timestamp'] - segment_frames[0]['timestamp']
        if duration >= min_duration:
            action_segments.append({
                'action': current_action,
                'start_time': segment_frames[0]['timestamp'],
                'end_time': segment_frames[-1]['timestamp'],
                'duration': duration,
                'num_frames': len(segment_frames)
            })
    
    # Build sequences from segments
    sequences = []
    current_sequence = []
    
    for i, segment in enumerate(action_segments):
        if not current_sequence:
            current_sequence.append(segment)
        else:
            # Check gap to previous segment
            gap = segment['start_time'] - current_sequence[-1]['end_time']
            
            if gap <= max_gap:
                current_sequence.append(segment)
            else:
                # Save current sequence if it has multiple actions
                if len(current_sequence) >= 2:
                    sequences.append(current_sequence)
                # Start new sequence
                current_sequence = [segment]
    
    # Save last sequence
    if len(current_sequence) >= 2:
        sequences.append(current_sequence)
    
    return sequences

def sequence_to_string(sequence):
    """Convert sequence to readable string"""
    return ' → '.join([seg['action'] for seg in sequence])

def find_common_patterns(all_sequences, min_occurrences=2):
    """Find common action patterns across videos"""
    
    # Convert sequences to strings for counting
    pattern_counts = defaultdict(int)
    pattern_examples = defaultdict(list)
    
    for video_id, sequences in all_sequences.items():
        for seq in sequences:
            pattern = sequence_to_string(seq)
            pattern_counts[pattern] += 1
            pattern_examples[pattern].append({
                'video_id': video_id,
                'sequence': seq
            })
    
    # Filter to common patterns
    common_patterns = {
        pattern: {
            'count': count,
            'examples': pattern_examples[pattern][:3]  # Keep 3 examples
        }
        for pattern, count in pattern_counts.items()
        if count >= min_occurrences
    }
    
    return common_patterns

def main():
    print("="*70)
    print("🔗 EXTRACTING ACTION SEQUENCES")
    print("="*70)
    
    # Load CLIP predictions
    print("\n📥 Loading CLIP predictions...")
    clip_predictions = load_clip_predictions()
    print(f"✅ Loaded predictions for {len(clip_predictions)} videos")
    
    # Extract sequences from each video
    print("\n🔍 Extracting sequences...")
    all_sequences = {}
    total_sequences = 0
    
    for video_id, predictions in clip_predictions.items():
        if not predictions:
            continue
        
        sequences = extract_action_sequences(predictions)
        
        if sequences:
            all_sequences[video_id] = sequences
            total_sequences += len(sequences)
            
            print(f"   {video_id}: {len(sequences)} sequences")
            # Show examples
            for seq in sequences[:2]:
                print(f"      • {sequence_to_string(seq)}")
    
    # Find common patterns
    print(f"\n🔎 Finding common patterns...")
    common_patterns = find_common_patterns(all_sequences, min_occurrences=2)
    
    print(f"\n📊 Found {len(common_patterns)} common patterns:")
    for pattern, data in sorted(common_patterns.items(), key=lambda x: -x[1]['count'])[:10]:
        print(f"   [{data['count']}x] {pattern}")
    
    # Save results
    output = {
        'sequences_by_video': {
            video_id: [
                {
                    'pattern': sequence_to_string(seq),
                    'segments': seq
                }
                for seq in sequences
            ]
            for video_id, sequences in all_sequences.items()
        },
        'common_patterns': common_patterns,
        'statistics': {
            'total_videos': len(all_sequences),
            'total_sequences': total_sequences,
            'unique_patterns': len(set(sequence_to_string(seq) for seqs in all_sequences.values() for seq in seqs)),
            'common_patterns': len(common_patterns)
        }
    }
    
    output_file = Path('data/processed/action_sequences.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✅ Extracted {total_sequences} sequences from {len(all_sequences)} videos")
    print(f"📊 {len(common_patterns)} common patterns discovered")
    print(f"💾 Saved to: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()
